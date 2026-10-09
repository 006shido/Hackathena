"""In-memory neural frame processing for video evaluation and integration.

One session owns one source identity and tracking state. Never share a
session between calls. This module does not change the live service default.
"""
import time
import cv2
import numpy as np
import torch
from PIL import Image
from ml.training.face_preprocessing import (
    align_face_similarity, estimate_umeyama_similarity_transform,
    get_canonical_landmarks, generate_facial_mask, generate_landmark_map,
    normalize_image_tensor, MEDIAPIPE_FACE_OVAL_INDICES, FaceDetectionError,
)
from ml.inference.face_correspondence import warp_and_prepare_source
from ml.training.robust_correspondence import visibility_warp


def restore_face(frame, swap, alpha, alignment):
    """Inverse-project premultiplied face color; keep all outside pixels exact."""
    frame = np.asarray(frame, dtype=np.uint8)
    swap = np.asarray(swap, dtype=np.float32)
    alpha = np.clip(np.asarray(alpha, dtype=np.float32).squeeze(),0,1)
    if swap.shape != (128,128,3) or alpha.shape != (128,128):
        raise ValueError('Expected aligned 128px swap and alpha.')
    inverse = cv2.invertAffineTransform(np.asarray(alignment,dtype=np.float32))
    dimensions = (frame.shape[1],frame.shape[0])
    opacity = cv2.warpAffine(alpha,inverse,dimensions,flags=cv2.INTER_LINEAR)
    color = cv2.warpAffine(swap*alpha[...,None],inverse,dimensions,flags=cv2.INTER_LINEAR)
    output = frame.copy()
    active = opacity > 0
    mixed = color + frame.astype(np.float32)*(1-opacity[...,None])
    output[active] = np.clip(np.rint(mixed[active]),0,255).astype(np.uint8)
    return output, opacity


class NeuralFrameSession:
    def __init__(self, engine, source: Image.Image, correspondence='legacy'):
        if correspondence not in ('legacy','visibility'):
            raise ValueError('Unsupported video correspondence')
        self.engine = engine
        self.correspondence = correspondence
        det = engine.preprocessor.detect_landmarks(source.convert('RGB'))
        self.source, _, self.source_dense = align_face_similarity(source.convert('RGB'),det.key_landmarks_5pts,
            target_size=128,extra_landmarks=det.dense_landmarks)
        tensor = torch.from_numpy(normalize_image_tensor(self.source)).unsqueeze(0).to(engine.device)
        with torch.inference_mode(): self.identity = engine.model.arcface(tensor)
        self.previous_landmarks = None
        self.timestamp = None

    def process(self, frame: Image.Image, timestamp_ms: float):
        if not np.isfinite(timestamp_ms) or (self.timestamp is not None and timestamp_ms <= self.timestamp):
            raise ValueError('Frame timestamps must be finite and strictly increasing.')
        self.timestamp = timestamp_ms
        started = time.perf_counter()
        frame = frame.convert('RGB')
        original = np.asarray(frame)
        try:
            det = self.engine.preprocessor.detect_landmarks(frame)
            dense = det.dense_landmarks
            if self.previous_landmarks is not None:
                motion = np.linalg.norm(dense-self.previous_landmarks,axis=1).mean()
                # Smooth only small tracking jitter; avoid lag on large head motion.
                if motion < 0.01*min(frame.size):
                    dense = 0.65*dense + 0.35*self.previous_landmarks
            self.previous_landmarks = dense.copy()
            alignment = estimate_umeyama_similarity_transform(det.key_landmarks_5pts,get_canonical_landmarks(128))
            target, _, target_dense = align_face_similarity(frame,det.key_landmarks_5pts,128,dense)
            mask = generate_facial_mask(target_dense[MEDIAPIPE_FACE_OVAL_INDICES],128,self.engine.preprocessor.feather_radius)
            landmarks = generate_landmark_map(target_dense,128)
            warp = warp_and_prepare_source if self.correspondence == 'legacy' else lambda *a: visibility_warp(*a,strength=0.25)[:2]
            aligned, confidence = warp(self.source,self.source_dense,target_dense,128)
            device, model = self.engine.device,self.engine.model
            target_tensor = torch.from_numpy(normalize_image_tensor(target)).unsqueeze(0).to(device)
            landmark_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)
            with torch.inference_mode():
                features = model.target_encoder(model.prepare_target_input(target_tensor,landmark_tensor))
                skips = model.source_encoder(aligned.to(device))
                swap, predicted_mask = model.generator(f_tgt=features,z_id=self.identity,
                    source_skips=skips,confidence_map=confidence.to(device),disable_skips=False)
                rgb = ((swap[0].permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255)
                # Restrict generated opacity to the detected face region.
                alpha = predicted_mask[0,0].cpu().numpy()*mask.squeeze()
            output, opacity = restore_face(original,rgb,alpha,alignment)
            diagnostic = dict(face_detected=True,changed_pixels=int((opacity>0).sum()))
        except FaceDetectionError as error:
            self.previous_landmarks = None
            output = original.copy()
            diagnostic = dict(face_detected=False,changed_pixels=0,reason=str(error))
        diagnostic['pipeline_ms'] = 1000*(time.perf_counter()-started)
        return output, diagnostic
