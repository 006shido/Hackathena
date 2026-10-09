"""Offline video acceptance probe for the research backend; no service defaults."""
import time
import cv2
import numpy as np
from ml.inference.frame_pipeline import restore_face
from ml.inference.light_detection import detect_with_light_retry
from ml.inference.identity_guard import IdentityGuard
from ml.training.face_preprocessing import estimate_umeyama_similarity_transform,get_canonical_landmarks
from ml.training.face_preprocessing import FaceDetectionError, generate_facial_mask, MEDIAPIPE_FACE_OVAL_INDICES


class ReferenceFrameSession:
    def __init__(self, engine, source, correspondence=None):
        self.engine=engine
        self.timestamp=None
        detection=engine.preprocessor.detect_landmarks(source.convert('RGB'))
        self.latent=engine.reference.source_identity(np.asarray(source.convert('RGB')),detection.key_landmarks_5pts)
        self.guard=IdentityGuard(engine.identity_model,source.convert('RGB'),detection.key_landmarks_5pts) if getattr(engine,'identity_guard',False) else None
        if getattr(engine,'occlusion_model',None) is not None and self.guard is None:
            raise ValueError('Experimental occlusion composition requires the unchanged identity guard.')
        self.preprocessor=engine.preprocessor
        self.video_detector=None
        self.closed=False
        self.occlusion_state={} if getattr(engine,'occlusion_temporal',False) else None
        if getattr(engine,'video_landmarks',False):
            from ml.training.face_preprocessing import RealFacePreprocessor
            from ml.experiments.reference_backend.video_landmarks import VideoLandmarkDetector
            self.preprocessor=RealFacePreprocessor(image_size=128,feather_radius=engine.preprocessor.feather_radius)
            self.video_detector=VideoLandmarkDetector()
            self.preprocessor.detector=self.video_detector

    def close(self):
        if self.video_detector:self.video_detector.close()
        if self.occlusion_state is not None:self.occlusion_state.clear()
        self.closed=True

    def process(self,frame,timestamp_ms):
        if self.closed:raise ValueError('Video session is closed.')
        if not np.isfinite(timestamp_ms) or (self.timestamp is not None and timestamp_ms<=self.timestamp):
            raise ValueError('Frame timestamps must be finite and strictly increasing.')
        self.timestamp=timestamp_ms
        if self.video_detector:self.video_detector.set_timestamp(timestamp_ms)
        start=time.perf_counter()
        frame=frame.convert('RGB'); original=np.asarray(frame)
        try:
            detection,retried=detect_with_light_retry(self.preprocessor,frame,
                enabled=getattr(self.engine,'lowlight_retry',False))
            swap,affine=self.engine.reference.swap(original,detection.key_landmarks_5pts,self.latent)
            dense=cv2.transform(detection.dense_landmarks[None].astype(np.float32),affine)[0]
            alpha=generate_facial_mask(dense[MEDIAPIPE_FACE_OVAL_INDICES],128,self.engine.preprocessor.feather_radius)
            output,opacity=restore_face(original,swap,alpha,affine)
            diagnostic=dict(face_detected=True,changed_pixels=int((opacity>0).sum()),lowlight_retry=retried)
            if getattr(self.engine,'occlusion_model',None) is not None:
                from ml.experiments.reference_backend.occlusion_composite import preserve_occluders
                output,occlusion=preserve_occluders(original,output,affine,self.engine.occlusion_model,getattr(self.engine,'occlusion_feather',0),self.occlusion_state,getattr(self.engine,'semantic_model',None))
                diagnostic.update(occlusion)
                diagnostic['changed_pixels']=int(np.any(output!=original,axis=2).sum())
            if getattr(self.engine,'capture_geometry',False):
                diagnostic['key_landmarks_5pts']=detection.key_landmarks_5pts.tolist()
            if self.guard:
                common=estimate_umeyama_similarity_transform(detection.key_landmarks_5pts,get_canonical_landmarks(128))
                target=cv2.warpAffine(original,common,(128,128))
                generated=cv2.warpAffine(output,common,(128,128))
                decision=self.guard.check(target,generated)
                diagnostic.update(decision)
                if not decision['identity_accepted']:
                    output=original.copy();diagnostic['changed_pixels']=0
                    diagnostic['reason']='Source identity gate rejected this frame.'
        except FaceDetectionError as error:
            if self.occlusion_state is not None:self.occlusion_state.clear()
            output=original.copy();diagnostic=dict(face_detected=False,changed_pixels=0,reason=str(error))
        diagnostic['pipeline_ms']=1000*(time.perf_counter()-start)
        diagnostic['face_changed']=diagnostic['changed_pixels']>0
        return output,diagnostic
