"""Independent pretrained detector inference; scores are not calibrated probabilities.

DeepfakeBench Xception (CC-BY-NC-4.0) and NAVER AASIST (MIT). Architecture,
licenses, exact commits and checksums are retained in assets/provenance.json.
No sender effect flags or swapper identity scores enter these models.
"""
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly
from ml.training.face_preprocessing import RealFacePreprocessor, FaceDetectionError

ASSETS = Path(__file__).parent / 'assets'
PINNED = {
    'xception_best.pth': 'dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47',
    'AASIST.pth': '51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0',
    'aasist_upstream.py': '9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50',
}

def load_architecture(name):
    manifest = json.loads((ASSETS / 'provenance.json').read_text())
    path = ASSETS / name
    expected = PINNED.get(name, manifest['sha256'][name])
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Detector asset checksum mismatch: {name}')
    spec = importlib.util.spec_from_file_location('deeptrace_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def state_dict(name):
    path = ASSETS / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED[name]:
        raise ValueError(f'Detector weight checksum mismatch: {name}')
    return torch.load(path, map_location='cpu', weights_only=True)

class PretrainedDetectors:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.face_model_name = 'DeepfakeBench Xception'
        self.face = load_architecture('xception_inference.py').Xception(
            dict(num_classes=2, mode='original', inc=3, dropout=False))
        self.face.load_state_dict({k.removeprefix('backbone.'): v for k, v in state_dict('xception_best.pth').items()}, strict=True)
        self.face.to(self.device).eval().requires_grad_(False)
        config = json.loads((ASSETS / 'AASIST.conf').read_text())['model_config']
        self.audio = load_architecture('aasist_upstream.py').Model(config)
        self.audio.load_state_dict(state_dict('AASIST.pth'), strict=True)
        self.audio.to(self.device).eval().requires_grad_(False)
        self.preprocessor = RealFacePreprocessor(image_size=256)

    def face_box(self, image):
        detection = self.preprocessor.detect_landmarks(image)
        points = detection.contour_landmarks
        lower, upper = points.min(0), points.max(0)
        center = (lower + upper) / 2
        half = max(upper - lower) * .65
        box = [int(center[0]-half), int(center[1]-half), int(center[0]+half), int(center[1]+half)]
        box = [max(0,box[0]),max(0,box[1]),min(image.width,box[2]),min(image.height,box[3])]
        if min(box[2]-box[0],box[3]-box[1]) < 48:
            raise ValueError('Face too small to analyze reliably.')
        if detection.num_faces_detected != 1:
            raise ValueError('A single visible face is required.')
        return box

    @torch.inference_mode()
    def video_score(self, image, box=None):
        started = time.perf_counter()
        try:
            box = self.face_box(image) if box is None else box
        except (FaceDetectionError, ValueError) as error:
            return dict(status='insufficient', reason=str(error), score=None, model=self.face_model_name)
        crop = image.convert('RGB').crop(box).resize((256,256), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(crop,dtype=np.float32).copy()).permute(2,0,1).unsqueeze(0).to(self.device) / 127.5 - 1
        logits, _ = self.face(tensor)
        score = float(logits.softmax(1)[0,1])
        if not math.isfinite(score): raise ValueError('Nonfinite face detector output.')
        return dict(status='measured', score=score, model=self.face_model_name, box=box,
                    pipeline_ms=(time.perf_counter()-started)*1000, calibrated=False)

    @torch.inference_mode()
    def audio_score(self, pcm, sample_rate):
        pcm = np.asarray(pcm,dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all() or np.max(np.abs(pcm),initial=0)>1.01:
            raise ValueError('Expected finite mono PCM in [-1,1].')
        if not 8000 <= sample_rate <= 96000: raise ValueError('Unsupported sample rate.')
        if not 3 <= len(pcm)/sample_rate <= 6: raise ValueError('Analyze 3–6 seconds of audio.')
        rms = float(np.sqrt(np.mean(pcm**2)))
        if rms < .005:
            return dict(status='insufficient', reason='Insufficient audible speech; silence is not genuine evidence.', score=None, model='AASIST', rms=rms)
        gcd = math.gcd(sample_rate,16000)
        pcm = resample_poly(pcm,16000//gcd,sample_rate//gcd).astype(np.float32)
        # Match upstream: first 64600 samples; repeat short clips, no peak normalization.
        pcm = np.tile(pcm,math.ceil(64600/len(pcm)))[:64600]
        started=time.perf_counter()
        _, logits=self.audio(torch.from_numpy(pcm.copy()).unsqueeze(0).to(self.device))
        score=float(logits.softmax(1)[0,0]) # Upstream: spoof=0, bona fide=1.
        if not math.isfinite(score): raise ValueError('Nonfinite audio detector output.')
        return dict(status='measured',score=score,model='AASIST',rms=rms,
                    pipeline_ms=(time.perf_counter()-started)*1000,calibrated=False)

    def close(self):
        self.preprocessor.detector._landmarker.close() if self.preprocessor.detector._landmarker else None
