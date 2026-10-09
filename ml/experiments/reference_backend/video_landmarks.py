"""Experimental per-video MediaPipe tracking; never shared across call sessions."""
from pathlib import Path
import math
from ml.training.face_preprocessing import MediaPipeLandmarkDetector

class VideoLandmarkDetector(MediaPipeLandmarkDetector):
    def __init__(self,frame_interval_ms=40,model_asset_path='client/public/models/face_landmarker.task'):
        super().__init__(model_asset_path)
        if not math.isfinite(frame_interval_ms) or frame_interval_ms<1:raise ValueError('Invalid video frame interval.')
        self.frame_interval_ms=frame_interval_ms
        self.frame_index=0
        self.task=None
        self.pending_timestamp=None
        self.last_timestamp=None
        self.closed=False

    def set_timestamp(self,timestamp_ms):
        if self.closed:raise ValueError('Video detector is closed.')
        if not math.isfinite(timestamp_ms) or timestamp_ms<0:raise ValueError('Invalid tracking timestamp.')
        self.pending_timestamp=round(timestamp_ms)

    def _get_landmarker(self):
        if self.closed:raise ValueError('Video detector is closed.')
        if self._landmarker is not None:return self._landmarker
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model=Path(self.model_asset_path)
        if not model.exists():model=Path(__file__).resolve().parents[3]/self.model_asset_path
        options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,min_tracking_confidence=self.min_detection_confidence)
        self.task=vision.FaceLandmarker.create_from_options(options)
        owner=self
        class ImageInterface:
            def detect(self,image):
                timestamp=owner.pending_timestamp if owner.pending_timestamp is not None else round(owner.frame_index*owner.frame_interval_ms)
                if owner.last_timestamp is not None:timestamp=max(timestamp,owner.last_timestamp+1)
                owner.last_timestamp=timestamp;owner.pending_timestamp=None
                owner.frame_index+=1
                return owner.task.detect_for_video(image,timestamp)
        self._landmarker=ImageInterface()
        return self._landmarker

    def close(self):
        if self.task is not None:self.task.close();self.task=None
        self._landmarker=None
        self.closed=True
