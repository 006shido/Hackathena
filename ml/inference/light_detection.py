"""Retry geometry detection on dark frames; synthesis still uses original RGB."""
import cv2
import numpy as np
from PIL import Image
from ml.training.face_preprocessing import NoFaceDetectedError


def detect_with_light_retry(preprocessor, image, enabled=True):
    try:
        return preprocessor.detect_landmarks(image), False
    except NoFaceDetectedError:
        if not enabled:raise
        rgb=np.asarray(image.convert('RGB'))
        luminance=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        # Avoid enhancing normally lit occlusions into false faces.
        if float(np.median(luminance))>=60:raise
        lookup=np.round(255*np.sqrt(np.arange(256,dtype=np.float32)/255)).astype(np.uint8)
        enhanced=cv2.LUT(rgb,lookup)
        return preprocessor.detect_landmarks(Image.fromarray(enhanced)), True
