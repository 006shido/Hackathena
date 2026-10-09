"""Verify identity correspondence and rejection of an inverted mesh."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from ml.training.face_preprocessing import RealFacePreprocessor
from ml.inference.face_correspondence import preprocess_single_face
from ml.training.robust_correspondence import robust_warp, reliable_legacy_warp

def main():
    pre=RealFacePreprocessor(image_size=128)
    image, tensor, _, _, dense, _=preprocess_single_face(pre,str(ROOT/'ml/data/celeba/img_align_celeba/197935.jpg'))
    warped,confidence,report=robust_warp(image,dense,dense)
    assert np.abs(warped.numpy()[0]-tensor).mean()<0.01
    assert np.isfinite(warped.numpy()).all()
    assert confidence.min()>=0 and confidence.max()<=1
    inverted=dense.copy();inverted[:,0]=127-inverted[:,0]
    _,bad_conf,bad_report=robust_warp(image,inverted,dense)
    assert bad_report['flipped_triangles']>100
    assert bad_report['unreliable_pixel_fraction']>report['unreliable_pixel_fraction']
    assert bad_conf.mean()<confidence.mean()
    masked, gated, _ = reliable_legacy_warp(image, inverted, dense)
    invalid = bad_conf.numpy()[0,0] == 0
    assert invalid.any()
    assert np.all(masked.numpy()[0,:,invalid] == 0)
    assert np.all(gated.numpy()[0,0,invalid] == 0)
    print('PASS: identity mapping preserved; inverted mesh suppressed.')

if __name__=='__main__':main()
