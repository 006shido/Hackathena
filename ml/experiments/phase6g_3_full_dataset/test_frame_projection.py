"""Check inverse face projection, alpha semantics, and background preservation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from ml.inference.frame_pipeline import restore_face


def main():
    rng=np.random.default_rng(47)
    frame=rng.integers(0,256,(192,256,3),dtype=np.uint8)
    swap=np.full((128,128,3),200,dtype=np.float32)
    alpha=np.zeros((128,128),dtype=np.float32);alpha[20:100,20:100]=0.5
    alignment=np.array([[1,0,-50],[0,1,-30]],dtype=np.float32)
    output,opacity=restore_face(frame,swap,alpha,alignment)
    assert np.array_equal(output[opacity==0],frame[opacity==0])
    expected=np.rint(100+0.5*frame[70,90]).astype(np.uint8)
    assert np.array_equal(output[70,90],expected)
    unchanged,_=restore_face(frame,swap,np.zeros_like(alpha),alignment)
    assert np.array_equal(unchanged,frame)
    assert output.shape==frame.shape and output.dtype==np.uint8
    print('PASS: correct inverse translation, single alpha blend, exact background preservation.')


if __name__=='__main__':main()
