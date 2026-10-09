"""Pixel preservation checks for experimental binary occlusion composition."""
import numpy as np
import torch
from ml.experiments.reference_backend.occlusion_composite import preserve_occluders

class FixedMask:
    def __init__(self,mask):self.mask=torch.from_numpy(mask)
    def __call__(self,input):
        assert input.shape==(1,256,256,3)
        return self.mask

def main():
    rng=np.random.default_rng(42);original=rng.integers(0,200,(64,64,3),dtype=np.uint8)
    baseline=original.copy();baseline[8:56,8:56]=255
    affine=np.array([[1,0,0],[0,1,0]],np.float32)
    visible,_=preserve_occluders(original,baseline,affine,FixedMask(np.ones((256,256),np.float32)))
    assert np.array_equal(visible,baseline)
    hidden,_=preserve_occluders(original,baseline,affine,FixedMask(np.zeros((256,256),np.float32)))
    assert np.array_equal(hidden,original)
    mask=np.ones((256,256),np.float32);mask[:,64:]=0
    mixed,_=preserve_occluders(original,baseline,affine,FixedMask(mask))
    assert np.array_equal(mixed[:,40:],original[:,40:])
    assert np.array_equal(mixed[:,:24],baseline[:,:24])
    untouched=np.all(baseline==original,axis=2)
    assert np.array_equal(mixed[untouched],original[untouched])
    softened,_=preserve_occluders(original,baseline,affine,FixedMask(mask),feather=3)
    assert np.array_equal(softened[:,40:],original[:,40:])
    assert np.array_equal(softened[untouched],original[untouched])
    state={}
    preserve_occluders(original,baseline,affine,FixedMask(mask),state=state)
    revealed,_=preserve_occluders(original,baseline,affine,FixedMask(np.ones((256,256),np.float32)),state=state)
    assert np.all(revealed[20,45]>original[20,45]) and np.all(revealed[20,45]<baseline[20,45])
    covered,_=preserve_occluders(original,baseline,affine,FixedMask(mask),state=state)
    assert np.array_equal(covered[:,40:],original[:,40:])
    assert np.array_equal(covered[untouched],original[untouched])
    class HairLabels:
        def __call__(self,input):
            assert input.shape==(1,3,512,512)
            logits=torch.zeros((1,19,512,512));logits[:,17]=1
            return logits
    hair_override,_=preserve_occluders(original,baseline,affine,FixedMask(np.zeros((256,256),np.float32)),semantic=HairLabels())
    assert np.array_equal(hair_override,baseline)
    print('Occlusion compositor preserves hidden pixels, visible baseline and untouched background.')
if __name__=='__main__':main()
