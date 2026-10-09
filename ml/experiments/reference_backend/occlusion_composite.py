"""Experimental occluder preservation; tuned settings are not release defaults."""
import time
import cv2
import numpy as np

def preserve_occluders(original,baseline,affine,model,feather=0,state=None,semantic=None):
    if not np.isfinite(feather) or feather<0:raise ValueError('Invalid occlusion feather radius')
    started=time.perf_counter()
    doubled=np.asarray(affine,dtype=np.float32)*2
    crop=cv2.warpAffine(original,doubled,(256,256))
    scores=model(crop[:,:,::-1].astype(np.float32)[None]/255).cpu().numpy()
    visible=(scores>=.1).astype(np.float32)
    visible=cv2.dilate(visible,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)))
    full=cv2.warpAffine(visible,cv2.invertAffineTransform(doubled),(original.shape[1],original.shape[0]),borderValue=1)
    # Conservative binary projection avoids mixing generated pixels into hidden regions.
    visible=full>=.999
    hair_pixels=0
    if semantic is not None:
        large=cv2.warpAffine(original,np.asarray(affine,dtype=np.float32)*4,(512,512))
        normalized=(large.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        labels=semantic(normalized.transpose(2,0,1)[None]).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        hair=cv2.warpAffine((labels==17).astype(np.uint8),cv2.invertAffineTransform(np.asarray(affine,dtype=np.float32)*4),(original.shape[1],original.shape[0]),flags=cv2.INTER_NEAREST)>0
        hair_pixels=int((hair&~visible).sum())
        visible|=hair
    weight=visible.astype(np.float32)
    if feather:
        distance=cv2.distanceTransform(visible.astype(np.uint8),cv2.DIST_L2,5)
        weight=np.minimum(distance/feather,1)
    stabilized=0
    if state is not None:
        gray=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)
        if state.get('gray') is not None and state['gray'].shape==gray.shape:
            backward=cv2.calcOpticalFlowFarneback(gray,state['gray'],None,.5,3,15,3,5,1.2,0)
            height,width=gray.shape;y,x=np.mgrid[:height,:width].astype(np.float32)
            mapx=x+backward[...,0];mapy=y+backward[...,1]
            previous=cv2.remap(state['weight'],mapx,mapy,cv2.INTER_LINEAR)
            old_gray=cv2.remap(state['gray'],mapx,mapy,cv2.INTER_LINEAR)
            valid=(mapx>=1)&(mapx<width-2)&(mapy>=1)&(mapy<height-2)&(np.abs(gray.astype(float)-old_gray)<20)
            blended=np.minimum(weight,.65*weight+.35*previous)
            stabilized=int((valid&(blended<weight)).sum())
            weight=np.where(valid,blended,weight)
        state['gray']=gray.copy();state['weight']=weight.copy()
    output=np.clip(np.rint(original.astype(np.float32)+(baseline.astype(np.float32)-original)*weight[...,None]),0,255).astype(np.uint8)
    diagnostic=dict(occlusion_mask_ms=1000*(time.perf_counter()-started),
        occlusion_restored_pixels=int(np.any(output!=baseline,axis=2).sum()),occlusion_stabilized_pixels=stabilized,semantic_hair_override_pixels=hair_pixels)
    return output,diagnostic
