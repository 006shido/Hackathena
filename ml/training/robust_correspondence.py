"""Orientation-aware source-to-target correspondence with explicit reliability."""
import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay
from ml.training.face_preprocessing import estimate_umeyama_similarity_transform


def robust_warp(source_image, source_points, target_points, size=128, smooth_fallback=False):
    source = np.asarray(source_image.convert('RGB'))
    src = np.asarray(source_points, dtype=np.float64)[:468, :2]
    dst = np.asarray(target_points, dtype=np.float64)[:468, :2]
    if src.shape != (468,2) or dst.shape != src.shape or not np.isfinite([src,dst]).all():
        raise ValueError('Finite matching 468-point meshes are required.')
    # Iris interior vertices change with gaze and can conflict with eyelids.
    # Surface vertices cover eyes without those unstable interior constraints.
    anchors = np.array([[0,0],[size-1,0],[0,size-1],[size-1,size-1],
        [(size-1)/2,0],[0,(size-1)/2],[size-1,(size-1)/2],[(size-1)/2,size-1]])
    sa,da = np.vstack([src,anchors]),np.vstack([dst,anchors])
    triangles = Delaunay(da)
    sv,dv = sa[triangles.simplices],da[triangles.simplices]
    se,de = sv[:,1:]-sv[:,:1],dv[:,1:]-dv[:,:1]
    signed_s = np.linalg.det(se)
    signed_d = np.linalg.det(de)
    # Absolute area ratios alone cannot distinguish valid maps from foldovers.
    usable = (np.abs(signed_s)>0.05)&(np.abs(signed_d)>0.05)&(signed_s*signed_d>0)
    stretch = np.zeros((len(sv),2))
    if usable.any():
        backward = np.linalg.solve(de[usable],se[usable])
        stretch[usable] = np.linalg.svd(backward,compute_uv=False)
    usable &= (stretch[:,0]<3)&(stretch[:,1]>1/3)
    confidence = np.zeros(len(sv),dtype=np.float32)
    distortion=np.maximum(stretch[:,0],1/np.maximum(stretch[:,1],1e-6))
    confidence[usable]=np.exp(-0.25*np.maximum(0,distortion[usable]-1))
    yy,xx=np.mgrid[:size,:size]
    coordinates=np.column_stack([xx.ravel(),yy.ravel()])
    simplex=triangles.find_simplex(coordinates)
    inside=simplex>=0
    good=inside.copy()
    good[inside]=usable[simplex[inside]]
    # Fallback is a global similarity warp, never a folded local triangle.
    transform=estimate_umeyama_similarity_transform(dst[[33,133,362,263,1,61,291]],src[[33,133,362,263,1,61,291]])
    mapped=coordinates@transform[:,:2].T+transform[:,2]
    global_mapped=mapped.copy()
    ids=simplex[good]
    bary=np.einsum('nij,nj->ni',triangles.transform[ids,:2],coordinates[good]-triangles.transform[ids,2])
    weights=np.column_stack([bary,1-bary.sum(axis=1)])
    mapped[good]=np.einsum('ni,nij->nj',weights,sv[ids])
    # A hard switch between global and local maps creates visible seams.
    # Fade within a narrow valid-side band; invalid pixels stay confidence zero.
    if smooth_fallback:
        distance=cv2.distanceTransform(good.reshape(size,size).astype(np.uint8),cv2.DIST_L2,5)
        local_weight=np.clip(distance.ravel()/2,0,1)
        mapped=global_mapped+(mapped-global_mapped)*local_weight[:,None]
    map_x,map_y=mapped[:,0].reshape(size,size).astype(np.float32),mapped[:,1].reshape(size,size).astype(np.float32)
    warped=cv2.remap(source,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    conf=np.zeros(size*size,dtype=np.float32)
    conf[good]=confidence[ids]
    conf=conf.reshape(size,size)
    # Filtering softens boundaries but never makes an invalid pixel reliable.
    conf=cv2.GaussianBlur(conf,(5,5),1)*good.reshape(size,size)
    normalized=warped.astype(np.float32)/127.5-1
    report=dict(rejected_triangle_fraction=float((~usable).mean()),unreliable_pixel_fraction=float((~good).mean()),
        flipped_triangles=int(((signed_s*signed_d)<=0).sum()))
    return torch.from_numpy(normalized.transpose(2,0,1)).unsqueeze(0),torch.from_numpy(conf).reshape(1,1,size,size),report


def reliable_legacy_warp(source_image, source_points, target_points, size=128):
    """Preserve trained legacy coordinates while suppressing invalid triangles.

    Mask before the source encoder as well as at skip gates, so invalid
    pixels cannot contaminate adjacent convolution features.
    """
    from ml.inference.face_correspondence import warp_and_prepare_source
    aligned, confidence = warp_and_prepare_source(source_image, source_points, target_points, size)
    _, reliability, report = robust_warp(source_image, source_points, target_points, size)
    valid = (reliability > 0).to(aligned.dtype)
    return aligned * valid, torch.minimum(confidence, reliability), report


def visibility_warp(source_image, source_points, target_points, size=128, strength=0.5):
    """Fill unreliable local coordinates with a smooth global source mapping.

    Blend sampling coordinates, never two RGB faces. The global fallback
    retains source texture rather than introducing neutral holes.
    """
    from ml.inference.face_correspondence import compute_piecewise_affine_map
    if not 0 <= strength <= 1:
        raise ValueError('strength must be in [0,1]')
    src, dst = np.asarray(source_points), np.asarray(target_points)
    mx, my = compute_piecewise_affine_map(src, dst, size, size)
    _, reliability, report = robust_warp(source_image, src, dst, size)
    stable = [33, 133, 362, 263, 1, 61, 291]
    transform = estimate_umeyama_similarity_transform(dst[stable,:2], src[stable,:2])
    yy, xx = np.mgrid[:size,:size]
    global_xy = np.stack([xx,yy],axis=-1) @ transform[:,:2].T + transform[:,2]
    invalid = (reliability.numpy()[0,0] == 0).astype(np.float32)
    weight = strength * cv2.GaussianBlur(invalid, (0,0), 3)
    mx = ((1-weight)*mx + weight*global_xy[:,:,0]).astype(np.float32)
    my = ((1-weight)*my + weight*global_xy[:,:,1]).astype(np.float32)
    warped = cv2.remap(np.asarray(source_image.convert('RGB')),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    confidence = reliability.clamp_min(0.5)
    normalized = warped.astype(np.float32)/127.5-1
    report.update(fallback_strength=strength, fallback_weight_mean=float(weight.mean()))
    return torch.from_numpy(normalized.transpose(2,0,1)).unsqueeze(0), confidence, report
