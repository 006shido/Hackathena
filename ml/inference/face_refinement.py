"""Conservative geometry/color refinement for aligned 128px neural outputs."""
import cv2
import numpy as np
from scipy.spatial import Delaunay
from PIL import Image
from ml.inference.face_correspondence import compute_piecewise_affine_map


def refine_face(swap, target, predicted_mask, face_mask, target_dense, preprocessor, identity_score=None, geometry_strength=0.0):
    """Return RGB composite and diagnostics; reject unreliable geometry explicitly.

    Inputs are RGB uint8 images and masks in [0,1]. No source/target blending
    occurs inside the face core; feathering is confined to the perimeter.
    """
    swap = np.asarray(swap, dtype=np.uint8)
    if not 0 <= geometry_strength <= 1:
        raise ValueError('geometry_strength must be in [0,1].')
    target = np.asarray(target, dtype=np.uint8)
    if swap.shape != target.shape or swap.shape != (128, 128, 3):
        raise ValueError('Expected matching 128x128 RGB inputs.')
    original_mask = np.asarray(predicted_mask, dtype=np.float32).squeeze()
    region = np.asarray(face_mask, dtype=np.float32).squeeze()
    if original_mask.shape != (128, 128) or region.shape != (128, 128):
        raise ValueError('Expected 128x128 masks.')
    fallback = np.clip(swap.astype(float)*original_mask[...,None] + target*(1-original_mask[...,None]),0,255).astype(np.uint8)
    details = dict(accepted=False, reason=None)
    try:
        detection = preprocessor.detect_landmarks(Image.fromarray(swap))
        generated = detection.dense_landmarks
        if generated is None: raise ValueError('Missing generated landmarks')
        # Redetect the generated face, then correct residual geometric drift.
        displacement = np.linalg.norm(generated-target_dense, axis=1)
        details['landmark_displacement_mean_px'] = float(displacement.mean())
        if np.percentile(displacement,95) > 12:
            details['reason'] = 'excessive_geometry_displacement'
            return fallback, details
        triangles = Delaunay(target_dense).simplices
        src = generated[triangles,:2]
        dst = target_dense[triangles,:2]
        def area(points):
            a,b = points[:,1]-points[:,0], points[:,2]-points[:,0]
            return a[:,0]*b[:,1]-a[:,1]*b[:,0]
        sa,ta = area(src),area(dst)
        useful = np.abs(ta)>0.5
        folds = useful & ((sa*ta<=0) | (np.abs(sa)/(np.abs(ta)+1e-6)>4) | (np.abs(sa)/(np.abs(ta)+1e-6)<0.25))
        ratio = float(folds.sum()/max(1,useful.sum()))
        details['unreliable_triangle_fraction'] = ratio
        if ratio > 0.15:
            details['reason'] = 'unreliable_correspondence'
            return fallback, details
        # Partial correction avoids rewriting source facial proportions to match
        # the target identity. Full landmark snapping degraded ArcFace identity.
        if geometry_strength:
            destination = generated + geometry_strength * (target_dense-generated)
            mx,my = compute_piecewise_affine_map(generated,destination,128,128)
            aligned = cv2.remap(swap,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        else:
            aligned = swap
        # Bounded LAB mean correction aligns lighting while retaining texture.
        core = cv2.erode((region>0.5).astype(np.uint8),np.ones((5,5),np.uint8))>0
        if core.sum()<100: raise ValueError('Insufficient face area')
        lab = cv2.cvtColor(aligned.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        reference = cv2.cvtColor(target.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        offset = np.clip(np.median(reference[core],axis=0)-np.median(lab[core],axis=0),[-8,-6,-6],[8,6,6])
        corrected = cv2.cvtColor((lab+offset).astype(np.float32),cv2.COLOR_LAB2RGB)
        corrected = np.clip(corrected*255,0,255)
        # Distance feathering keeps the interior opaque (avoids doubled features).
        binary = (region>0.5).astype(np.uint8)
        distance = cv2.distanceTransform(binary,cv2.DIST_L2,5)
        alpha = np.clip(distance/4,0,1)
        composite = np.clip(corrected*alpha[...,None]+target*(1-alpha[...,None]),0,255).astype(np.uint8)
        if identity_score is not None:
            before, after = float(identity_score(fallback)), float(identity_score(composite))
            details.update(identity_before=before, identity_after=after)
            if after < before - 0.02:
                details['reason'] = 'identity_regression'
                return fallback, details
        details.update(accepted=True, reason='refined', lab_offset=offset.tolist())
        return composite,details
    except Exception as error:
        details['reason'] = 'refinement_failed: '+str(error)
        return fallback,details
