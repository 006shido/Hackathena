import numpy as np,cv2
from ml.experiments.reference_backend.flow_stability import residuals
rng=np.random.default_rng(41)
base=cv2.GaussianBlur(rng.integers(40,170,(96,96,3),dtype=np.uint8),(5,5),0)
old=base.copy();old[20:76,20:76]=np.minimum(old[20:76,20:76].astype(int)+12,255)
affine=np.array([[1,0,2],[0,1,1]],np.float32)
current=cv2.warpAffine(base,affine,(96,96),borderMode=cv2.BORDER_REFLECT)
stable=cv2.warpAffine(old,affine,(96,96),borderMode=cv2.BORDER_REFLECT)
flickering=stable.copy();flickering[21:77,22:78]=np.minimum(flickering[21:77,22:78].astype(int)+20,255)
result=residuals(base,current,[old,old],[stable,flickering])
assert result and result['valid_pixels']>100
assert result['output_mae'][1]>result['output_mae'][0]+10,result
print('PASS: known translated effect remains stable while an injected brightness pulse increases the residual.')
