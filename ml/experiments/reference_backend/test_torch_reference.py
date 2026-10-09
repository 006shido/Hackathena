"""Compare the restricted fp32 executor against OpenCV on fixed inputs."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.torch_reference import TorchResearchSwapper

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
reference=OpenCVReference()
swapper=TorchResearchSwapper()
rng=np.random.default_rng(20261006)
records=[]
for index in range(3):
    target=rng.random((1,3,128,128),dtype=np.float32)
    source=rng.normal(size=(1,512)).astype(np.float32)
    source/=np.linalg.norm(source)
    reference.swapper.setInput(target,'target')
    reference.swapper.setInput(source,'source')
    expected=reference.swapper.forward()
    actual=swapper(target,source).cpu().numpy()
    difference=np.abs(actual-expected)
    record=dict(case=index,max_absolute_error=float(difference.max()),mean_absolute_error=float(difference.mean()))
    records.append(record)
    print(record,flush=True)
    assert record['max_absolute_error']<.002 and record['mean_absolute_error']<.0001, 'Research executor failed numerical parity.'
times=[]
for index in range(15):
    torch.cuda.synchronize();start=time.perf_counter()
    output=swapper(target,source).cpu().numpy()
    torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
    if index>=3:times.append(elapsed)
report=dict(parity_cases=records,fp32=True,swap_with_transfers_ms_median=float(np.median(times)),
    swap_with_transfers_ms_p95=float(np.percentile(times,95)),scope='Swap graph and transfers; excludes detection, recognition, compositing and network.')
(Path(__file__).parent/'torch_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
