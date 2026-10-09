"""Verify readiness metadata identifies the actual configured model."""
import importlib, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
service=importlib.import_module('ml.api.app')
service.torch.cuda.is_available=lambda: False


def completed(coroutine):
    # This route contains no awaits. Avoid an OS socket-based event loop for
    # a pure metadata unit test; fail if the route starts yielding I/O.
    try:
        coroutine.send(None)
    except StopIteration as finished:
        return finished.value
    raise AssertionError('Metadata route unexpectedly awaited I/O')


def main():
    service.app.state.ready=False
    response=completed(service.health_check())
    assert response.status_code==503
    service.app.state.ready=True
    service.app.state.gpu_name='test-gpu'
    service.app.state.engine=SimpleNamespace(model_variant='fullres',checkpoint_path='test/selected.pt',
        checkpoint_sha256='a'*64,correspondence='visibility')
    response=completed(service.health_check())
    assert response['model']=='fullres'
    assert response['checkpoint']=='selected.pt'
    assert response['checkpoint_sha256']=='a'*64
    assert response['correspondence']=='visibility'
    assert response['ready'] is True
    print('PASS: unready service refuses readiness; ready response identifies actual model and checkpoint.')


if __name__=='__main__':main()
