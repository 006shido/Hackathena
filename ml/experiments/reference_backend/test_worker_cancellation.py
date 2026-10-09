"""Request cancellation must await inference and release orphaned resources."""
import asyncio,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from ml.api.video_preview import run_worker

async def main():
    started=threading.Event();finish=threading.Event();closed=[]
    class Resource:
        def close(self):closed.append(True)
    def worker():started.set();finish.wait(timeout=5);return Resource()
    task=asyncio.create_task(run_worker(worker))
    try:
        async def wait_started():
            while not started.is_set():
                if task.done():await task;raise AssertionError('Worker finished without starting')
                await asyncio.sleep(.005)
        await asyncio.wait_for(wait_started(),timeout=5)
        task.cancel();await asyncio.sleep(.02)
        assert not task.done(),'Cancelled request released its worker before inference finished'
        finish.set()
        try:await task;raise AssertionError('Cancellation was swallowed')
        except asyncio.CancelledError:pass
        assert closed==[True],'Cancelled session construction leaked its tracking resource'
    finally:finish.set()
    print('PASS: cancelled request waits for its worker and closes orphaned session resources.')

if __name__=='__main__':asyncio.run(main())
