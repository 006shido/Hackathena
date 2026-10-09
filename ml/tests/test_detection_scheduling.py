import asyncio
import threading
import unittest
from types import SimpleNamespace
from fastapi import HTTPException
from ml.api.media_detection import execute

class Scheduling(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inference_lock=asyncio.Lock())))

    async def test_busy_swap_gives_detector_a_turn(self):
        request=self.request();lock=request.app.state.inference_lock
        await lock.acquire()
        detector=asyncio.create_task(execute(request,('user','video'),lambda:'measured'))
        await asyncio.sleep(.02)
        self.assertFalse(detector.done(),'A busy swap must not immediately reject detection')
        lock.release()
        self.assertEqual(await detector,'measured')
        self.assertFalse(lock.locked())
        self.assertEqual(request.app.state.pending_detection,set())

    async def test_duplicate_sample_does_not_build_a_queue(self):
        request=self.request();lock=request.app.state.inference_lock
        await lock.acquire()
        task=asyncio.create_task(execute(request,('user','video'),lambda:'sample'))
        await asyncio.sleep(.02)
        with self.assertRaises(HTTPException) as caught:
            await execute(request,('user','video'),lambda:'duplicate')
        self.assertEqual(caught.exception.status_code,429)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(lock.locked(),'Canceled waiter must not release the swap lock')
        self.assertFalse(request.app.state.pending_detection)
        lock.release()

    async def test_cancel_does_not_release_running_worker(self):
        request=self.request();started=threading.Event();finish=threading.Event()
        def worker():started.set();finish.wait(2);return 'done'
        task=asyncio.create_task(execute(request,('user','audio'),worker))
        await asyncio.to_thread(started.wait,1)
        task.cancel();await asyncio.sleep(.02)
        self.assertTrue(request.app.state.inference_lock.locked())
        finish.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertFalse(request.app.state.inference_lock.locked())
        self.assertFalse(request.app.state.pending_detection)

if __name__=='__main__':unittest.main()
