"""受限沙箱兼容层：把 multiprocessing.pool.ThreadPool 换成串行实现。

原因：本机沙箱禁止创建命名管道，`multiprocessing.pool.Pool.__init__` 里的
SimpleQueue 会抛 PermissionError [WinError 5]，导致 ultralytics 缓存标签时崩溃。
ThreadPool 本来就是 I/O 密集任务，串行执行只慢一点，功能完全等价。

必须在 `import ultralytics` 之前 import 本模块。
"""
import multiprocessing.pool as _mp


class _SerialPool:
    def __init__(self, processes=None, *args, **kwargs):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func=None, iterable=None, chunksize=1, **kw):
        return map(func, iterable)

    def imap_unordered(self, func=None, iterable=None, chunksize=1, **kw):
        return map(func, iterable)

    def map(self, func=None, iterable=None, chunksize=None, **kw):
        return list(map(func, iterable))

    def map_async(self, func=None, iterable=None, chunksize=None, **kw):
        return _Done(list(map(func, iterable)))

    def apply(self, func, args=(), kwds=None):
        return func(*args, **(kwds or {}))

    def apply_async(self, func, args=(), kwds=None, **kw):
        return _Done(func(*args, **(kwds or {})))

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


class _Done:
    def __init__(self, value):
        self._value = value

    def get(self, timeout=None):
        return self._value

    def ready(self):
        return True

    def successful(self):
        return True

    def wait(self, timeout=None):
        return True


_mp.ThreadPool = _SerialPool
