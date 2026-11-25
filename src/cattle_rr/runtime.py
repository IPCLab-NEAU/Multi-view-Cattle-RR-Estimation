from collections import defaultdict
import time
class TimerBook:
    def __init__(self): self.values=defaultdict(float)
    def measure(self,name): return _TimerContext(self,name)
class _TimerContext:
    def __init__(self,book,name): self.book,self.name=book,name
    def __enter__(self): self.start=time.perf_counter(); return self
    def __exit__(self,*exc): self.book.values[self.name]+=time.perf_counter()-self.start
