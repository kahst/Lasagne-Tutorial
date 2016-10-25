import time

class Timer(object):
    
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.diff_ms = 0
        self.average_time = 0.
        self.average_time_ms = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        self.average_time_ms = int(self.average_time * 1000)
        self.diff_ms = int(self.diff * 1000)
