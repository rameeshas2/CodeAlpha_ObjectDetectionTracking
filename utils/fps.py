import time

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()

    def update(self):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        return int(fps)