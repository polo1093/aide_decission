import time


class Timer():
    """Timer used to check whether a waiting period has passed.

    Args:
        time_wait (float): Duration to wait in seconds.

    Returns:
        bool: True if the timer has expired.
    """
    def __init__(self,time_wait):
        self.start_time = time.perf_counter()
        self.time_wait = time_wait
    
    def is_expire(self):
        return time.perf_counter()-self.start_time >= self.time_wait    
    
    def is_running(self):
        return time.perf_counter()-self.start_time < self.time_wait
    
    def refresh(self,time_wait=0):
        if time_wait > 0: 
            self.time_wait = time_wait
        self.start_time = time.perf_counter()



