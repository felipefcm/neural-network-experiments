import time


class Timer:
    name: str
    start_time: float

    def start(self, name: str):
        self.name = name
        self.start_time = time.time()

    def stop(self):
        print(f'{self.name} took {time.time() - self.start_time} seconds')
