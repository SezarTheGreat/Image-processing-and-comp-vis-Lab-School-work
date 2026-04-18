#Reader Writer Problem
import threading
import time
import random
class Reader(threading.Thread):
    def __init__(self, reader_id, rw_lock):
        super().__init__()
        self.reader_id = reader_id
        self.rw_lock = rw_lock

    def run(self):
        while True:
            time.sleep(random.uniform(0.1, 0.5))  # Simulate reading time
            with self.rw_lock:
                print(f"Reader {self.reader_id} is reading.")
                time.sleep(random.uniform(0.1, 0.5))  # Simulate reading time
class Writer(threading.Thread):
    def __init__(self, writer_id, rw_lock):
        super().__init__()
        self.writer_id = writer_id
        self.rw_lock = rw_lock

    def run(self):
        while True:
            time.sleep(random.uniform(0.1, 0.5))  # Simulate writing time
            with self.rw_lock:
                print(f"Writer {self.writer_id} is writing.")
                time.sleep(random.uniform(0.1, 0.5))  # Simulate writing time
rw_lock = threading.Lock()
readers = [Reader(i, rw_lock) for i in range(1, 4)]
writers = [Writer(i, rw_lock) for i in range(1, 3)]
for r in readers:
    r.start()
for w in writers:
    w.start()