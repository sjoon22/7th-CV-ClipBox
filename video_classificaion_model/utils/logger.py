import os
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_dir, filename='train.log'):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)

        self.terminal = sys.stdout  # 원래 stdout
        self.log = open(log_path, 'a')
        self._write_header()

    def _write_header(self):
        self.log.write("\n========== New Training Session ==========\n")
        self.log.write(f"{datetime.now()}\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
