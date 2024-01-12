from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import queue
import threading
from config import base_dir
import os

file_monitor_queue = queue.Queue()


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print("文件创建 %s" % event.src_path)
        file_monitor_queue.put(event.src_path)


def start_watchdog():
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, base_dir, recursive=True)
    observer.start()
    print("文件夹监控开启：" + base_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def start_file_monitor():
    worker = threading.Thread(target=start_watchdog)
    worker.start()
