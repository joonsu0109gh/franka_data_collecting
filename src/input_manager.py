import pyspacemouse
from multiprocessing import Process, Queue, Lock


class SpaceMouseManager:
    """
    Manages reading from a SpaceMouse in a separate process to prevent
    blocking the main application.
    """

    def __init__(self):
        self.queue = Queue(maxsize=1)
        self.lock = Lock()
        self.process = None

    def _read_loop(self):
        """The target function for the separate process."""
        # This will block until the device is connected
        success = pyspacemouse.open()
        if success:
            while True:
                state = pyspacemouse.read()
                with self.lock:
                    # If the queue is full, discard the old state and put the new one
                    if self.queue.full():
                        self.queue.get()
                    self.queue.put(state)

    def start(self):
        """Starts the background process for reading mouse input."""
        if self.process is None:
            self.process = Process(target=self._read_loop, daemon=True)
            self.process.start()
            print("SpaceMouse process started.")

    def get_state(self):
        """
        Retrieves the latest state from the queue.
        Returns None if the queue is empty.
        """
        state = None
        with self.lock:
            if not self.queue.empty():
                state = self.queue.get()
        return state

    def stop(self):
        """Terminates the background process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            print("SpaceMouse process stopped.")
