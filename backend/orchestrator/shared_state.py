import multiprocessing as mp

class SharedState:
    """Centralizes queues and shared memory to avoid circular imports."""
    def __init__(self):
        self.frame_queue = mp.Queue(maxsize=2)
        self.detection_queue = mp.Queue(maxsize=2)
        self.broadcast_queue = mp.Queue(maxsize=2)
        self.stop_event = mp.Event()

# Singleton instance available to the orchestrator
shared_state = SharedState()
