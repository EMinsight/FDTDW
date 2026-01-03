import warp as wp
import time


class GraphTape:
    def __init__(self, device, max_nodes=1000):
        self.device = device
        self.max_nodes = max_nodes
        self.graphs = []
        self.current_nodes = 0
        self.recording = False

    def __enter__(self):
        self.begin_chunk()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recording:
            self.end_chunk()

    def begin_chunk(self):
        wp.capture_begin(device=self.device)
        self.recording = True
        self.current_nodes = 0

    def end_chunk(self):
        g = wp.capture_end(device=self.device)
        self.graphs.append(g)
        self.recording = False

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
        self.current_nodes += 1

        if self.current_nodes >= self.max_nodes:
            self.end_chunk()
            self.begin_chunk()

    def replay(self):
        wp.synchronize()
        t_start = time.perf_counter()
        for g in self.graphs:
            wp.capture_launch(g)
        wp.synchronize()
        duration = time.perf_counter() - t_start
        print(f"simulation executed in {duration}s")

    def __call__(self):
        self.replay()
