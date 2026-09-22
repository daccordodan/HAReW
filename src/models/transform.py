import numpy as np

class DopplerTraceTransformation:
    def __init__(self):
        None

    def __call__(self, window):
        noise = np.random.randn(*window.shape) * 0.1
        return np.clip(window + noise, 0, 1)