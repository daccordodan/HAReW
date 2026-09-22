import numpy as np

def dopplerTraceTransformation(window):
    noise = np.random.randn(*window.shape) * 0.1
    return np.clip(window + noise, 0, 1)