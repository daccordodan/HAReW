import numpy as np
from scipy.interpolate import interp1d

def dopplerTraceTransformation(window):
    p=0.6
    augmented = doppler_shift(p, window, shift_bin=5)
    augmented = time_stretch(p, augmented, scale_range=(0.9, 1.1))
    augmented = gaussian_noise(p, augmented, noise_std=0.04)
    augmented = temporal_mask(p, augmented, mask_width=(3, 10), n_masks=4)
    augmented = frequency_mask(p, augmented, mask_width=(2, 5), n_masks=3)
    return augmented

def doppler_shift(p, trace, shift_bin):
    if np.random.rand() > p:
        return trace
    shift = np.random.randint(-shift_bin,shift_bin+1)
    shifted = np.roll(trace, shift, axis=1)

    if shift > 0:
        shifted[:, :shift, :]=trace.min()
    elif shift < 0:
        shifted[:, shift:, :]=trace.min()
    return shifted

def time_stretch(p, trace, scale_range):
    if np.random.rand() > p:
        return trace
    scale = np.random.uniform(scale_range[0], scale_range[1])
    ch, ts, fb = trace.shape
    
    time_ax_old=np.arange(ts)
    time_ax_new=np.linspace(0, ts-1, int(ts/scale))
    stretched = np.zeros((ch, len(time_ax_new), fb))
    for c in range(ch):
        for i in range(fb):
            stretched[c,:,i]=interp1d(time_ax_old, trace[c,:,i], kind="linear", bounds_error=False, fill_value=0)(time_ax_new)

    if stretched.shape[1] < ts:
        pad_width = ((0, 0), (0, ts - stretched.shape[1]), (0, 0))
        stretched = np.pad(stretched, pad_width, "constant", constant_values=stretched.min())
    else:
        stretched = stretched[:, :ts, :]
    return stretched

def gaussian_noise(p, trace, noise_std=0.05):
    if np.random.rand() > p:
        return trace

    noise = np.random.randn(*trace.shape) * noise_std
    return np.clip(trace + noise, 0, 1)

def temporal_mask(p, trace, mask_width, n_masks):
    if np.random.rand() > p:
        return trace

    masked=trace.copy()
    ts=trace.shape[1]
    n_masks=np.random.randint(1, n_masks)
    for _ in range(n_masks):
        width=np.random.randint(mask_width[0], mask_width[1])
        start_time=np.random.randint(0, max(1, ts-width))
        masked[:, start_time:start_time+width, :]=trace.min()
    return masked

def frequency_mask(p, trace, mask_width, n_masks):
    if np.random.rand() > p:
        return trace

    masked=trace.copy()
    fb=trace.shape[2]
    n_masks=np.random.randint(1, n_masks)
    for _ in range(n_masks):
        width=np.random.randint(mask_width[0], mask_width[1])
        start_freq=np.random.randint(0, max(1, fb-width))
        masked[:, :, start_freq:start_freq+width]=trace.min()
    return masked