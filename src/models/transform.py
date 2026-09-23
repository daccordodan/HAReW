import numpy as np
from scipy.interpolate import interp1d

def dopplerTraceTransformation(window):
    p=0.7
    augmented = doppler_shift(p, window, shift_bins=5)
    augmented = time_stretch(p, augmented, scale_range=(0.85, 1.15))
    augmented = gaussian_noise(p, augmented, noise_std=0.08)
    augmented = temporal_mask(p, augmented, mask_width_range=(3, 10))
    augmented = frequency_mask(p, augmented, mask_width_range=(2, 6))
    return augmented

def doppler_shift(p, trace, shift_bin):
    if np.random.rand() > p:
        return trace
    shift = np.random.randInt(-shift_bin,shift_bin+1)
    shifted = np.roll(trace, shift, axis=0)

    if shift > 0:
        shifted[:shift, :]=0
    elif shift < 0:
        shifted[shift:, :]=0
    return shifted

def time_stretch(p, trace, scale_range):
    if np.random.rand() > p:
        return trace
    scale = np.random.uniform(scale_range[0], scale_range[1])
    fb, ts = trace.shape

    time_ax_old=np.arrange(ts)
    time_ax_new=np.linspace(0, ts-1, int(ts/scale))
    stretched = np.zeros((fb,len(time_ax_new)))
    for i in range(fb):
        stretched[i]=interp1d(time_ax_old, trace[i,:], kind="linear", bounds_error=False, fill_value=0)(time_ax_new)

    if stretched.shape[1] < ts:
        pad_width = ((0, 0), (0, ts - stretched.shape[1]))
        stretched = np.pad(stretched, pad_width)
    else:
        stretched = stretched[:, :ts]
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
        mask_width=np.random.randint(n_masks[0], n_masks[1])
        start_time=np.random.randint(0, max(1, ts-mask_width))
        masked[:, start_time:start_time+mask_width]=0
    return masked

def frequency_mask(p, trace, mask_width, n_masks):
    if np.random.rand() > p:
        return trace

    masked=trace.copy()
    ts=trace.shape[0]
    n_masks=np.random.randint(1, n_masks)
    for _ in range(n_masks):
        mask_width=np.random.randint(n_masks[0], n_masks[1])
        start_time=np.random.randint(0, max(1, ts-mask_width))
        masked[start_time:start_time+mask_width, :]=0
    return masked