"""
Silence detector - finds initial silence segment using energy data.
Detects low-energy periods at the start of audio (e.g., title slides, setup time).
"""
import json
import numpy as np

def detect_initial_silence(energies, threshold_multiplier=1.5, initial_window=60, smoothing_window=30):
    """Detect initial silence using energy threshold method."""
    energies = np.array(energies)
    initial_avg = np.mean(energies[:min(initial_window, len(energies))])
    threshold = initial_avg * threshold_multiplier
    if len(energies) >= smoothing_window:
        smoothed = np.convolve(energies, np.ones(smoothing_window) / smoothing_window, mode='valid')
    else:
        smoothed = energies
    silence_end = 0
    for i in range(len(smoothed)):
        if smoothed[i] > threshold:
            silence_end = i
            break
    return (silence_end, {'initial_avg': float(initial_avg), 'threshold': float(threshold)})
