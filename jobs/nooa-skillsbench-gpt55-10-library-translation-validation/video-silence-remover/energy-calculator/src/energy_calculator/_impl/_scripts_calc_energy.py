"""
Energy calculator - calculates per-second RMS energy from audio.
"""
import json
import wave
import numpy as np

def calculate_energy(audio_path, window_seconds=1):
    """Calculate per-second RMS energy from audio file."""
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_data = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    window_size = int(sample_rate * window_seconds)
    energies = []
    for i in range(0, len(audio_array), window_size):
        window = audio_array[i:i + window_size]
        if len(window) > 0:
            rms = np.sqrt(np.mean(window ** 2))
            energies.append(float(rms))
    return {'sample_rate': sample_rate, 'window_seconds': window_seconds, 'total_seconds': len(energies) * window_seconds, 'energies': energies, 'stats': {'min': float(np.min(energies)), 'max': float(np.max(energies)), 'mean': float(np.mean(energies)), 'std': float(np.std(energies))}}
