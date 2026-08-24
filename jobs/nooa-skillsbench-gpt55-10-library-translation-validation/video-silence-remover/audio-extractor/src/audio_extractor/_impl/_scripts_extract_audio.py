"""
Audio extractor - extracts audio from video to WAV format.
"""
import os
import subprocess

def extract_audio(video_path, output_path, sample_rate=16000, duration=None):
    """Extract audio from video to WAV format."""
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '1']
    if duration:
        cmd.extend(['-t', str(duration)])
    cmd.extend([output_path, '-y'])
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
