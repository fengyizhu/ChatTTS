import io
import math
import wave

import numpy as np
from numba import jit


@jit
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)

def pcm_to_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    buf.write(float_to_int16(audio))
    buf.seek(0, 0)
    pcm_data = buf.getvalue()
    buf.close()
    return pcm_data

def pcm_to_wav_bytes(pcm_data: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(float_to_int16(pcm_data))
    buf.seek(0, 0)
    wav_data = buf.getvalue()
    buf.close()
    return wav_data

def response_format_to_bytes(audio: np.ndarray, response_format: str) -> bytes:
    if response_format == "pcm":
        return pcm_to_bytes(audio)
    elif response_format == "wav":
        return pcm_to_wav_bytes(audio)
    else:
        raise ValueError(f"Unsupported response format: {response_format}")

