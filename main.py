import numpy as np
import audioread
from piano_transcription_inference import PianoTranscription, sample_rate
from io import BytesIO
import base64
import os
import io

def load_audio_from_memory(audio_bytes, sr=22050, mono=True, offset=0.0, duration=None,
                            dtype=np.float32, res_type='kaiser_best', backends=None):
    """
    Load audio data from bytes in memory.

    Parameters:
    - audio_bytes: Bytes containing the audio data.
    - sr: Sample rate (default: 22050).
    - mono: Whether to convert audio to mono (default: True).
    - offset: Offset to trim the audio (default: 0.0).
    - duration: Duration to trim the audio (default: None).
    - dtype: Data type of the audio array (default: np.float32).
    - res_type: Resample type (default: 'kaiser_best').
    - backends: List of audioread backends to try (default: None).

    Returns:
    - audio: Loaded audio data.
    - sample_rate: Sample rate of the loaded audio.
    """
    with audioread.audio_open(io.BytesIO(audio_bytes), backends=backends) as f:
        audio = f.read_frames(f.get_length())
    
    return audio, f.samplerate

print('Transcribe piano')

audio_path = 'C:\\Users\\sarank\\Desktop\\New folder\\1.wav'
path = os.path.realpath(audio_path)

with open(path, 'rb') as audiofile:
    audio_bytes = audiofile.read()

print('Uploaded file')

(audio, loaded_sample_rate) = load_audio_from_memory(
    audio_bytes, sr=sample_rate, mono=True, offset=0.0, duration=None,
    dtype=np.float32, res_type='kaiser_best', backends=[audioread.ffdec.FFmpegAudioFile]
)

print('Resampling...')

transcriptor = PianoTranscription(device='cpu', checkpoint_path='C:\\Users\\sarank\\Desktop\\New folder\\model.pth')

buf = BytesIO()

def print_progress(current, total):
    print(f'Transcribing ({current + 1} / {total + 1} segments)...')

transcribed_dict = transcriptor.transcribe(audio, None, print_progress, buf)

filename = f'transcribed_{os.path.basename(path)}.mid'

b64 = base64.b64encode(buf.getvalue()).decode()
print(f'Download your MIDI file: data:audio/midi;base64,{b64}')
