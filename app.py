import streamlit as st
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio_from_memory
from io import BytesIO
import base64
import os
st.title('Transcribe piano')

audiofile = st.file_uploader('Upload audio file', type=['.wav'], accept_multiple_files=False)
my_bar = None

def print_progress(current, total):
    my_bar.progress(current / total)
    my_bar.text(f'Transcribing ({current + 1} / {total + 1} segments)...')

if audiofile is not None:
    audio_bytes = audiofile.read()
    st.text('Uploaded file')
    st.audio(audio_bytes, format='audio/wav')
    path = "C:\path\to\your\audio\file.wav"

    print("Path before realpath:", path)
    path = os.path.realpath(path)
    print("Path after realpath:", path)

    with st.spinner('Resampling...'):
        (audio, _) = load_audio_from_memory(audio_bytes, sr=sample_rate, mono=True)
    st.success('Resampling complete.')

    my_bar = st.progress(0)
    my_bar.text('Transcribing...')

    transcriptor = PianoTranscription(device='cpu', checkpoint_path='C:\\Users\\sarank\\Desktop\\audio to midi\\model.pth')

    buf = BytesIO()

    transcribed_dict = transcriptor.transcribe(audio, None, print_progress, buf)

    filename = f'transcribed_{audiofile.name}.mid'

    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<a href="data:audio/midi;base64,{b64}" download="{filename}">Download MIDI</a>', unsafe_allow_html=True)
    st.balloons()
