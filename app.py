import streamlit as st
import whisper
import tempfile
import os
import torchaudio
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import soundfile as sf
from io import BytesIO

# Title and description
st.title("🎧 Whisper Audio Transcriber")
st.markdown("Upload a `.wav` or `.mp3` file or record audio using your microphone to get transcribed text with timestamps using Whisper.")

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()
st.success("✅ Whisper model loaded!")

# File uploader
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

# Microphone recording
st.subheader("🎙️ Record Audio")
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame):
        self.audio_buffer.append(frame.to_ndarray())
        return frame

ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)
if ctx.audio_processor:
    if st.button("Stop and Transcribe Recording"):
        if ctx.audio_processor.audio_buffer:
            st.info("📝 Processing recorded audio...")
            # Combine audio frames
            audio_data = np.concatenate(ctx.audio_processor.audio_buffer, axis=0)
            # Save as WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio_data, 16000)  # WebRTC typically uses 16kHz
                temp_path = tmp_file.name

            # Transcription
            st.info("📝 Transcribing...")
            result = model.transcribe(temp_path)

            # Display segments
            st.subheader("🕒 Segments with Timestamps")
            for segment in result["segments"]:
                st.markdown(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]**: {segment['text']}")

            # Full transcription
            st.subheader("🧾 Full Transcript")
            st.text_area("Transcribed Text", result["text"], height=250, key="recorded_transcript")

            # Clean up
            os.remove(temp_path)
            ctx.audio_processor.audio_buffer = []  # Clear buffer
        else:
            st.warning("⚠️ No audio recorded.")

# Process uploaded file
if audio_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        temp_path = tmp_file.name

    # Convert MP3 to WAV if needed
    if audio_file.name.endswith(".mp3"):
        waveform, sample_rate = torchaudio.load(temp_path)
        wav_path = temp_path.replace(".wav", "_converted.wav")
        torchaudio.save(wav_path, waveform, sample_rate)
        os.remove(temp_path)
        temp_path = wav_path

    # Transcription
    st.info("📝 Transcribing...")
    result = model.transcribe(temp_path)

    # Display segments
    st.subheader("🕒 Segments with Timestamps")
    for segment in result["segments"]:
        st.markdown(f"**[{segment['start']:.2f}s - {segment['end']:.2f}s]**: {segment['text']}")

    # Full transcription
    st.subheader("🧾 Full Transcript")
    st.text_area("Transcribed Text", result["text"], height=250, key="uploaded_transcript")

    # Clean up
    os.remove(temp_path)
