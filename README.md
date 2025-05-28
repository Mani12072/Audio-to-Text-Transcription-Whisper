# Audio-to-Text-Transcription-Whisper
# 📢 Speech-to-Text Transcription System Using Whisper
 
The Streamlit application provided below only supports uploading audio files for transcription and does not include functionality for recording audio using a microphone.
### **Link**: https://huggingface.co/spaces/Mpavan45/Audio_to_Text_by_Whisper

## Overview
This project creates a system that converts spoken words into written text using OpenAI's Whisper model, a powerful tool for speech recognition. 

You can either record audio using a microphone or upload an audio file (like .wav or .mp3). The system cleans the audio to remove background noise, transcribes it into text with timestamps (showing when each part was spoken), and lets you play back the processed audio to check the results. 

It's like having a personal transcription assistant that listens to your voice and writes down what you say, with clear start and end times for each sentence.

We'll walk through every step in detail, from installing the necessary tools to running the system, with explanations of what each part does and why it’s needed. By the end, you'll have a working speech-to-text system on your computer.

## ✨ Features
- 🎙 **Record Audio**: Use your microphone to capture your voice.
- 📁 **Upload Audio Files**: Use pre-recorded .wav or .mp3 files.
- 🔇 **Noise Reduction**: Clean up background noise (like fans or chatter) for clearer audio.
- 🕐 **Transcription with Timestamps**: Get written text of the audio, with exact start and end times for each segment.
- 🔊 **Audio Playback**: Listen to the cleaned audio to verify it sounds good.

## Requirements
To make this system work, you need a few tools installed on your computer. Think of these as ingredients for a recipe—you need them all before you can start cooking.

- **Python 3.7 or Higher**: Python is the programming language that runs the system. It’s like the kitchen where everything happens.
- **Python Libraries**:
  - `whisper`: The OpenAI tool that converts audio to text (the brain of the system).
  - `sounddevice`: Captures audio from your microphone.
  - `librosa`: Helps process audio files.
  - `noisereduce`: Removes unwanted background noise.
  - `scipy`: Saves recorded audio as a file.
  - `ipython`: Lets you display and play audio in environments like Jupyter Notebook.
- **FFmpeg**: A free tool that Whisper uses to handle different audio formats (like .mp3 or .wav). It’s like a translator that ensures Whisper can understand your audio files.

You’ll install these tools step-by-step later in the guide.

I’ve uploaded upload_conversion.py and microphone.py, but the output may vary despite using the same code. Additionally, I’ve included test audio files for evaluating the output.

Note: Google Colab does not support audio recording, so it’s recommended to use Jupyter Notebook for recording functionality.

## 🔧 Project Modules and Pipeline

This project implements a speech-to-text transcription system using OpenAI's Whisper model. It supports audio input via microphone recording or file upload, applies noise reduction, and generates transcribed text with timestamps.

### 📁 Project Structure

    SpeechTranscription/
    
    ├── upload_document.py          # Handles upload and transcription of audio files (.wav, .mp3)
    
    ├── microphone.py               # Manages real-time audio recording and transcription
    
    ├── Streamlit-App              # Streamlit web interface for user-friendly interaction
    
    ├── Test-Files                 # Sample audio files for testing the transcription system
    
    ├── requirements.txt            # Project dependencies
    
## 📥 How It Works (Step-by-Step)
Here’s a deep dive into how the system operates, explained as if you’re building it from scratch. Each step is broken down to make it clear what’s happening and why it’s important.

### Step 1: Set Up Your Environment
Before you can run the system, you need to prepare your computer by installing the required tools. This is like setting up your workspace before starting a project.

**Install Python**:
  - Python is the programming language that runs the script. If you don’t have it, download it from https://www.python.org/downloads/.
  - Choose Python 3.7 or later (e.g., 3.10 or 3.11). During installation, check the box to “Add Python to PATH” so your computer knows where to find it.
  - To verify Python is installed, open a Command Prompt (Windows) or Terminal (Mac/Linux) and type:
python --version

You should see something like `Python 3.10.4`. If not, reinstall Python and ensure it’s added to PATH.

**Install FFmpeg**:
- FFmpeg is a tool Whisper uses to process audio files. It converts audio into a format Whisper can understand.

1. Go to https://www.gyan.dev/ffmpeg/builds/ and download `ffmpeg-release-essentials.zip` or `ffmpeg-n7.0-latest-win64-gpl.zip`.
2. Extract the ZIP file to a folder, like `C:\ffmpeg`.
3. Find the `bin` folder inside (e.g., `C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin`).
4. Add this folder to your system’s PATH:
   - Right-click **This PC** (or My Computer) → **Properties** → **Advanced system settings** → **Environment Variables**.
   - Under **System variables**, find **Path**, click **Edit**, then **New**, and paste the path to the `bin` folder (e.g., `C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin`).
   - Click **OK** to save everything.
5. Open a new Command Prompt and type:
ffmpeg -version

If you see `ffmpeg version n7.0...`, it’s installed correctly. If not, double-check the PATH or restart your computer.


**Install Python Libraries**:
- Open a Command Prompt or Terminal and install the required libraries using `pip`, Python’s package manager:
pip install whisper sounddevice librosa noisereduce scipy ipython

- If you see errors (e.g., `pip` not found), ensure Python is added to PATH. You can also try:
python -m pip install --upgrade pip

Then rerun the install command.
- If `sounddevice` fails, you may need additional audio drivers:
- Install PortAudio with:

            pip install pyaudio
            sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev


### Step 2: Load the Whisper Model
Whisper is the heart of the system—it’s the AI model that listens to audio and turns it into text. 
When you start the system, it loads a pre-trained Whisper model called “base.” This model is small enough to run on most computers but still powerful enough to transcribe accurately.

- Why “base”? Whisper offers different model sizes (tiny, base, small, medium, large). The “base” model is a good balance between speed and accuracy. Larger models are more accurate but need more powerful computers.
- The system downloads the model automatically the first time you use it, so you need an internet connection initially.

### Step 3: Choose Your Input Method
The system gives you two ways to provide audio:
- **Record from Microphone**:
- The system uses the `sounddevice` library to capture audio from your microphone.
- You specify how long to record (e.g., 5 seconds). The system listens for that duration, saves the audio as a temporary .wav file, and prepares it for processing.
- It checks if a microphone is connected by listing available input devices. If none are found, it alerts you to plug in a microphone or upload a file instead.
- **Upload an Audio File**:
- You can upload a pre-recorded .wav or .mp3 file. This is useful if you already have audio you want to transcribe.
- The system expects you’re using an environment like Google Colab or Jupyter Notebook, which has a file upload feature. If you’re running the script elsewhere, you’d need to modify this part to point to a file on your computer (e.g., `C:\audio\myfile.wav`).

### Step 4: Clean the Audio
Background noise (like a fan, traffic, or people talking) can make transcription less accurate. The system uses the `noisereduce` library to clean the audio:
- It analyzes the audio to identify noise patterns (e.g., a constant hum).
- It removes or reduces those noises, leaving the speech clearer.
- This step happens automatically before transcription, so Whisper gets the cleanest possible audio.

### Step 5: Transcribe the Audio
Whisper processes the cleaned audio and converts it to text:
- It breaks the audio into small segments (like individual sentences or phrases).
- For each segment, it provides:
- The transcribed text (what was said).
- Start and end timestamps (e.g., “0.00s - 2.50s” for when that text was spoken).
- The timestamps are useful for applications like subtitling videos or analyzing conversations.
- Whisper supports many languages, so it can transcribe audio in English, Spanish, French, and more, automatically detecting the language.

### Step 6: Play Back the Audio
After transcription, the system lets you listen to the cleaned audio using the `ipython` library’s `Audio` feature. This is helpful to:
- Confirm the audio sounds clear after noise reduction.
- Verify that the transcription matches what you hear.
- The playback works best in environments like Jupyter Notebook or Google Colab, where you can see an audio player widget.

## 📌 Usage Instructions
Here’s how to use the system once everything is set up:
1. **Prepare Your Environment**:
- Ensure Python, FFmpeg, and the required libraries are installed (see Step 1 above).
- Use an environment like Jupyter Notebook or Google Colab for the best experience, as they support interactive features like file uploads and audio playback.
2. **Run the Script**:
- You’ll need a Python script that implements the steps above (loading Whisper, recording/uploading, cleaning, transcribing, and playing back). You can write this based on the logic described or find a similar script online.
3. **Choose an Input Mode**:
- When prompted, type `1` to upload a .wav or .mp3 file, or `2` to record from your microphone.
- If recording, enter the duration (e.g., `5` for 5 seconds).
4. **Wait for Processing**:
- If recording, the system captures audio and saves it as a temporary .wav file.
- If uploading, you’ll select a file from your computer.
- The system cleans the audio, transcribes it, and shows the text with timestamps.
5. **Review the Output**:
- Read the transcribed text, which shows each segment with start/end times (e.g., `[0.00s - 2.50s]: Hello, how are you?`).
- Listen to the cleaned audio to ensure it matches the transcription.

## 🧰 Step-by-Step Guide to Install FFmpeg on Windows
Since FFmpeg is critical and can be tricky on Windows, here’s a detailed recap:
1. **Download FFmpeg**:
- Visit https://www.gyan.dev/ffmpeg/builds/.
- Download `ffmpeg-release-essentials.zip` or `ffmpeg-n7.0-latest-win64-gpl.zip`.
2. **Extract the File**:
- Unzip to a folder like `C:\ffmpeg`.
- Inside, find the `bin` folder (e.g., `C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin`).
3. **Add to PATH**:
- Right-click **This PC** → **Properties** → **Advanced system settings** → **Environment Variables**.
- Under **System variables**, select **Path**, click **Edit**, then **New**.
- Paste the `bin` folder path (e.g., `C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin`).
- Click **OK** to save.
4. **Verify Installation**:
- Open a new Command Prompt and type:
ffmpeg -version

- You should see version details. If not, check the PATH or restart your computer.
5. **Fallback Option**:
- If FFmpeg still doesn’t work, you can tell your Python script where to find it by adding this line at the start:

        import os
        os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"


## 🛠 Troubleshooting
If something goes wrong, here are common issues and fixes:
- **Microphone Not Working**:
- Ensure your microphone is plugged in and enabled in your computer’s sound settings.
- Test it in another app (e.g., Windows Voice Recorder or Zoom).
- Check that `sounddevice` detects an input device (the system lists available devices when it starts).
- **Sounddevice Installation Fails**:
- Upgrade `pip` with:
python -m pip install --upgrade pip


- Install PortAudio (see “Install Python Libraries” above).
- **Whisper Can’t Process Audio**:
- Ensure FFmpeg is installed and in your PATH (run `ffmpeg -version` to check).
- Verify the audio file is in a supported format (.wav or .mp3).
- **Poor Transcription Quality**:
- Record in a quiet environment to minimize background noise.
- Speak clearly and close to the microphone.
- Try a longer recording duration (e.g., 10 seconds instead of 5).
- Ensure the audio is clean; check the playback to confirm noise reduction worked.

## Additional Tips
- **Environment**: Jupyter Notebook or Google Colab is ideal because they support interactive features like file uploads and audio playback. If using a regular Python script, you may need to modify the file upload part to point to a local file path.
- **Audio Quality**: For best results, use a good microphone and record in a quiet space. Background noise like music or conversations can confuse Whisper.
- **Model Choice**: The “base” Whisper model is fast but not perfect. If you have a powerful computer, try the “small” or “medium” model for better accuracy (update the model name in the script).
- **File Cleanup**: The system creates temporary .wav files for recordings. These aren’t deleted automatically, so you may want to clean them up manually from your temporary folder (e.g., `C:\Users\YourName\AppData\Local\Temp` on Windows).

## Why This System is Useful
This speech-to-text system is versatile and can be used for:
- Creating subtitles for videos by transcribing audio with timestamps.
- Taking notes from meetings or lectures without typing.
- Converting old audio recordings into searchable text.
- Building voice-controlled applications (with additional coding).

By following this guide, you’ve set up a powerful tool that combines cutting-edge AI (Whisper) with practical audio processing, all from scratch. If you hit any roadblocks, revisit the troubleshooting section or check online communities for Python or Whisper support.
