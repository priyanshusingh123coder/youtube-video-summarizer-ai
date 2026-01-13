import gradio as gr
import yt_dlp
import whisper
from transformers import pipeline
import os
import subprocess

# Load models once
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

AUDIO_FILE = "audio.mp3"


# Make sure FFmpeg exists
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def download_audio(url):
    # Delete old audio
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "geo_bypass": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        }
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def summarize_video(url):
    try:
        if not check_ffmpeg():
            return "❌ FFmpeg not installed. Please install FFmpeg and restart."

        download_audio(url)

        if not os.path.exists(AUDIO_FILE):
            return "❌ Audio download failed. Try another video."

        # Transcribe & Translate to English
        result = whisper_model.transcribe(
            AUDIO_FILE,
            task="translate",
            language="en"
        )

        text = result["text"].strip()

        if len(text) < 200:
            return "❌ Not enough speech detected in this video."

        # Break large transcript into chunks
        max_chunk = 1000
        chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

        summaries = []
        for chunk in chunks:
            out = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
            summaries.append(out[0]["summary_text"])

        final_summary = " ".join(summaries)

        return final_summary

    except Exception as e:
        return f"❌ ERROR: {str(e)}"


# Gradio UI
ui = gr.Interface(
    fn=summarize_video,
    inputs=gr.Textbox(label="Enter YouTube URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="GenAI Learniverse – YouTube Video Summarizer",
    description="Summarize ANY YouTube video using Whisper + Transformers (Auto-translated to English)"
)

ui.launch(share=True)
