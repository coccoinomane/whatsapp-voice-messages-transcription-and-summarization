#!/usr/bin/env python3
"""
Transcribes an audio file with GPT-4o-transcribe and generates a summary with o3.

Usage:
    python transcribe_and_summarize.py path/to/audio_file.ext
"""

import os
import sys
import subprocess
from openai import OpenAI
from pathlib import Path


# --------------------- API configuration --------------------- #
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("❌  Error: OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

# --------------------- Argument handling --------------------- #
if len(sys.argv) != 2:
    sys.exit("Usage: python transcribe_and_summarize.py path/to/audio_file.ext")

input_path = Path(sys.argv[1]).expanduser().resolve()
print(input_path)
if not input_path.is_file():
    sys.exit(f"❌  File not found: {input_path}")

# --------------------- OPUS → WAV conversion ----------------- #
SUPPORTED_EXT = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga",
                 ".m4a", ".ogg", ".wav", ".webm"}
ext = input_path.suffix.lower()

if ext == ".opus":
    wav_path = input_path.with_suffix(".wav")
    print(f"🔄  Converting {input_path.name} → {wav_path.name} ...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1", str(wav_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("❌  OPUS conversion failed. Make sure FFmpeg is installed and supports the opus codec.")
    audio_path = wav_path
elif ext in SUPPORTED_EXT:
    audio_path = input_path
else:
    supported = ", ".join(sorted(SUPPORTED_EXT | {'.opus'}))
    sys.exit(f"❌  Unsupported format {ext}. Supported formats: {supported}")

# --------------------- Transcription ------------------------- #
print("📝  Transcribing the audio with gpt-4o-transcribe ...")
with open(audio_path, "rb") as audio_file:
    # transcript_text = client.audio.transcribe(model="gpt-4o-transcribe",
    transcript_text = client.audio.transcriptions.create(model="gpt-4o-transcribe",
    file=audio_file,
    response_format="text")

# Save transcription
transcription_file = input_path.with_name("transcription.md")
transcription_file.write_text("# Transcription\n\n" + transcript_text, encoding="utf-8")
print(f"✅  Transcription saved to {transcription_file}")

# --------------------- Summary with o3 ----------------------- #
print("🧠  Creating summary with o3 ...")
completion = client.chat.completions.create(model="o3",
messages=[
    {"role": "system", "content": "You are an assistant that summarises spoken texts into clear and concise bullet points."},
    {"role": "user",   "content": f"Here is the transcription of an audio recording.\n\n{transcript_text}\n\nGenerate a bullet‑point summary (•), maximum 12 points."}
])
summary_text = completion.choices[0].message.content.strip()

# Save summary
summary_file = input_path.with_name("summary.md")
summary_file.write_text("# Summary\n\n" + summary_text, encoding="utf-8")
print(f"✅  Summary saved to {summary_file}")

print("🎉  Done.")
