from flask import Blueprint, render_template, request, jsonify
import openai
import os
import tempfile
from moviepy import *

video_bp = Blueprint("video", __name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@video_bp.route("/upload_videos")
def upload_page():
    return render_template("upload_videos.html", title="Upload Videos")

@video_bp.route("/process_video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            file.save(temp_video_file.name)
            video_path = temp_video_file.name

        audio_path = video_path.replace(".mp4", ".mp3")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        os.remove(video_path)
        os.remove(audio_path)

        return jsonify({"summary": transcript.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
