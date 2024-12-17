from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API Key: {openai.api_key[:5]}********")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/youtube_summary")
def youtube_summary():
    return render_template("youtube_summary.html")

@app.route("/upload_videos")
def upload_videos():
    return render_template("upload_videos.html")

@app.route("/live_webcam")
def live_webcam():
    return render_template("live_webcam.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400

    try:
        transcript, language_code = get_transcript(youtube_url)
        summary = summarize_with_openai(transcript, language_code)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Other function definitions like get_transcript, summarize_with_openai, etc.

if __name__ == "__main__":
    app.run(debug=True)
