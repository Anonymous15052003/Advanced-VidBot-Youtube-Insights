from flask import Flask, render_template, request, jsonify, session, Response
import os
import openai
import pickle
import json
import tempfile
from moviepy import *
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Required for session to work

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure uploads directory exists
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def landing():
    return render_template('landing.html', title='Welcome')

@app.route('/youtube_summary')
def youtube_summary():
    return render_template('youtube_summary.html', title='YouTube Summary')

@app.route('/upload_videos')
def upload_videos():
    return render_template('upload_videos.html', title='Upload Videos')

@app.route('/live_webcam')
def live_webcam():
    return render_template('live_webcam.html', title='Live Webcam')


def get_transcript(youtube_url):
    try:
        video_id = youtube_url.split("v=")[-1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            generated_transcripts = [t for t in transcript_list if t.is_generated]
            transcript = generated_transcripts[0] if generated_transcripts else None
        if not transcript:
            raise Exception("No suitable transcript found.")
        full_transcript = " ".join([part["text"] for part in transcript.fetch()])
        return full_transcript, transcript.language_code
    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")

def summarize_with_openai(transcript, language_code, model_name="gpt-3.5-turbo"):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize the following text in {language_code} language:\n\n{transcript}"}
        ]
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        raise Exception(f"Error during summarization: {e}")


@app.route("/summarize", methods=["POST"])
def summarize():
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400
    try:
        transcript, language_code = get_transcript(youtube_url)
        summary = summarize_with_openai(transcript, language_code)
        video_id = youtube_url.split("v=")[-1]
        transcript_file = os.path.join(UPLOAD_DIR, f"{video_id}_transcript.pkl")
        with open(transcript_file, "wb") as f:
            pickle.dump(transcript, f)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    youtube_url = request.form.get("youtube_url")
    question = request.form.get("question")
    if not youtube_url or not question:
        return jsonify({"error": "YouTube URL and Question are required"}), 400
    try:
        video_id = youtube_url.split("v=")[-1]
        transcript_file = os.path.join(UPLOAD_DIR, f"{video_id}_transcript.pkl")
        if not os.path.exists(transcript_file):
            return jsonify({"error": "Transcript not found. Summarize the video first."}), 400
        with open(transcript_file, "rb") as f:
            transcript = pickle.load(f)
        transcript_text = str(transcript)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Translate this text into English."},
                    {"role": "user", "content": f"Translate this: {transcript_text}"}
                ]
            )
            translated = response.choices[0].message['content']
        except:
            translated = transcript_text

        with open('abc.txt', 'w', encoding='utf-8') as f:
            f.write(translated)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript_text)

        vector_store_file = os.path.join(UPLOAD_DIR, f"{video_id}_vector_store.pkl")
        if os.path.exists(vector_store_file):
            with open(vector_store_file, "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(vector_store_file, "wb") as f:
                pickle.dump(vector_store, f)

        docs = vector_store.similarity_search(question, k=3)
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            file.save(temp_video_file.name)
            video_path = temp_video_file.name

        audio_path = video_path.replace('.mp4', '.mp3')
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
        raw_text = transcript["text"]

        translation_prompt = f"Translate the following transcript into fluent English:\n\n{raw_text}"
        translation_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates transcripts."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.7
        )
        translated_text = translation_response['choices'][0]['message']['content'].strip()
        summary = summarize_with_openai(translated_text, language_code="English")

        base_filename = os.path.splitext(file.filename)[0]
        transcript_pickle = os.path.join(UPLOAD_DIR, f"{base_filename}_transcript.pkl")
        with open(transcript_pickle, "wb") as f:
            pickle.dump(translated_text, f)

        session['upload_pickle_filename'] = transcript_pickle

        os.remove(video_path)
        os.remove(audio_path)

        return Response(json.dumps({'summary': summary}, ensure_ascii=False), content_type='application/json; charset=utf-8')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/ask_upload", methods=["POST"])
def ask_question_upload():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        pickle_filename = session.get('upload_pickle_filename')
        if not pickle_filename or not os.path.exists(pickle_filename):
            return jsonify({"error": "Transcript not available. Please summarize first."}), 400

        with open(pickle_filename, "rb") as f:
            transcript = pickle.load(f)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)

        vector_store_file = os.path.splitext(pickle_filename)[0] + "_vector_store.pkl"
        if os.path.exists(vector_store_file):
            with open(vector_store_file, "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(vector_store_file, "wb") as f:
                pickle.dump(vector_store, f)

        docs = vector_store.similarity_search(question, k=3)
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
