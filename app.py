from flask import Flask, render_template, request, jsonify, session

from flask import Flask, request, jsonify, render_template
import os
import openai
import pickle
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

from flask import Blueprint, render_template, request, jsonify, Response
import openai
import os
import tempfile
import json
from moviepy import *


app = Flask(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    """Fetches the transcript of a YouTube video."""
    try:
        video_id = youtube_url.split("v=")[-1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching the manual transcript
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0] if generated_transcripts else None

        if not transcript:
            raise Exception("No suitable transcript found.")

        full_transcript = " ".join([part["text"] for part in transcript.fetch()])
        return full_transcript, transcript.language_code
    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")


def summarize_with_openai(transcript, language_code, model_name="gpt-3.5-turbo"):
    """Summarizes the transcript using OpenAI."""
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
        # Get the video transcript and summarize it
        transcript, language_code = get_transcript(youtube_url)
        summary = summarize_with_openai(transcript, language_code)

        # Save the transcript for later use, uniquely identified by video ID
        video_id = youtube_url.split("v=")[-1]
        with open(f"{video_id}_transcript.pkl", "wb") as f:
            pickle.dump(transcript, f)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import pickle
from googletrans import Translator

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles question-answering based on the transcript."""
    youtube_url = request.form.get("youtube_url")
    question = request.form.get("question")

    if not youtube_url or not question:
        return jsonify({"error": "YouTube URL and Question are required"}), 400

    try:
        # Step 1: Load the transcript specific to the video
        video_id = youtube_url.split("v=")[-1]
        transcript_file = f"{video_id}_transcript.pkl"
        if not os.path.exists(transcript_file):
            return jsonify({"error": "Transcript for this video is not available. Summarize the video first."}), 400

        with open(transcript_file, "rb") as f:  # Replace 'transcript_file.pkl' with the actual file name
            transcript = pickle.load(f)

        # Step 2: Convert transcript to a string (if it's not already)
        transcript_text = str(transcript)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # You can also use 'gpt-3.5-turbo'
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text into English."},
                    {"role": "user", "content": f"Translate the following text to English: {transcript_text}"}
                ]
            )
            translated = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error during translation: {e}")
            translated = transcript_text  # Fallback to the original transcript

        # Step 5: Write the translated transcript to abc.txt
        with open('abc.txt', 'w', encoding='utf-8') as f:
            f.write(translated)
        
        # Step 2: Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)

        # Step 3: Use a video-specific vector store
        vector_store_file = f"{video_id}_vector_store.pkl"
        if os.path.exists(vector_store_file):
            with open(vector_store_file, "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()  # Replace with HuggingFaceEmbeddings if needed
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(vector_store_file, "wb") as f:
                pickle.dump(vector_store, f)

        # Step 4: Perform similarity search and generate response
        docs = vector_store.similarity_search(question, k=3)
        llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use ChatOpenAI for chat models
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
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            file.save(temp_video_file.name)
            video_path = temp_video_file.name

        # Extract audio from video
        audio_path = video_path.replace('.mp4', '.mp3')
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        # Transcribe using Whisper
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )

        raw_text = transcript["text"]

        # Step 1: Translate and clean content into meaningful English using GPT
        translation_prompt = f"""You are a helpful assistant. Translate the following transcript into fluent English:\n\n{raw_text}"""
        translation_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates transcripts."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.7
        )

        translated_text = translation_response['choices'][0]['message']['content'].strip()

        # Step 2: Summarize translated English content using same summarize_with_openai() logic
        summary = summarize_with_openai(translated_text, language_code="English")

        # Construct dynamic pickle filename
        video_filename = file.filename  # original uploaded filename
        base_filename = os.path.splitext(video_filename)[0]  # without extension
        pickle_filename = f"{base_filename}_transcript.pkl"

        # Save transcript
        with open(pickle_filename, "wb") as f:
            pickle.dump(translated_text, f)

        # Save pickle filename in session
        session['upload_pickle_filename'] = pickle_filename

        # Clean up
        os.remove(video_path)
        os.remove(audio_path)

        # Final response with clean summary
        response_json = json.dumps({'summary': summary}, ensure_ascii=False)
        return Response(response_json, content_type='application/json; charset=utf-8')

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

        # Step 2: Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)

        # Step 3: Load or create vector store
        vector_store_file = f"{os.path.splitext(pickle_filename)[0]}_vector_store.pkl"
        if os.path.exists(vector_store_file):
            with open(vector_store_file, "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(vector_store_file, "wb") as f:
                pickle.dump(vector_store, f)

        # Step 4: Q&A
        docs = vector_store.similarity_search(question, k=3)
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'  # Required for session to work
    app.run(debug=True)
