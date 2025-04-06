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


app = Flask(__name__)


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")



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

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/youtube_summary")
def youtube_summary():
    return render_template("youtube_summary.html")

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



@app.route("/upload_videos")
def upload_videos():
    return render_template("upload_videos.html")

@app.route("/live_webcam")
def live_webcam():
    return render_template("live_webcam.html")

if __name__ == "__main__":
    app.run(debug=True)
