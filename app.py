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
            {"role": "user", "content": f"Summarize the following text in {language_code}.\n\n{transcript}"}
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

def youtube_summary():
    """Handles the summarization of a YouTube video transcript."""
    youtube_url = request.form.get("youtube_url")
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400

    try:
        transcript, language_code = get_transcript(youtube_url)
        summary = summarize_with_openai(transcript, language_code)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles question-answering based on the transcript."""
    youtube_url = request.form.get("youtube_url")
    question = request.form.get("question")

    if not youtube_url or not question:
        return jsonify({"error": "YouTube URL and Question are required"}), 400

    try:
        # Step 1: Get transcript and split into chunks
        transcript, _ = get_transcript(youtube_url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)

        # Step 2: Load or create vector store
        store_name = "vector_store"
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        # Step 3: Perform similarity search and generate response
        docs = vector_store.similarity_search(question, k=3)
        llm = OpenAI(model_name="gpt-3.5-turbo")
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
