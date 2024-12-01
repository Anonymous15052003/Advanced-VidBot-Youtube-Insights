import streamlit as st
import os
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv, find_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


# Set a default state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"


# Navigation Function
def set_page(page):
    st.session_state.current_page = page

# Sidebar Navigation
st.sidebar.button("Go to Page 1", on_click=set_page, args=("summary",))
st.sidebar.button("Go to Page 2", on_click=set_page, args=("chatbot",))



def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try fetching the manual transcript
    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code  # Save the detected language
    except:
        # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code  # Save the detected language
        except:
            # If no auto-generated transcript is found, raise an exception
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code  # Return both the transcript and detected language



def only_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try fetching the manual transcript
    try:
        transcript = transcript_list.find_manually_created_transcript()
    except:
        # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
        except:
            # If no auto-generated transcript is found, raise an exception
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript  # Return the transcript



def summarize_with_langchain_and_openai(transcript, language_code, model_name='gpt-3.5-turbo'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4]) # Adjust this as needed

    # Prepare the prompt for summarization
    system_prompt = 'I want you to act as a Life Coach that can create good summaries!'
    prompt = f'''Summarize the following text in {language_code}.
    Text: {text_to_summarize}

    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    # Start summarizing using OpenAI
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=1
    )
    
    return response['choices'][0]['message']['content']




# Page Routing Logic
if st.session_state.current_page == "home":
    
    # Set the background image
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/3473569/pexels-photo-3473569.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Page Title
    st.title("Welcome to Our Project")  # Title for the home page

    # Highlighting Project Features
    st.markdown("""
    ## Key Features
    - **Advanced AI Integration**: Utilizes OpenAI GPT-3.5 for text summarization and question answering, providing intelligent responses.
    - **YouTube Transcript Extraction**: Fetches and processes transcripts from YouTube videos for easy analysis and interaction.
    - **Flexible Page Navigation**: Dynamic switching between different functionalities without reloading the app.
    - **Interactive User Inputs**: Allows users to submit YouTube links and ask questions for a more engaging experience.
    - **Real-Time Progress Indicators**: Visual feedback with progress bars and status updates to keep users informed.
    - **Multifunctional Capabilities**: Offers YouTube video summarization and question answering based on video transcripts.
    - **Robust Error Handling**: Includes fallback mechanisms to maintain smooth operation in various scenarios.
    - **Persistent Data Storage**: Saves vector store data for reuse, reducing redundant computations.
    - **Scalable Design**: Built with a modular approach, allowing easy modification and scalable cloud-based resources with OpenAI.
    """)



elif st.session_state.current_page == "summary":
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1500236861371-c749e9a06b46?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Specify the path to your .env file
    env_path = '/home/USER/.env/openai_api' # Change the Path
    # Load the OpenAI API key from the .env file
    load_dotenv(env_path)
    openai.api_key = os.getenv('OPENAI_API_KEY')

    def main():
        st.title('YouTube video summarizer')
        link = st.text_input('Enter the link of the YouTube video you want to summarize:')

        if st.button('Start'):
            if link:
                try:
                    progress = st.progress(0)
                    status_text = st.empty()

                    status_text.text('Loading the transcript...')
                    progress.progress(25)

                    # Getting both the transcript and language_code
                    transcript, language_code = get_transcript(link)

                    status_text.text(f'Creating summary...')
                    progress.progress(75)

                    model_name = 'gpt-3.5-turbo'
                    summary = summarize_with_langchain_and_openai(transcript, language_code, model_name)

                    status_text.text('Summary:')
                    st.markdown(summary)
                    progress.progress(100)

                    text = only_transcript(link)
                    
                    text_splitter =RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks=text_splitter.split_text(text=text)

                    store_name="sharon"

                    embeddings=OpenAIEmbeddings()
                    VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
                    with open(f"{store_name}.pkl","wb") as f:
                        pickle.dump(VectorStore,f)

                except Exception as e:
                    st.write(str(e))
            else:
                st.write('Please enter a valid YouTube link.')

    if __name__ == "__main__":
        main()



elif st.session_state.current_page == "chatbot":
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://en.idei.club/pic1/uploads/posts/2023-12/thumbs/1702371242_en-idei-club-p-wood-colour-background-dizain-vkontakte-8.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    load_dotenv()

    def main():
        st.title('Ask Questions here -')
        link = st.text_input('Enter the link of the YouTube video you want to summarize:')
        text = only_transcript(link)
        
        text_splitter =RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)

        store_name="sharon"

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)

        query=st.text_input("Ask questions")
        
        if query:
            docs=VectorStore.similarity_search(query=query, k=3)

            llm=OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

    if __name__=="__main__":
        main()