# Imports the streamlit library for building web applications.
import streamlit as st

# Imports the os library to interact with the operating system, useful for file handling and environment variable access.
import os

# Imports the openai library for interacting with OpenAI's API, allowing you to use GPT models.
import openai

# Imports the YouTubeTranscriptApi library for fetching transcripts from YouTube videos.
from youtube_transcript_api import YouTubeTranscriptApi

# Imports functions from dotenv to load environment variables from a .env file, useful for managing API keys.
from dotenv import load_dotenv, find_dotenv

# Imports pickle for serialization and deserialization of Python objects, useful for saving and loading data.
import pickle

# Imports a text splitter class from langchain to split text into manageable chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imports the OpenAIEmbeddings class to generate embeddings for text, allowing vectorization for AI operations.
from langchain.embeddings.openai import OpenAIEmbeddings

# Imports FAISS for creating and using vector-based data stores for efficient similarity search.
from langchain.vectorstores import FAISS

# Imports the OpenAI class to work with OpenAI's language models, such as GPT-3.5.
from langchain.llms import OpenAI

# Imports the load_qa_chain function to create a question-answering chain with OpenAI's language models.
from langchain.chains.question_answering import load_qa_chain

# Imports the get_openai_callback function, allowing you to track OpenAI API usage and costs.
from langchain.callbacks import get_openai_callback



# Initializes a default session state variable current_page to track which page the app should display.
# Set a default state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"


# Defines a function set_page to update the current page in the session state.
# Navigation Function
def set_page(page):
    st.session_state.current_page = page


# Adds buttons to the sidebar for navigating between pages, using the set_page function to switch between "page1" and "page2".
# Sidebar Navigation
st.sidebar.button("Go to Page 1", on_click=set_page, args=("page1",))
st.sidebar.button("Go to Page 2", on_click=set_page, args=("page2",))



# Defines a function get_transcript to fetch the transcript from a given YouTube URL. Extracts the video ID from the URL.
def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    # Fetches a list of available transcripts for the given video ID.
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Attempts to find a manually created transcript and retrieves its language code.
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
            # If neither manual nor auto-generated transcripts are found, it raises an exception.
            raise Exception("No suitable transcript found.")

    # Joins all transcript parts into a single string.
    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code  # Return both the transcript and detected language



# Defines a function only_transcript to fetch just the transcript (without language code) from a YouTube URL.
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



# Defines a function summarize_with_langchain_and_openai to summarize a given transcript using OpenAI. 
# It uses a text splitter to divide the transcript into chunks.
def summarize_with_langchain_and_openai(transcript, language_code, model_name='gpt-3.5-turbo'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    # Splits the transcript into smaller texts and takes the first few parts to summarize.
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4]) # Adjust this as needed

    # Prepare the prompt for summarization
    # Defines the system and user prompts for the OpenAI model
    # Specifying that the model should act as a life coach and summarize the text in the detected language.
    system_prompt = 'I want you to act as a Life Coach that can create good summaries!'
    prompt = f'''Summarize the following text in {language_code}.
    Text: {text_to_summarize}

    Add a title to the summary in {language_code}. 
    Include an INTRODUCTION, BULLET POINTS if possible, and a CONCLUSION in {language_code}.'''

    # Sends the prompt to OpenAI's chat completion endpoint to generate the summary.
    # Start summarizing using OpenAI
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        temperature=1
    )
    
    return response['choices'][0]['message']['content']     # Returns the generated summary from the response.




# Page Routing Logic
# Checks if the current page in the session state is "home".
if st.session_state.current_page == "home":
    # Set the background image
    # Defines a background image for the "home" page using custom CSS.
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/3473569/pexels-photo-3473569.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-size: cover;
    }
    </style>
    '''

    # Applies the background image and sets the page title for the home page.
    st.markdown(page_bg_img, unsafe_allow_html=True)
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



# Checks if the current page in the session state is "page1".
elif st.session_state.current_page == "page1":
    # Defines a different background image for "page1".
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1500236861371-c749e9a06b46?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    }
    </style>
    '''

    # Applies the new background image to "page1".
    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Specifies the path to the .env file containing OpenAI's API key
    env_path = '/home/USER/.env/openai_api'

    # Loads environment variables and sets OpenAI's API key for interaction with the OpenAI API.
    load_dotenv(env_path)
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Defines the main function for "page1", setting the title and providing a text input field to enter a YouTube link.
    def main():
        st.title('YouTube video summarizer')
        link = st.text_input('Enter the link of the YouTube video you want to summarize:')

        # When the "Start" button is pressed, checks if a link is entered. 
        # If so, initializes a progress bar and a status text to show progress.
        if st.button('Start'):
            if link:
                try:
                    progress = st.progress(0)
                    status_text = st.empty()

                    # Updates the status text and progress bar to indicate that the transcript is being loaded.
                    status_text.text('Loading the transcript...')
                    progress.progress(25)

                    # Calls get_transcript to fetch the YouTube transcript and language code.
                    transcript, language_code = get_transcript(link)

                    # Updates the status text and progress bar to indicate that a summary is being created.
                    status_text.text(f'Creating summary...')
                    progress.progress(75)

                    # Uses OpenAI's GPT-3.5 model to summarize the transcript.
                    model_name = 'gpt-3.5-turbo'
                    summary = summarize_with_langchain_and_openai(transcript, language_code, model_name)

                    # Displays the generated summary and completes the progress bar.
                    status_text.text('Summary:')
                    st.markdown(summary)
                    progress.progress(100)

                    # Fetches only the transcript and splits it into smaller chunks for further processing.
                    text = only_transcript(link)
                    
                    text_splitter =RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks=text_splitter.split_text(text=text)

                    # Specifies the name for storing the vector store.
                    store_name="sharon"

                    # Creates an OpenAI-based embeddings object and initializes a FAISS vector store with the chunks.
                    embeddings=OpenAIEmbeddings()
                    VectorStore=FAISS.from_texts(chunks,embedding=embeddings)

                    # Saves the vector store to a file for future use, reducing redundant computation.
                    with open(f"{store_name}.pkl","wb") as f:
                        pickle.dump(VectorStore,f)

                # Catches and displays any exceptions that occur during the summarization process.
                except Exception as e:
                    st.write(str(e))

            # If no link is entered, prompts the user to enter a valid YouTube link.
            else:
                st.write('Please enter a valid YouTube link.')

    # Runs the main function when this script is executed as the main module.
    if __name__ == "__main__":
        main()



# Checks if the current page in the session state is "page2".
elif st.session_state.current_page == "page2":
    # Defines a different background image for "page2".
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://en.idei.club/pic1/uploads/posts/2023-12/thumbs/1702371242_en-idei-club-p-wood-colour-background-dizain-vkontakte-8.jpg");
    background-size: cover;
    }
    </style>
    '''

    # Applies the new background image to "page2".
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Loads environment variables from the default .env file.
    load_dotenv()

    # Defines the main function for "page2" and sets the page title.
    def main():
        st.title('Ask Questions here -')

        # Provides a text input field for the user to enter a YouTube link.
        link = st.text_input('Enter the link of the YouTube video you want to summarize:')

        # Fetches only the transcript from the given YouTube link.
        text = only_transcript(link)
        
        # Splits the transcript into smaller chunks for further processing.
        text_splitter =RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)

        # Specifies the name for storing the vector store.
        store_name="sharon"

        # If a previously saved vector store exists, it loads it from the file.
        # If the vector store does not exist, it creates a new one from the chunks and saves it to a file.
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
                
        # Provides a text input field for users to ask questions.
        query=st.text_input("Ask questions")
        
        # If a query is entered, it performs a similarity search on the vector store to find the most relevant documents.
        if query:
            docs=VectorStore.similarity_search(query=query, k=3)

            # Initializes the OpenAI GPT-3.5 language model and creates a question-answering chain.
            llm=OpenAI(model_name="gpt-3.5-turbo-instruct")
            chain=load_qa_chain(llm=llm,chain_type="stuff")

            # Runs the question-answering chain with the retrieved documents and the user's question. 
            # Uses a callback to track OpenAI API usage.
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
                # Prints the callback information and displays the response to the user's question.
                print(cb)
            st.write(response)

    # Runs the main function when this script is executed as the main module.
    if __name__=="__main__":
        main()