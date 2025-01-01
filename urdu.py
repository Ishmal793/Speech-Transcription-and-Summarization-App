import os
import time
import tempfile
import requests
import textwrap
from fpdf import FPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import streamlit as st
import whisper
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from PyPDF2 import PdfReader  # Ensure PyPDF2 is imported
import shutil
from datetime import datetime
import warnings
from pytube import YouTube
from pydub import AudioSegment
import yt_dlp

warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit configuration
st.set_page_config(
    page_title="Speech Transcription and Summarization App",
    page_icon="Logo.jpg",
    layout="wide",
)

# Header Section with Logo
st.markdown(
    """
    <style>
        .main-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background-color: #2c3e50;
            border-radius: 8px;
            color: white;
        }
        .main-header img {
            height: 50px;
            margin-right: 15px;
        }
        .main-header h1 {
            font-size: 1.5rem;
        }
    </style>
    <div class="main-header">
        <img src="Logo1.jpg" alt="Logo">
        <h1>Speech Transcription and Summarization App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Styling
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            background-color: #34495e;
            color: white;
        }
        section[data-testid="stSidebar"] * {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Content
with st.sidebar:
    st.header("Input Options")
    input_type = st.radio(
        "Choose input type:",
        ["Upload Audio or Video File", "Enter URL"],
        index=0,
        key="input_type_radio",
    )
    st.markdown("""---""")
    st.info("Select an audio or video input method.")

# Input Fields
st.markdown(
    """
    <style>
        .input-section {
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
            color: #34495e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='input-section'><h2>Input Selection</h2></div>", unsafe_allow_html=True)

if input_type == "Upload Audio or Video File":
    audio_file = st.file_uploader(
        "Upload Audio or Video File", type=["mp3", "wav", "m4a", "mp4"]
    )
elif input_type == "Enter URL":
    audio_file = st.text_input("Enter URL")

# Process Button with Styling
st.markdown(
    """
    <style>
        .process-btn {
            background-color: #2980b9;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
        }
        .process-btn:hover {
            background-color: #3498db;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

process_button = st.button("Start Processing", key="process_button")

# Results Section
st.markdown(
    """
    <style>
        .results-section {
            margin-top: 20px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 8px;
        }
        .results-section h2 {
            color: #34495e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='results-section'><h2>Results</h2></div>", unsafe_allow_html=True)

# Ensure session state for file downloads
if "transcript_pdf" not in st.session_state:
    st.session_state.transcript_pdf = None
if "summary_pdf" not in st.session_state:
    st.session_state.summary_pdf = None

cookies_file_path = "cookies.txt" 

# Process Audio or Video
if process_button:
    if audio_file is not None:
        if input_type == "Enter URL":
            if is_youtube_link(audio_file):
                st.info("Processing YouTube link. Please wait...")
                try:
                    audio_path = youtube_to_mp3(audio_file, cookies_file=cookies_file_path)  # Convert YouTube to MP3
                    st.success("YouTube link processed successfully!")
                except Exception as e:
                    st.error(f"Error processing YouTube link {e}")
            else:
                audio_path = audio_file  # Use URL directly
        elif input_type == "Upload Audio or Video File":
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as temp_file:
                temp_file.write(audio_file.read())
                audio_path = temp_file.name

        try:
            # Transcription with Whisper
            st.info("Transcribing. Please wait...")
            transcript = transcribe_with_whisper(audio_path)

            # Save Transcription as PDF
            transcript_pdf = f"{audio_file.name}_transcription.pdf" if input_type != "Enter URL" else "transcription.pdf"
            save_to_pdf(transcript, transcript_pdf, title="Transcription")
            st.success("Transcription complete!")
            st.session_state.transcript_pdf = transcript_pdf
            st.text_area("Transcribed Text", value=transcript, height=300)

            # Summarization
            st.info("Generating summary. Please wait...")
            summary = summarize_pdf(transcript_pdf)

            # Save Summary as PDF
            summary_pdf = f"{audio_file.name}_summary.pdf" if input_type != "Enter URL" else "summary.pdf"
            save_to_pdf(summary, summary_pdf, title="Summary")
            st.session_state.summary_pdf = summary_pdf
            st.success("Summary generation complete!")
            st.text_area("Summary", value=summary, height=300)

        except Exception as e:
            st.error(f"Error during processing: {e}")
    else:
        st.warning("Please upload file or URL")

# Download Buttons
if st.session_state.transcript_pdf:
    with open(st.session_state.transcript_pdf, "rb") as f:
        st.download_button(
            "Download Transcription PDF",
            data=f,
            file_name=st.session_state.transcript_pdf,
            mime="application/pdf",
        )

if st.session_state.summary_pdf:
    with open(st.session_state.summary_pdf, "rb") as f:
        st.download_button(
            "Download Summary PDF",
            data=f,
            file_name=st.session_state.summary_pdf,
            mime="application/pdf",
        )
