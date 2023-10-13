import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Load the pre-trained language model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function for file loading and preprocessing
def file_preprocessing(file):
    # Use the PyPDFLoader to load the PDF file
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    final_texts = ""
    for text in pages:
        final_texts = final_texts + text.page_content
    return final_texts

# Language model (LLM) pipeline for summarization
def llm_pipeline(filepath):
    # Create a summarization pipeline using the loaded model and tokenizer
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )

    # Preprocess the file and get the input text
    input_text = file_preprocessing(filepath)

    # Generate the summary using the pipeline
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Cache the PDF display function to improve performance
@st.cache_data
def displayPDF(file):
    # Open the file from the file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embed the PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Display the PDF file
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
