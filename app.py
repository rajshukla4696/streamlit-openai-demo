import streamlit as st
import fitz  # PyMuPDF
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Config
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to query Azure OpenAI
def query_openai(prompt, document_text):
    response = openai.ChatCompletion.create(
        engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
            {"role": "user", "content": f"Document:\n{document_text}\n\nUser Question: {prompt}"}
        ],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("📄 PDF Query App with Azure OpenAI")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")
    st.text_area("Extracted Text (Preview)", pdf_text[:1000], height=200)

    user_question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Querying Azure OpenAI..."):
                answer = query_openai(user_question, pdf_text[:5000])  # Limit for context size
            st.success("Response:")
            st.write(answer)
        else:
            st.warning("Please enter a question!")
