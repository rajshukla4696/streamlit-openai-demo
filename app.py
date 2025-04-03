import streamlit as st
import pdfplumber
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Azure OpenAI Config
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-02-01"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Function to extract tables from PDF
def extract_tables_from_pdf(pdf_file):
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table)
                df.columns = df.iloc[0]  # Set first row as header
                df = df[1:].reset_index(drop=True)  # Remove header row
                tables.append(df)
    return tables

# Function to query Azure OpenAI using table data
def query_openai_table(prompt, table_data):
    structured_table = tabulate(table_data, headers='keys', tablefmt='plain')  # Convert DataFrame to readable text
    full_prompt = f"Here is a table:\n{structured_table}\n\nUser Question: {prompt}"

    response = openai.ChatCompletion.create(
        engine=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are an AI that answers questions based on tables."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("📊 PDF Table Query App with Azure OpenAI")
st.write("Upload a PDF containing tables and ask questions about them.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting tables..."):
        pdf_tables = extract_tables_from_pdf(uploaded_file)

    if not pdf_tables:
        st.error("No tables found in the PDF!")
    else:
        st.success(f"Extracted {len(pdf_tables)} tables.")
        selected_table_index = st.selectbox("Select a table to query:", list(range(len(pdf_tables))))
        table_df = pdf_tables[selected_table_index]
        
        st.write("### Extracted Table")
        st.dataframe(table_df)

        user_question = st.text_input("Ask a question about the table:")

        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Querying Azure OpenAI..."):
                    answer = query_openai_table(user_question, table_df)
                st.success("Response:")
                st.write(answer)
            else:
                st.warning("Please enter a question!")
