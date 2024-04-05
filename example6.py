import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Load the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

# Load data from CSV
vectordb_file_path = "faiss_data"

def create_vector_db():
    loader = CSVLoader(file_path='code_data.csv', encoding='utf-8')
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(data, embeddings)
    db.save_local(vectordb_file_path)

def QA_get_chain(question):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

if __name__ == "__main__":
    st.title('T5 Question Answering Demo')
    question = st.text_input("Enter your question: ")

    if st.button("Submit"):
        if question:
            result = QA_get_chain(question)
            st.write("Answer:", result)
        else:
            st.write("Please enter a question.")