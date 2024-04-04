import os
from constants import openai_key
from langchain_community.llms import GooglePalm
from langchain_core.prompts import PromptTemplate 
from langchain.chains import RetrievalQA
from dotenv import load_dotenv 
import streamlit as st 
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
    

# Load environment variables from .env file
load_dotenv()

# Create an instance of GooglePalm model
llm = GooglePalm(google_api_key=os.environ['API_KEY'], temperature=0.3)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load data from CSV

vectordb_file_path = "faiss_data"

def create_vector_db():
    loader = CSVLoader(file_path='code_data.csv', encoding='utf-8')
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(vectordb_file_path)

def QA_get_chain():

    # Store embeddings into vector database
    db = FAISS.load_local(vectordb_file_path, embeddings,allow_dangerous_deserialization=True)

    # Make database for retrieving documents
    retriever = db.as_retriever(score_threshold = 0.7) 


    #  Create prompt template
    prompt_template = """Given the context and question, please try to generate the answer based on given source document,
                         If it is present in the document, give answer, otherwise say no.

    CONTEXT = {context}
    QUESTION = {question} """

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=['context', 'question']
    
    )

    #  Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True, 
        chain_type="stuff", 
        chain_type_kwargs={"prompt": prompt},
        )

    return chain

if __name__ == "__main__":
    # create_vector_db()
    st.title('Langchain demo with GooglePalm')
    input_text = st.text_input("Enter your query: ")

    chain = QA_get_chain()
    result = chain("import torch")
    st.write(result)

    # if st.button("Submit"):
    #     chain1 = QA_get_chain(input_text)
    #     result = chain1(input_text)
    #     if result:
    #         st.write("Answer:", result)
    #     else:
    #         st.write("No answer found in the data for the given query.")

    # chain = QA_get_chain(input_text)
    # result = chain()
    # st.write(chain)
