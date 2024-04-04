import os
import streamlit as st
# import reportlab

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from constants import openai_key
from langchain_community.llms import GooglePalm
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
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
    db = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Make database for retrieving documents
    retriever = db.as_retriever(score_threshold=0.7)

    # Create prompt template
    prompt_template = """Given the context and question, please try to generate the answer based on given source document,
                         If it is present in the document, give answer, otherwise say no.

    CONTEXT = {context}
    QUESTION = {question} """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def generate_pdf(result):
    # Extract individual components from the result dictionary
    query = result.get('query', '')
    # answer = result.get('result', '')
    source_documents = result.get('source_documents', [])

    # Generate PDF using ReportLab
    pdf_path = "result.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 770, "Query:")
    c.drawString(150, 770, query)
    # c.drawString(100, 750, "Answer:")
    # c.drawString(150, 750, answer)
    c.drawString(100, 730, "Source Documents:")

    # Add a blank string for a new line
    c.drawString(100, 710, "")

    # Write source documents starting from the new line
    y_coordinate = 690  # Initial vertical position for source documents
    for i, document in enumerate(source_documents):
        if isinstance(document, str):
            lines = document.split('\n')  # Split long string into lines
        else:
            lines = str(document).split('\n')  # Convert to string and split
        for j, line in enumerate(lines):
            if j == 0:
                c.drawString(80, y_coordinate - i*10, f"{i+1}. {line}")
            else:
                # Adjust font size dynamically based on text length
                font_size = min(8, 1200 // len(line))  # Adjust maximum font size as needed
                c.setFont("Helvetica", font_size)
                c.drawString(120, y_coordinate - i*5- j*5, line)
            # Adjust the y-coordinate for the next line
            if (i+1) % 3 == 0:
                y_coordinate -= 5  # Adjust for a new line
    
    c.showPage()
    c.save()
    return pdf_path


if __name__ == "__main__":
    # create_vector_db()
    st.title('Langchain demo with GooglePalm')
    input_text = st.text_input("Enter your query: ")

    chain = QA_get_chain()
    result = chain("import torch")
    st.write(result)

    if st.button("Generate PDF"):
        pdf_path = generate_pdf(result)
        st.write("PDF generated successfully!")
        st.markdown(f"Download the PDF [here](/{pdf_path})")
