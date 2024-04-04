import os 
from constants import openai_key 
from langchain_community.llms import GooglePalm
from langchain import PromptTemplate 
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv 
load_dotenv()

# os.environ['OPENAI_API_KEY'] = openai_key

""" step 1: interface designed using streamlit """

def implementation():
    import streamlit as st 
    st.title('Langchain demo with GooglePalm')
    input = st.text_input("Search the topic you want")


    """ step 2: getting api key for GooglePalm LLMs model """
    from constants import openai_key
    api_key = "AIzaSyBkIxznvzMOl84OIaSEU6YOhQXNNlmuDAg" 
    llm = GooglePalm(google_api_key = os.environ['API_KEY'],temperature = 0.3)


    """ step 3: loading the data """
    from langchain.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(file_path='Data_built.csv',encoding='utf-8')
    data = loader.load() 

    # print(data)
 
    """ step 4: generating embeddgings""" 

    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    """ step 5: storing embeddgings into vector database """
    from langchain_community.vectorstores import FAISS
    db = FAISS.from_documents(data, embeddings)

    """ step 6: making database as for retrieving documents """
    retriever = db.as_retriever() 
    rdocs = retriever.get_relevant_documents("what about library staff member ? ")
    rdocs


    """ step 7: making prompt template """ 
    from langchain.prompts import PromptTemplate 
    prompt_template = """Given the context and question, please try to generate the answer based on given source document,
                        If it is present in the document, give answer, otherwise say no.

    CONTEXT = {context}
    QUESTION = {question} """

    prompt = PromptTemplate(template = prompt_template,
                            input_variables= ['context', 'question'])


    """ step 8: Question and Answering """

    from langchain.chains import RetrievalQA
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        return_source_documents=True, 
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}  # Remove the 'prompt' field from chain_type_kwargs
    )

    return chain  

    # asking question 
    # chain("do you have javascript course ")

if __name__ == "__main__":
    chain = implementation()
    result = chain.run("As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles.")
    st.write(result)

    # if input :
    #     st.write(chain.run("As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles."))
    # st.write(parent_chain({'name':input}))

    # with st.expander('Person Name'):
    #     st.info(person_memory.buffer)

    # with st.expander('DOB '):
    #     st.info(dob_memory.buffer)




