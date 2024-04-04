import os 
from constants import openai_key 
from langchain.llms import GooglePalm
from langchain import PromptTemplate 
from langchain.chains import LLMChain
import streamlit as st 
from constants import openai_key
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

# os.environ['OPENAI_API_KEY'] = openai_key

# step 1: interface designed using streamlit
# streamlit framework 
st.title('Langchain demo with GooglePalm')
input = st.text_input("Search the topic you want")


# step 2: getting api key for GooglePalm LLMs model
# api key 
api_key = "AIzaSyBkIxznvzMOl84OIaSEU6YOhQXNNlmuDAg" 
llm = GooglePalm(google_api_key = api_key,temperature = 0.3)


# step 3: creeating prompt template 

# PromptTemplate
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about {name}"
)

# we can create multiple templates and chains depend on our requirements, but for multiple templates we have to make parent  chain , for that use {SimpleSequentialChain}


# drawback of {SimpleSequentialChain} = can't get all data history(provide information about last prompt template) """

# memory
person_memory = ConversationBufferMemory(input_key = 'name', memory_key = 'chat_history')
dob_memory = ConversationBufferMemory(input_key = 'person', memory_key = 'chat_history')
# desc_memory = ConversationBufferMemory(input_key = 'dob', memory_key = 'chat_history')


# chain 
chain = LLMChain(llm=llm,prompt = first_input_prompt, verbose=True,output_key = 'person', memory = person_memory)

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "Tell me about the time when {person} born"
)

chain2 = LLMChain(llm=llm,prompt = second_input_prompt, verbose=True,output_key = 'dob', memory = dob_memory)

# parent_chain = SimpleSequentialChain(
#     chains=[chain,chain2], 
#     verbose=True
# )

parent_chain = SequentialChain(
    chains=[chain,chain2], 
    input_variables = ['name'], 
    output_variables=['person','dob'],
    verbose=True
)


if input :
    # st.write(parent_chain.run(input))
    st.write(parent_chain({'name':input}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('DOB '):
        st.info(dob_memory.buffer)




