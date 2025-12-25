from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import main
from dotenv import load_dotenv
main.load_dotenv()
import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

os.environ['LANGCHAIN_TRACING_V2'] = "true" # Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Defining Prompt Template
prompt = ChatPromptTemplate.from_messages([("system","You are a helpful assistant. Please respond to the user queries"),
                                           ("user","Question:{question}")])

# Streamlit Frame work
st.title("LangChain Demo with OPEN AI API")
input_text = st.text_input("Search the topic you want")

# Open AI LLMs
# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = OllamaLLM(model="llama2")

# Output Parsers
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

