import streamlit as st
from langchain_helper import create_vector_db
from langchain_helper import get_qa_chain

st.title("Course QA")

btn = st.button("Create knowledgebase")

if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])
