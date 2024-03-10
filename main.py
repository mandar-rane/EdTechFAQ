import streamlit as st
from langchain_helper import create_vector_db
from langchain_helper import get_qa_chain

st.title("Educational Course QnA 🌱")
url = "https://github.com/mandar-rane"
st.text("- Mandar Rane")
st.write("GitHub: (%s)" % url)

# btn = st.button("Create knowledgebase")
# if btn:
#     pass
st.text("Example questions: \nQ. What are the prerequisites of this course?\nQ. What is the duration of this course?\nQ. I don't have a laptop, can I take this course?")


question = st.text_input("Question (Press Enter for answer): ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])
