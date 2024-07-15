import time
import streamlit as st
import logging
import sys

st.set_page_config(
    page_title="AgentD",
    page_icon="./public/dickinson_favicon.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

st.title("AgentD v0.1")

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="./public/dickinson_picture.png" if message["role"] == "assistant" else None):
        st.write(message["content"])

if not st.session_state.messages:
    init = "My name is AgentD. I'm an AI Agent designed to help with Dickinson College related inquiries!"
    st.session_state.messages.append({"role": "assistant", "content": init})
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        st.write(init)

prompt = st.chat_input("Your question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        time.sleep(2)  
        response = "This is a test message."

    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


