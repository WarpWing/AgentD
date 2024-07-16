import time
import streamlit as st
import logging
import sys
import os
import cohere

# Cohere Auth Setup
os.environ["COHERE_API_KEY"] = ""
co = cohere.Client(os.environ["COHERE_API_KEY"])

# Set up Streamlit page configuration
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

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="./public/dickinson_picture.png" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# Initialize with a welcome message if no messages exist
if not st.session_state.messages:
    init = "My name is AgentD. I'm an AI Agent designed to help with Dickinson College related inquiries!"
    st.session_state.messages.append({"role": "assistant", "content": init})
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        st.markdown(init)

# Get user input
prompt = st.chat_input("Your question")

if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Cohere's chat stream
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Create a spinner in the same container as the message
        with st.spinner("Thinking..."):
            stream = co.chat_stream(
                model='command-r-plus',
                message=prompt,
                temperature=0.3,
                prompt_truncation='AUTO',
                connectors=[{"id":"web-search","options":{"site":"dickinson.edu"}}]
            )

            for event in stream:
                if event.event_type == "text-generation":
                    full_response += event.text
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

        message_placeholder.markdown(full_response)

    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})