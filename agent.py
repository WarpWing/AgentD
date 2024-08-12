import os
import time
import json
import requests
import streamlit as st
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Perplexity API setup
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_KEY")

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

def perplexity_search_stream(query: str):
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are AgentD, a question and answering agent created by Ty Chermsirivatana. You were made to answer questions about Dickinson College using the various tools available to youYou need to be as verbose and detailed as possible. Make sure to mention a specific professor or staff member as a point of contact if the topic has it (like directors or chairs of certain centers of programs).  When giving information about faculty or staff, make sure to mention their department, title, phone number, building and most importantly, their classes (separated in bullet points by Fall and Spring which you should provide in bullet point form)."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    try:
        with requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
    except Exception as e:
        logging.error(f"Error fetching information from Perplexity API: {str(e)}")
        yield f"I apologize, but I encountered an error while fetching information. Please try again later."

# Get user input
prompt = st.chat_input("Your question")

if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Perplexity API stream
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        message_placeholder = st.empty()
        full_response = ""

        # Create a spinner in the same container as the message
        with st.spinner("Thinking..."):
            for chunk in perplexity_search_stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)

        message_placeholder.markdown(full_response)

    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})