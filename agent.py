import time
import streamlit as st
import logging
import sys
st.set_page_config(page_title=f"AgentD", page_icon="🌎", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(f"AgentD")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [{"role": "assistant", "content": "My name is AgentD. I'm an Chat Agent designed to assist with Dickinson related queries with a focus on sustainability."}
    ]

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = int(5) # Create ReAct Agent 

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Main Chat Logic
if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            res_box = st.empty()  # Placeholder for the response text
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.stream_chat(prompt)
                full_response = ""
                for token in response.response_gen:
                    time.sleep(0.0001)
                    full_response += "".join(token)
                    res_box.write(full_response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)