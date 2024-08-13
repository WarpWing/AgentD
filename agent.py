import os
import time
import streamlit as st
import logging
import sys
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
import cohere
import requests
import json
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

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
    init = "My name is AgentD. I'm an AI Assistant designed to help with Dickinson College related inquiries!"
    st.session_state.messages.append({"role": "assistant", "content": init})
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        st.markdown(init)

# Initialize external services and clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
firecrawl = FirecrawlApp(api_url=os.getenv("FIRECRAWL_API_ENDPOINT"), api_key="Banana Pudding")
params = {
    'pageOptions': {
        'onlyMainContent': True
    }
}

# Tool functions
def return_faculty_profile(query) -> str:
    f = requests.get('https://www2.dickinson.edu/lis/angularJS_services/directory_open/data.cfc?method=f_getAllPeople').json()
    
    record = None
    parts = query.split()
    
    if len(parts) == 2:
        first_name, last_name = parts
        for r in f:
            if r['FIRSTNAME'].lower() == first_name.lower() and r['LASTNAME'].lower() == last_name.lower():
                record = r
                break
    elif '@' in query:
        email = query
        email_domain = email.split('@')[-1] if '@' in email else ''
        for r in f:
            if r['EMAIL'] == email.split('@')[0] and email_domain == 'dickinson.edu':
                record = r
                break
    
    if record:
        email = record['EMAIL']
        url = f"https://www.dickinson.edu/site/custom_scripts/dc_faculty_profile_index.php?fac={email}"
        
        try:
            profile = firecrawl.scrape_url(url, params=params)
            profile_content = profile["content"]
        except Exception as e:
            profile_content = f"Error fetching faculty profile: {str(e)}"
        
        result = json.dumps({
            "directory_info": record,
            "faculty_profile": profile_content
        }, indent=2)
        return result
    else:
        return "No matching record found."

def dickinson_search_function(query: str) -> str:
    results = ""
    internet_results = DDGS().text(f"{query} site:dickinson.edu", max_results=1)
   
    for result in internet_results:
        url = result["href"]
        firecrawl_content = firecrawl.scrape_url(url, params=params)
        results += f"{firecrawl_content['content']}\n"
    
    return results

# Set up tools
return_faculty_prof = Tool(
    name="return_faculty_profile",
    func=return_faculty_profile,
    description="Return details about a Faculty Member (Professors). You should use this tool when needing information a professor's education, courses they teach or their biography/ personal information. Input should be the faculty member's email address. If you get a First and Last Name, you should use the find_record tool first to get the email address."
)

dickinson_search_tool = Tool(
    name="dickinson_search",
    func=dickinson_search_function,
    description="Get search results on information related to Dickinson College. Should always be used for grounding after exhausting all relevant tools. Input should be a query to search up."
)

tools = [dickinson_search_tool, return_faculty_prof]

def query_agent(query):
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_KEY"), model_name="llama3-70b-8192")
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor.invoke({"input": query},
                                 {"callbacks": [StreamlitCallbackHandler(st.container())]})

# Get user input
prompt = st.chat_input("Your question")
if prompt:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="./public/dickinson_picture.png"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                results = query_agent(prompt)
                full_response = results['output']
                message_placeholder.markdown(full_response)
                st.success(f"Completed in {round(time.time() - start_time, 2)} seconds.")
            except Exception as e:
                st.error(f"An error occurred while processing your request: {str(e)}")
                st.stop()
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})