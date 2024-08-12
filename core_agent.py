import os
import json
import time
import requests
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import cohere

# Load environment variables
load_dotenv()

# Initialize external services and clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
firecrawl = FirecrawlApp(api_url=os.getenv("FIRECRAWL_API_ENDPOINT"), api_key="Banana Pudding")

params = {
    'pageOptions': {
        'onlyMainContent': True
    }
}

agentd_prompt = """
You are an AI assistant for Dickinson College, designed to provide comprehensive, accurate, and user-friendly responses to queries. Your responses should be detailed and offer a clear overview, including relevant contact information and course details where applicable. When providing information about faculty or staff, ensure you include their full name, department, title, and a list of courses taught, separated by semester (Fall, Spring, and Summer). Additionally, mention specific professors or staff as points of contact if they are directors or hold similar positions. 

Your responses should be concise, well-organized, and within a word limit of 600 words, focusing on providing a complete and accurate response. 
"""

# Tool functions
def return_faculty_profile(query: str) -> str:
    st.write(f"🔎 Searching for faculty profile: `{query}`")

    # Fetch directory data
    st.write("📚 Fetching directory data...")
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
        st.write(f"✅ Found record for: {record['FIRSTNAME']} {record['LASTNAME']}")
        # If a record is found, try to fetch the faculty profile
        email = record['EMAIL']
        url = f"https://www.dickinson.edu/site/custom_scripts/dc_faculty_profile_index.php?fac={email}"
        st.write(f"🌐 Fetching profile from URL: `{url}`")
        
        try:
            profile = firecrawl.scrape_url(url, params=params)
            profile_content = profile["content"]
            st.write("✅ Successfully scraped profile content")
        except Exception as e:
            st.write(f"❌ Error fetching faculty profile: {str(e)}")
            profile_content = f"Error fetching faculty profile: {str(e)}"
        
        result = json.dumps({
            "directory_info": record,
            "faculty_profile": profile_content
        }, indent=2)
        return [{"results": result}]
    else:
        st.write("❌ No matching record found.")
        return "No matching record found."

def perplexity_search(query: str) -> str:
    st.write(f"🔍 Performing Perplexity search for: `{query}`")

    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {
                "role": "system",
                "content": agentd_prompt
            },
            {
                "role": "user",
                "content": query + "Dickinson College"
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {os.getenv('PERPLEXITY_KEY')}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        st.write("✅ Successfully retrieved information from the Perplexity API")
        return [{"results": result['choices'][0]['message']['content']}]
    except Exception as e:
        st.write(f"❌ Error fetching information from Perplexity API: {str(e)}")
        return f"Error fetching information: {str(e)}"

# Set up tools
return_faculty_prof = {
    "name": "return_faculty_profile",
    "description": "Return details about a Faculty Member (Professors). Use this tool when needing specific information about a professor's education, courses they teach or their biography/personal information. Input should be the faculty member's email address or full name.",
    "parameter_definitions": {
        "query": {
            "description": "The faculty member's email address or full name",
            "type": "str",
            "required": True
        }
    }
}

perplexity_search_tool = {
    "name": "perplexity_search",
    "description": "Get information related to Dickinson College using the Perplexity API. Use this tool first for general queries about Dickinson College. Input should be a query to search up.",
    "parameter_definitions": {
        "query": {
            "description": "The query to search for information about Dickinson College",
            "type": "str",
            "required": True
        }
    }
}

tools = [perplexity_search_tool, return_faculty_prof]


st.title("🤖 AgentD Demo by Ty Chermsirivatana")
st.write("This AgentD demo is powered by the Cohere and Perplexity APIs. It is designed to answer questions about Dickinson College. It currently has a Perplexity search tool and a faculty profile tool.")

user_input = st.text_input("🙋 Enter your question:")
if st.button("🚀 Get Answer"):
    if user_input:
        st.write(f"📝 Processing query: `{user_input}`")
        
        start_time = time.time()
        
        message = agentd_prompt + user_input
        model = "command-r-plus"

        res = co.chat(model=model, message=message, force_single_step=False, tools=tools)

        while res.tool_calls:
            st.write(f"💭 Thought: {res.text}")  
            tool_results = []
            for call in res.tool_calls:
                if call.name == "return_faculty_profile":
                    tool_output = return_faculty_profile(call.parameters["query"])
                elif call.name == "perplexity_search":
                    tool_output = perplexity_search(call.parameters["query"])
                else:
                    tool_output = "Unknown tool called"
                
                search_results = {"call": call, "outputs": tool_output}
                tool_results.append(search_results)
            
            res = co.chat(
                model="command-r-plus",
                chat_history=res.chat_history,
                message="",
                force_single_step=False,
                tools=tools,
                tool_results=tool_results
            )
        
        end_time = time.time()
        
        st.write(f"🏁 Final Answer:\n\n{res.text}")
        st.write(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
    else:
        st.write("⚠️ Please enter a question.")