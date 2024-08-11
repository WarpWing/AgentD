import os
import json
import time
import requests
import re
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from bs4 import BeautifulSoup
import cohere

# Load environment variables
load_dotenv()

# Initialize external services and clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
firecrawl = FirecrawlApp(api_url=os.getenv("FIRECRAWL_API_ENDPOINT"), api_key="Banana Pudding")
pplx_key = os.getenv("PERPLEXITY_API_Key")

params = {
    'pageOptions': {
        'onlyMainContent': True
    }
}

# Tool functions
def return_faculty_profile(query) -> str:
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
        return result
    else:
        st.write("❌ No matching record found.")
        return "No matching record found."

def dickinson_search_function(query: str) -> str:
    st.write(f"🔍 Performing Dickinson search for: `{query}`")

    results = ""
    internet_results = DDGS().text(f"{query} site:dickinson.edu", max_results=3)
   
    for result in internet_results:
        url = result["href"]
        st.write(f"🌐 Found relevant link: `{url}`")
        st.write("🔄 Scraping content for this URL")
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

# Set up agent and executor
prompt_template = """
You are AgentD, a question and answering agent created by Ty Chermsirivatana. You were made to answer questions about Dickinson College using the various tools available to you. 
You need to be as verbose and detailed as possible. Make sure to mention a specific professor or staff member as a point of contact if the topic has it (like directors or chairs of certain centers of programs). 
When giving information about faculty or staff, make sure to mention their department, title, phone number, building and most importantly, their classes (separated in bullet points by Fall and Spring and Summer if) 
(which you should provide in bullet point form).
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

llm = ChatCohere(model="command-r-plus", temperature=0.2, streaming=True)
agent = create_cohere_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# Main function
def main():
    st.title("🤖 AgentD ReAct Demo by Ty Chermsirivatana")
    st.write("This AgentD ReAct demo is powered by the Cohere and Firecrawl API. It is designed to answer questions about Dickinson College. It currently has a faculty profile tool and a grounded web search tool.")
    
    user_input = st.text_input("🙋 Enter your question:")
    if st.button("🚀 Get Answer"):
        if user_input:
            st.write(f"📝 Processing query: `{user_input}`")
            
            start_time = time.time()
            result = agent_executor.invoke({"input": user_input})
            end_time = time.time()
            
            st.write(f"🏁 Final Answer:\n\n{result['output']}")
            st.write(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
        else:
            st.write("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()