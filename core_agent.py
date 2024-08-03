import os
import json
import time
import requests
import re
import faiss
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import cohere

# Load environment variables
load_dotenv()

# Initialize external services and clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
firecrawl = FirecrawlApp(api_url=os.getenv("FIRECRAWL_API_KEY"), api_key="Banana Pudding")

params = {
    'pageOptions': {
        'onlyMainContent': True
    }
}

# Semantic Cache class
class SemanticCache:
    def __init__(self, dimension=768, threshold=0.8):
        self.index = faiss.IndexFlatL2(dimension)
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.threshold = threshold
        self.cache = []

    def add(self, query, result):
        embedding = self.encoder.encode([query])[0]
        self.index.add(embedding.reshape(1, -1))
        self.cache.append((query, result))
        st.write(f"🔒 Added to semantic cache: `{query}`")

    def search(self, query):
        embedding = self.encoder.encode([query])[0]
        D, I = self.index.search(embedding.reshape(1, -1), 1)
        if D[0][0] < self.threshold and I[0][0] < len(self.cache):
            st.write(f"🔓 Found in semantic cache: `{query}`")
            return self.cache[I[0][0]][1]
        st.write(f"🔍 Not found in semantic cache: `{query}`")
        return None

    def get_all_entries(self):
        return self.cache

semantic_cache = SemanticCache()

# Document Cache class
class DocumentCache:
    def __init__(self):
        self.cache = {}

    def clean_content(self, content):
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1', content)
        soup = BeautifulSoup(content, 'html.parser')
        content = soup.get_text()
        content = re.sub(r'\s+', ' ', content).strip()
        return content

    def add(self, url, content):
        cleaned_content = self.clean_content(content)
        self.cache[url] = cleaned_content
        st.write(f"🔒 Added to document cache: `{url}`")

    def get(self, url):
        if url in self.cache:
            st.write(f"🔓 Found in document cache: `{url}`")
            return self.cache[url]
        st.write(f"🔍 Not found in document cache: `{url}`")
        return None

    def get_all_entries(self):
        return self.cache

document_cache = DocumentCache()

# Tool functions
def return_faculty_profile(query) -> str:
    st.write(f"🔎 Searching for faculty profile: `{query}`")
    cached_result = semantic_cache.search(query)
    if cached_result:
        return cached_result

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
        
        cached_content = document_cache.get(url)
        if cached_content:
            profile_content = cached_content
        else:
            try:
                profile = firecrawl.scrape_url(url, params=params)
                profile_content = profile["content"]
                document_cache.add(url, profile_content)
                st.write("✅ Successfully scraped and cached profile content")
            except Exception as e:
                st.write(f"❌ Error fetching faculty profile: {str(e)}")
                profile_content = f"Error fetching faculty profile: {str(e)}"
        
        result = json.dumps({
            "directory_info": record,
            "faculty_profile": profile_content
        }, indent=2)
        semantic_cache.add(query, result)
        return result
    else:
        st.write("❌ No matching record found.")
        result = "No matching record found."
        semantic_cache.add(query, result)
        return result

def dickinson_search_function(query: str) -> str:
    st.write(f"🔍 Performing Dickinson search for: `{query}`")
    cached_result = semantic_cache.search(query)
    if cached_result:
        return cached_result

    results = ""
    internet_results = DDGS().text(f"{query} site:dickinson.edu", max_results=2)
   
    for result in internet_results:
        url = result["href"]
        st.write(f"🌐 Found relevant link: `{url}`")
        cached_content = document_cache.get(url)
        if cached_content:
            results += f"{cached_content}\n"
        else:
            st.write("🔄 Scraping new content for this URL")
            firecrawl_content = firecrawl.scrape_url(url, params=params)
            document_cache.add(url, firecrawl_content["content"])
            results += f"{firecrawl_content['content']}\n"
    
    semantic_cache.add(query, results)
    return results

def query_semantic_cache(query: str) -> str:
    st.write(f"🗄️ Querying semantic cache for: `{query}`")
    result = semantic_cache.search(query)
    if result:
        return f"Found in semantic cache: {result}"
    return "No matching entries found in the semantic cache."

def query_document_cache(url: str) -> str:
    st.write(f"📄 Querying document cache for: `{url}`")
    result = document_cache.get(url)
    if result:
        return f"Found in document cache: {result[:500]}..."
    return "No matching entries found in the document cache."

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

semantic_cache_tool = Tool(
    name="query_semantic_cache",
    func=query_semantic_cache,
    description="Query the content of the semantic cache. Use this tool to search for information that might have been previously cached based on semantic similarity. Input should be a search query."
)

document_cache_tool = Tool(
    name="query_document_cache",
    func=query_document_cache,
    description="Query the content of the document cache. Use this tool to retrieve cached content for a specific URL. Input should be a URL."
)

tools = [dickinson_search_tool, return_faculty_prof, semantic_cache_tool, document_cache_tool]

# Set up agent and executor
prompt_template = """
You are AgentD, a question and answering agent created by Ty Chermsirivatana. You were made to answer questions about Dickinson College using the various tools available to you. 
You need to be as verbose and detailed as possible. Make sure to mention a specific professor or staff member as a point of contact if the topic has it (like directors or chairs of certain centers of programs). 
When giving information about faculty or staff, make sure to mention their department, title, phone number, building and most importantly, their classes (separated in bullet points by Fall and Spring and Summer if) 
(which you should provide in bullet point form).
Always check the semantic cache first using the query_semantic_cache tool before using other tools to fetch new information. If you need to retrieve content for a specific URL, use the query_document_cache tool.
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
    st.write("This AgentD ReAct demo is powered by the Cohere and Firecrawl API. It is designed to answer questions about Dickinson College. It currently has a vector search (with database cache), a semantic cache for common questions, a faculty profile tool, and a grounded web search tool.")
    
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

    if st.button("📑 Index Whole File"):
        index_whole_file()

def index_whole_file():
    st.write("📊 Indexing the whole file...")
    with open(__file__, 'r') as file:
        content = file.read()
    
    chunks = content.split('\n\n')
    
    for i, chunk in enumerate(chunks):
        semantic_cache.add(f"File chunk {i}", chunk)
    
    st.write(f"✅ Indexed {len(chunks)} chunks from the file.")

    cache_contents = {
        "semantic_cache": semantic_cache.get_all_entries(),
        "document_cache": document_cache.get_all_entries()
    }
    
    with open('index.json', 'w') as f:
        json.dump(cache_contents, f, indent=2)
    
    st.write("💾 Cache contents written to index.json")

if __name__ == "__main__":
    main()