import streamlit as st
import os
import cohere
import json
import time
import requests
from langchain_cohere import ChatCohere
from langchain_cohere import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any
from langchain.schema import AgentAction, AgentFinish
from duckduckgo_search import DDGS
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
firecrawl = FirecrawlApp(api_url=os.getenv("FIRECRAWL_API_KEY"), api_key="Banana Pudding")

# Static Params for Firecrawl
params = {
    'pageOptions': {
        'onlyMainContent': True
    }
}

class StreamlitMarkdownLogger(BaseCallbackHandler):
    def __init__(self):
        self.intermediate_container = st.empty()
        self.final_answer_container = st.empty()
        self.intermediate_steps = []
        self.final_answer = ""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.intermediate_steps.append(f"Using tool: {serialized['name']}\nInput: {input_str}")
        self._update_intermediate_steps()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.intermediate_steps.append(f"Tool output: {output}")
        self._update_intermediate_steps()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.intermediate_steps.append(f"Thought: {action.log}\nAction: {action.tool}\nAction Input: {action.tool_input}")
        self._update_intermediate_steps()

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.final_answer = finish.return_values['output']
        self.stream_final_answer()

    def _update_intermediate_steps(self):
        self.intermediate_container.markdown("**Intermediate Steps:**\n\n" + "\n\n".join(self.intermediate_steps))

    def stream_final_answer(self):
        streamed_answer = "Final Answer: \n\n"
        for char in self.final_answer:
            streamed_answer += char
            self.final_answer_container.markdown(streamed_answer)
            time.sleep(0.0001)  # Adjust this value to control streaming speed

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # We don't need to do anything here as we're only streaming the final answer
        pass

def return_faculty_profile(query) -> str:
    # Fetch directory data
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
        # If a record is found, try to fetch the faculty profile
        email = record['EMAIL']
        try:
            profile = firecrawl.scrape_url(f"https://www.dickinson.edu/site/custom_scripts/dc_faculty_profile_index.php?fac={email}", params=params)
            return json.dumps({
                "directory_info": record,
                "faculty_profile": profile["content"]
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "directory_info": record,
                "faculty_profile": f"Error fetching faculty profile: {str(e)}"
            }, indent=2)
    else:
        return "No matching record found."

def dickinson_search_function(query: str) -> str:
    results = ""
    internet_results = DDGS().text(f"{query} site:dickinson.edu", max_results=2)
   
    for result in internet_results:
        firecrawl_content = firecrawl.scrape_url(result["href"], params=params)
        results += f"{firecrawl_content['content']}\n"
    
    return results

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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are AgentD, a question and answering agent created by Ty Chermsirivatana. You were made to answer questions about Dickinson College using the various tools available to you. You need to be as verbose and detailed as possible. Make sure to mention a specific professor or staff member as a point of contact if the topic has it (like directors or chairs of certain centers of programs). When giving information about faculty or staff, make sure to mention their department, title, phone number, building and most importantly, their classes (separated in bullet points by Fall and Spring and Summer if) (which you should provide in bullet point form.)",
        ),
        ("user", "{input}"),
    ]
)

# Set up the ChatCohere model
llm = ChatCohere(model="command-r-plus", temperature=0.2, streaming=True)

# Create the ReAct agent
agent = create_cohere_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

def main():
    st.title("AgentD ReAct Demo by Ty Chermsirivatana")
    st.write("This demo now includes tools to search the Dickinson College directory and retrieve faculty course information.")
    
    user_input = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if user_input:
            logger = StreamlitMarkdownLogger()
            agent_executor.invoke(
                {"input": user_input},
                config={"callbacks": [logger]}
            )
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()