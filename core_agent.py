import streamlit as st
import os
import cohere
import json
from langchain_cohere import ChatCohere
from langchain_cohere import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, Any
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options

os.environ["COHERE_API_KEY"] = ""
os.environ['TAVILY_API_KEY'] = ""

co = cohere.Client(os.environ["COHERE_API_KEY"])

class StreamlitMarkdownLogger(BaseCallbackHandler):
    def __init__(self):
        self.logs = []
        self.chain_started = False
        self.step_count = 0
        self.output_container = st.empty()

    def log(self, message: str, level: str = "info") -> None:
        prefix = {
            "info": "ℹ️ ",
            "step": "👉 ",
            "tool": "🛠️ ",
            "output": "📤 ",
            "thought": "💭 ",
            "final": "🎯 "
        }.get(level, "")
        self.logs.append(f"{prefix}{message}")
        self.update_output()

    def update_output(self) -> None:
        markdown_output = "**Agent Process:**\n\n" + "\n\n".join(self.logs)
        self.output_container.markdown(markdown_output)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        if not self.chain_started:
            self.log("Chain started", "info")
            self.chain_started = True

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self.log(f"Using tool: {serialized['name']}", "tool")
        self.log(f"Tool input: {input_str}", "tool")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.log(f"Tool output: {output}", "output")

    def on_agent_action(self, action, **kwargs: Any) -> Any:
        self.step_count += 1
        self.log(f"Step {self.step_count}", "step")
        self.log(f"Thought: {action.log}", "thought")
        self.log(f"Action: {action.tool}", "info")
        self.log(f"Action Input: {action.tool_input}", "info")

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        self.log(f"Final Answer: {finish.return_values['output']}", "final")
        self.log("Chain ended", "info")

# Set up the ChatCohere model
llm = ChatCohere(model="command-r-plus",connectors=[{"id": "web-search"}])

# Fetch people data
f = requests.get('https://www2.dickinson.edu/lis/angularJS_services/directory_open/data.cfc?method=f_getAllPeople').json()

def find_record(query: str) -> str:
    parts = query.split()
    if len(parts) == 2:
        first_name, last_name = parts
        for record in f:
            if record['FIRSTNAME'].lower() == first_name.lower() and record['LASTNAME'].lower() == last_name.lower():
                return json.dumps(record, indent=2)
    elif '@' in query:
        email = query
        email_domain = email.split('@')[-1] if '@' in email else ''
        for record in f:
            if record['EMAIL'] == email.split('@')[0] and email_domain == 'dickinson.edu':
                return json.dumps(record, indent=2)
    return "No matching record found."

# Configure Chrome options for headless mode
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

def return_faculty_classes(email: str) -> str:
    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.dickinson.edu/site/custom_scripts/dc_faculty_profile_index.php?fac={email}")
    
    # Extract Education Details
    education_details = []
    try:
        education_section = driver.find_element(By.XPATH, "//h3[contains(text(), 'Education')]/following-sibling::ul")
        for item in education_section.find_elements(By.TAG_NAME, "li"):
            education_details.append(item.text)
    except NoSuchElementException:
        print("No Education section found.")

    # Click on the 'Course Info' tab
    try:
        course_info_tab = driver.find_element(By.XPATH, "//a[@href='#courses']")
        course_info_tab.click()
    except NoSuchElementException:
        print("Course Info tab not found.")

    # Function to extract course names
    def extract_courses(xpath):
        course_elements = driver.find_elements(By.XPATH, xpath)
        unique_courses = set()
        for course in course_elements:
            course_name = course.text.split('\n')[0]
            unique_courses.add(course_name)
        return sorted(list(unique_courses), key=lambda x: int(x.split(' ')[1]))

    # Extract Course Names for Fall and Spring
    fall_courses = extract_courses('//h3[contains(text(), "Fall")]/following-sibling::p')
    spring_courses = extract_courses('//h3[contains(text(), "Spring")]/following-sibling::p')

    driver.quit()
    
    result = {
        "education": education_details,
        "fall_courses": fall_courses,
        "spring_courses": spring_courses
    }
    return json.dumps(result, indent=2)

# Define tools
find_record_tool = Tool(
    name="find_record",
    func=find_record,
    description="Find a person's record in the Dickinson College directory. Input should be either a full name (first and last) or an email address."
)

return_faculty_classes_tool = Tool(
    name="return_classes",
    func=return_faculty_classes,
    description="Get education details and courses taught by a faculty member. Input should be the faculty member's email address."
)

internet_search = Tool(
  name="internet_search",
  func=TavilySearchResults(include_domain=["dickinson.edu"], max_result=3,search_depth="advanced"),
  description="Returns a list of relevant document snippets for a textual query retrieved from the internet. Input should be a query to search the internet with."
)



tools = [find_record_tool, return_faculty_classes_tool,internet_search]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are AgentD, a question and answering agent created by Ty Chermsirivatana. You were made to answer questions about Dickinson College using the various tools available to you. You need to be as verbose and detailed as possible. Make sure to mention a specific professor or staff member as a point of contact if the topic has it (like directors or chairs of certain centers of programs). When giving information about faculty or staff, make sure to mention their department, title, phone number, building and most importantly, their classes (separated in bullet points by Fall and Spring and Summer if applicable) (which you should provide in bullet point form.)",
        ),
        ("user", "{input}"),
    ]
)

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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