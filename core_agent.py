import streamlit as st
import os
import cohere
from langchain_cohere import ChatCohere
from langchain_cohere import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any

os.environ["COHERE_API_KEY"] = ""

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
llm = ChatCohere(model="command-r-plus")

# Fake weather tool for demonstration
def fake_weather(location: str) -> str:
    weathers = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    import random
    return f"The weather in {location} is {random.choice(weathers)}."

weather_tool = Tool(
    name="weather",
    func=fake_weather,
    description="Check the current weather in a given location. Input should be a city or location name."
)

tools = [weather_tool]

prompt = ChatPromptTemplate.from_template("{input}")

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    st.title("AgentD ReAct Demo by Ty Chermsirivatana")

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