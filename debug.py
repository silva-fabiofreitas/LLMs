from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
    AgentType
)
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()


instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
You have qrcode package installed
If you get an error, debug your code and not try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    
template_react_agent ="""
    {instructions}

    TOOLS:
    ------

    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    Begin!

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}
"""
# base_prompt = hub.pull("langchain-ai/react-agent-template")    
base_prompt = PromptTemplate.from_template(template_react_agent)

prompt = base_prompt.partial(instructions=instructions, chat_history='')

tools = [PythonREPLTool()]
python_agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    tools=tools,
)

python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

csv_agent_executor: AgentExecutor = create_csv_agent(
    llm=ChatOpenAI(temperature=0, model='gpt-4o-mini'),
    path='data/iris.csv',
    verbose=True,
    # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)

generic_chain = ChatOpenAI(temperature=.5, model='gpt-4o-mini')


def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
    return python_agent_executor.invoke({"input": original_prompt})

from langchain.tools import StructuredTool
from pydantic import BaseModel

class CSVAgentInput(BaseModel):
    input: str  # Nome e tipo do argumento esperado
    

def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
    return python_agent_executor.invoke({"input": original_prompt})

tools = [
    StructuredTool(
        name="Python_Agent",
        func=python_agent_executor.invoke,
        description="""useful when you need to transform natural language to python and execute the python code,
                        returning the results of the code execution
                        DOES NOT ACCEPT CODE AS INPUT""",
        args_schema=CSVAgentInput,
    
    ),
    StructuredTool(
        name="CSV_Agent",
        func=csv_agent_executor.invoke,
        description="""Useful when you need to answer questions over the iris.csv file.
                        Takes as input the entire question and returns the answer after running pandas calculations.""",
        args_schema=CSVAgentInput,
    ),
     Tool(
        name="Generic_chain",
        func=generic_chain.invoke,
        description="""use it when you want an answer that doesn't fit into the other tools. It is a generic tool""",
    ),
]

from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [('system', 'You,re a  helpful assistant'),
    ('human', "{input}"),
    ('placeholder', "{agent_scratchpad}")]
)

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)


agent_executor.invoke(
    {
        # "input": "Qual a média de comprimento das flores?",
        "input": "Generate and save in current working directory 15 qrcodes that point to https://github.com/silva-fabiofreitas/",
        # "input": "Conte uma piada",
    }
)
