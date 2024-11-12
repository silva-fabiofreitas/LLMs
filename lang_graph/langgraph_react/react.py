from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent

load_dotenv()

react_prompt = hub.pull("hwchase17/react")

@tool
def triple(num: float):
    """

    :param num: a number to triple
    :return: the number tripled -> multiplied by 3
    """
    return float(num) * 3

tools = [TavilySearchResults(max_result=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini")

react_agent_runnable = create_react_agent(llm, tools, react_prompt)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    react_prompt = hub.pull("hwchase17/react")