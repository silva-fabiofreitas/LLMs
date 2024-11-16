from dotenv import load_dotenv

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph

load_dotenv()

from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

AGENT_REASON = 'agent_reason'
ACT = "act"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT

flow = StateGraph(AgentState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, execute_tools)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path='graph.png')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res = app.invoke({'input':"Qual foi o ultimo resultado do jogo do flamengo na copa do brasil em 2024? Jogou contra quem, multiplique o resultado por três"})
    # res = app.invoke({'input':"What is the weather in sf? Write it and then Trible it"})
    print(res['agent_outcome'].return_values['output'])

