from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from graph.consts import RETRIEVE, GENERATE, WEBSEARCH, GRADE_DOCUMENTS
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

# --------------------- EDGE DECISION -------------------------
def decide_to_generation(state: GraphState):
    print('---ASSESS GRADE DOCUMENTS---')
    if state['web_search']:
        print('DECISION: NOT ALL DOCUMENTS RELEVANT')
        return WEBSEARCH
    else:
        print('DECISION: GENERATE')
        return GENERATE

# -------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generation,
    {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')