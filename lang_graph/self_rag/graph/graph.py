from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from graph.chains.answer_grader import answer_grader
from graph.chains.router import question_router
from graph.chains.hallucination_grader import hallucinations_grader
from graph.consts import RETRIEVE, GENERATE, WEBSEARCH, GRADE_DOCUMENTS
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

# --------------------- EDGE DECISION -------------------------
def decide_to_generation(state: GraphState):
    print('---ASSESS GRADE DOCUMENTS---')
    if state['web_search']:
        print('DECISION: NOT ALL DOCUMENTS RELEVANT, INCLUDE WEBSEARCH')
        return WEBSEARCH
    else:
        print('DECISION: GENERATE')
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("---Check Hallucination---")

    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucinations_grader.invoke({
        'documents': documents, 'generation': generation
    })

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({
            'question': question, 'generation': generation
        })
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return 'useful'
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        return "not supported"

def route_question(state: GraphState):
    print('---ROUTE QUESTION---')
    question = state['question']
    source = question_router.invoke({'question': question})
    if source.datasource == WEBSEARCH:
        print('---ROUTE QUESTION TO WEBSEARCH---')
        return WEBSEARCH
    elif source.datasource == 'vectorstore':
        print('---ROUTE QUESTION TO RAG')
        return RETRIEVE



# -------------------------------------------------------------

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE
    }
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generation,
    {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        'not supported': GENERATE,
        'useful':END,
        "not useful": WEBSEARCH
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')