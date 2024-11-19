from typing import Dict, Any

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print('---GENERATING---')
    question = state['question']
    documents = state['documents']

    generation = generation_chain.invoke({
        'context':documents, "question":question
    })

    return {'question':question, 'documents':documents, 'generation':generation}


