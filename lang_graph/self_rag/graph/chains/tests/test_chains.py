from pprint import pprint

from dotenv import load_dotenv


load_dotenv()

from ingestion import retriever
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucinations_grader
from graph.chains.router import question_router


def test_retrival_grader_answer_yes():
    question = 'ciclo virtuoso'
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke({
        'question': question, 'document':doc_text
    })

    assert res.binary_score

def test_retrival_grader_answer_no():
    question = 'ciclo virtuoso'
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke({
        'question': 'foo', 'document':doc_text
    })

    assert not res.binary_score

def test_generation_chain():
    question = 'ciclo virtuoso'
    docs = retriever.invoke(question)
    res = generation_chain.invoke({'context':docs, 'question':question})
    pprint(res)

def test_hallucination_grader_yes():
    question = 'ciclo virtuoso'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({'context': docs, 'question': question})
    res = hallucinations_grader.invoke({'documents':docs, 'generation':generation})

    assert res.binary_score

def test_hallucination_grader_no():
    question = 'ciclo virtuoso'
    docs = retriever.invoke(question)
    res = hallucinations_grader.invoke({'documents':docs, 'generation':'A vaca foi para o brejo'})

    assert not res.binary_score

def test_to_vectorstore():
    question = 'ciclo virtuoso'
    res = question_router.invoke({'question':question})

    assert res.datasource == 'vectorstore'

def test_to_websearch():
    question = 'Vaca Maro'
    res = question_router.invoke({'question':question})

    assert res.datasource == 'websearch'

