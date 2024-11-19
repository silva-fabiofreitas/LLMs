from pprint import pprint

from dotenv import load_dotenv


load_dotenv()

from ingestion import retriever
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.chains.generation import generation_chain



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
