from typing import List, Any, Dict

from ..chains.retrieval_grader import retrieval_grader
from ..state import  GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke({
            'question':question, 'document':doc.page_content
        })

        if score.binary_score:
            print("---GRADE: DOCUMNET RELEVANTE---")
            filtered_documents.append(doc)
        else:
            print("---GRADE: DOCUMNET NOT RELEVANTE---")
            web_search = True
            continue
    return {"documents": filtered_documents, "web_search": web_search, "question": question}