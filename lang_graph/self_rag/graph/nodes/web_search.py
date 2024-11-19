from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain.schema import Document


from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("WEB SEARCH")
    question = state['question']
    documents = state['documents']

    tavily_results = web_search_tool.invoke({'query':question})
    web_results = "\n".join([d["content"] for d in tavily_results])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}





