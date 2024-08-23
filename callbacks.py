from typing import Dict, List, Any
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],  **kwargs: Any) -> Any:
        print(f'***Prompt to LLM \n{prompts[0]}')
        print('***')
        
        return 
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f'*** LLM response:*** \n{response.generations[0][0].text}')
        print('****')
        return 




from dotenv import load_dotenv
import bs4
import langchainhub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

collection_name='docs_langchain'

def run_llm(query:str):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    docsearch = Chroma(
            persist_directory='db/', 
            collection_name=collection_name, 
            embedding_function=embeddings
        )
    
    retriever = docsearch.as_retriever()
    
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True)
    
    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    
    rag_chain = {"context": retriever , "question": RunnablePassthrough()} | prompt | chat
    
    res = rag_chain.invoke(query) 
    
        
    return  res
    

if __name__ == "__main__":
    
    query = 'Whats is a langchain Chain?'
        
    run_llm(query)