from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

# Configuração de variáveis
PDF_PATH = '../../data/pdfs/Tese_Ficha_catalografica[v3].pdf'
CHROMA_DIR = './.chroma'
COLLECTION_NAME = 'rag_tese'

# Inicialize o cliente ChromaDB
client = Chroma(persist_directory=CHROMA_DIR)._client

# Verificar se a coleção já existe
existing_collections = client.list_collections()
collection_exists = COLLECTION_NAME in [col.name for col in existing_collections]

if not collection_exists:
    # Carregar o documento PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # Dividir o documento em chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Criar o banco vetorial e carregar os documentos
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Documentos carregados na coleção '{COLLECTION_NAME}'.")
else:
    print(f"A coleção '{COLLECTION_NAME}' já existe. Nenhuma ação adicional foi realizada.")


# Configurar o retriever
retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
