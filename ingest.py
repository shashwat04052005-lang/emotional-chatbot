from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from embeddings import ChromaMiniLMEmbeddings

load_dotenv()

print("Loading PDFs from data folder...")
loader = PyPDFDirectoryLoader("data/")
books = loader.load()
print(f"Loaded {len(books)} pages.")

if not books:
    raise RuntimeError("No PDF pages were found in the data folder.")

print("Splitting text into manageable chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(books)
print(f"Created {len(chunks)} chunks.")

print("Generating local ONNX embeddings (all-MiniLM-L6-v2)...")
embeddings = ChromaMiniLMEmbeddings()

# Rebuild the collection instead of appending duplicate chunks on every run.
existing_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
existing_store.delete_collection()

vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("Vector database successfully built.")
