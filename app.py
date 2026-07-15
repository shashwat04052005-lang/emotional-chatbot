import subprocess
import sys
from pathlib import Path


# Make `python app.py` use this project's environment even when the venv was
# not activated in the terminal.
venv_python = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
if sys.prefix == sys.base_prefix and venv_python.exists():
    raise SystemExit(subprocess.call([str(venv_python), *sys.argv]))

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from embeddings import ChromaMiniLMEmbeddings


load_dotenv()

# The database must be queried with the same embedding model used by ingest.py.
embeddings = ChromaMiniLMEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

system_prompt = (
    "You are Relationship AI, an empathetic, expert relationship coach.\n"
    "Your goal is to provide insightful guidance based strictly on the provided book contexts.\n"
    "Use the following pieces of retrieved context from 'Emotional Intelligence', 'Attached', "
    "and 'Wired for Love' to answer the user's concern. Answer the question directly and naturally. "
    "Do not begin with phrases such as 'Based on the context' and do not mention the books or sources "
    "unless the user explicitly asks for the source, citation, reference, or which book the answer came from. "
    "Only when the user asks for sources, say: \"Based on the context from 'Emotional Intelligence', "
    "'Attached', and 'Wired for Love'\" and then briefly identify the relevant source.\n\n"
    "CRITICAL RULE: If the answer cannot be found in the context, or if the user asks something completely "
    "unrelated to relationships/emotional health, respond politely: 'I can only guide you based on our core "
    "relationship texts. Let's try rephrasing your concern.' Do not make up outside information.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def main() -> None:
    print("\nRelationship AI is ready! (Type 'exit' to quit)\n")
    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_query.lower() == "exit":
            break
        if not user_query:
            continue

        print("\nThinking...")
        response = rag_chain.invoke({"input": user_query})
        print(f"\nCoach: {response['answer']}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
