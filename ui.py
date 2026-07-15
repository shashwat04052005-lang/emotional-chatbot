import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from embeddings import ChromaMiniLMEmbeddings


load_dotenv()

st.set_page_config(page_title="Relationship AI", page_icon="💬")
st.title("Relationship AI")
st.write("An expert guide for relationship dynamics, book recommendations, and emotional growth.")


@st.cache_resource
def init_rag():
    embeddings = ChromaMiniLMEmbeddings()
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

    system_prompt = (
        "You are Relationship AI, an empathetic expert in relationships and emotional intelligence.\n"
        "You can use two sources of knowledge:\n"
        "1. Retrieved excerpts from 'Emotional Intelligence', 'Attached', and 'Wired for Love'.\n"
        "2. Your general knowledge of self-improvement, relationship psychology, communication, "
        "emotional growth, and relevant books.\n\n"
        "ROUTING INSTRUCTIONS:\n"
        "- For questions about attachment styles, conflict dynamics, emotional regulation, or ideas "
        "covered by the retrieved excerpts, ground the answer heavily in that context.\n"
        "- For broad questions such as what to do next, how to heal, general relationship guidance, "
        "or book recommendations, use your general knowledge to answer broadly, empathetically, and "
        "constructively. You may recommend other established relationship books when useful.\n"
        "- Answer directly and naturally. Do not announce which route you selected.\n"
        "- Do not mention context, books, or sources unless the user explicitly asks for a source, "
        "citation, reference, or book behind the answer. When asked, clearly distinguish retrieved book "
        "sources from general recommendations. For retrieved sources, begin: \"Based on the context from "
        "'Emotional Intelligence', 'Attached', and 'Wired for Love'\" and identify what was relevant.\n"
        "- If the request is unrelated to relationships, emotions, communication, personal growth, or "
        "relevant books, reply: \"I am here to guide your emotional and relationship journey. Let's stick "
        "to those themes.\"\n\n"
        "Retrieved context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


try:
    rag_chain = init_rag()
except Exception as error:
    st.error(f"Relationship AI could not start: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input(
    "Ask about relationship dynamics, advice, or book recommendations..."
):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Reflecting..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
            except Exception as error:
                answer = f"I couldn't generate a response: {error}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
