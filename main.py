# import sqlite_fix
import os
import logging
import streamlit as st
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from redis_client import check_redis_cache, cache_query_answer
from build_db import create_vector_db

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx._client").setLevel(logging.WARNING)

st.title("Haq ki Chabhi: Neetiyon Ka Dvar Khole")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", key="openai_api_key", type="password")
model = st.sidebar.selectbox("Select Model:", ("gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"), index=0)
user_query = st.text_input("Enter your query:", placeholder="Enter your query here...")
initialize_db = st.sidebar.button("Create Vector Database")

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    if(st.session_state.embeddings is None):
        st.warning("Kindly Create Vector Database")

    if(initialize_db and st.session_state.embeddings is None):
        st.progress(15, text="Creating vector store...")
        create_vector_db()
        st.progress(100, text="Successfully created vector store")
        st.session_state.embeddings = True

    if(user_query and st.session_state.embeddings is not None):
    
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(curr_dir, "db", "bns_db")
    
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
        retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
        )
    
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    
        llm = ChatOpenAI(model=model)
    
        contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as it is."
        )
    
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )
    
        history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
        )
    
        qa_system_prompt = (
        "You are an assistant for question-answering tasks related to Indian Government Policies. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use five to six sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
        )
    
        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )
    
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
        st.info("This application uses a RAG model on the Government Policy documents, with Redis caching and Flash Reranking for improved performance.")
    
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
        if user_query:
            with st.spinner("Generating response..."):
                in_cache = check_redis_cache(user_query)
                print("Checking redis cache")
                if in_cache:
                    print("Found answer in cache")
                    answer = in_cache
                    st.write(f"{answer}")
                
                else:
                    print("Couldn't find answer in cache")
                    result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                    answer = result["answer"]
                    st.write(f"{answer}")
                    cache_query_answer(user_query, answer)
    
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(SystemMessage(content=answer))

else:

    st.sidebar.warning("Please enter your OpenAI API key in the sidebar.")








