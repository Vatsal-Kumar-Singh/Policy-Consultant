import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def create_vector_db():

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(curr_dir, "data", "Govt_Pol.pdf")
    db_dir = os.path.join(curr_dir, "db", "govt_pol")

    if not os.path.exists(db_dir):
        print(f"Creating database directory at {db_dir}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        docs = text_splitter.split_documents(documents)

        print("\n--- Creating embeddings ---")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        print("\n--- Finished creating embeddings ---")

        print("\n--- Creating and persisting vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=db_dir)
        print("\n--- Finished creating and persisting vector store ---")

    else:
        print(f"Database directory already exists at {db_dir}, skipping creation.")
