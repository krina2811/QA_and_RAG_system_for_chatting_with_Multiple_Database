import os
from typing import List, Tuple
from agent_graph.load_tools_config import LoadToolsConfig
from sqlalchemy import create_engine, inspect
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyprojroot import here
import pdfplumber
import hashlib
import chromadb
client = chromadb.Client()

APPCFG = LoadToolsConfig()

def load_or_create_collection(collection_name="pdf_embeddings"):
    try:
        # Try to load existing collection
        collection = client.get_or_create_collection(collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading collection: {e}")
        collection = client.create_collection(collection_name)
        print(f"New collection '{collection_name}' created.")
    return collection


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to hash the PDF content
def hash_pdf_content(pdf_path):
    # Extract the content of the PDF
    content = extract_text_from_pdf(pdf_path)
    
    # Compute the SHA256 hash of the content
    pdf_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return pdf_hash

# Function to check if the PDF hash exists in Chroma DB
def check_if_pdf_exists_in_db(pdf_path, collection):
    # Hash the content of the PDF
    pdf_hash = hash_pdf_content(pdf_path)
    
    # Query the Chroma collection for the PDF hash
    query_results = collection.query(
        query_texts=[pdf_hash],  # Use the hash as the query text
        n_results=5  # Search for the top 5 results (adjust as needed)
    )
    
    # Check if the hash already exists in the database
    for result in query_results['documents']:
        if pdf_hash in result:
            print(f"Found matching document (hash match) in the database.")
            return True  # The PDF is found in the DB
    
    return False  # No match found



class ProcessFiles:
    """
    A class to process uploaded files, converting them to a SQL database format.

    This class handles both CSV and XLSX files, reading them into pandas DataFrames and
    storing each as a separate table in the SQL database specified by the application configuration.
    """
    def __init__(self, files_dir: List, chatbot: List) -> None:
        """
        Initialize the ProcessFiles instance.

        Args:
            files_dir (List): A list containing the file paths of uploaded files.
            chatbot (List): A list representing the chatbot's conversation history.
        """
        self.files_dir = files_dir
        self.chatbot = chatbot
        db_path = APPCFG.upload_csv_directory
        db_path = f"sqlite:///{db_path}"
        self.engine = create_engine(db_path)
        print("Number of uploaded files:", len(self.files_dir))

    def _process_uploaded_files(self) -> Tuple:
        """
        Private method to process the uploaded files and store them into the SQL database.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        for file_dir in self.files_dir:
            print("^^^^^^^^^^^^^^^^^", file_dir)
            file_names_with_extensions = os.path.basename(file_dir)
            file_name, file_extension = os.path.splitext(file_names_with_extensions)
            if file_extension == ".csv":
                df = pd.read_csv(file_dir)
                df.to_sql(file_name, self.engine, index=False)
                print("==============================")
                print("All csv files are saved into the sql database.")
                self.chatbot.append((" ", "Uploaded files are ready. Please ask your question"))
            elif file_extension == ".pdf":
                collection_name = APPCFG.for_uploaded_pdf_rag_collection_name
                docs = [PyPDFLoader(file_dir).load_and_split()]
                docs_list = [item for sublist in docs for item in sublist]
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                doc_splits = text_splitter.split_documents(docs_list)
                chunk_size = APPCFG.for_uploaded_pdf_rag_chunk_size
                chunk_overlap = APPCFG.for_uploaded_pdf_rag_chunk_overlap
                collection_name = APPCFG.for_uploaded_pdf_rag_collection_name
                embedding_model = APPCFG.for_uploaded_pdf_rag_embedding_model
                vectordb_dir = APPCFG.for_uploaded_pdf_rag_vectordb_directory

                # Add to vectorDB
                vectordb = Chroma.from_documents(
                    documents=doc_splits,
                    collection_name=collection_name,
                    embedding=OpenAIEmbeddings(model=embedding_model),
                    persist_directory=str(here(vectordb_dir))
                )
                self.chatbot.append((" ", "Uploaded files are ready. Please ask your question"))
                print("VectorDB is created and saved.")
                print("Number of vectors in vectordb:", vectordb._collection.count(), "\n\n")
            else:
                raise ValueError("The selected file type is not supported")
            
        # self.chatbot.append((" ", "Uploaded files are ready. Please ask your question"))
        return "", self.chatbot

    def _validate_db(self):
        """
        private method to validate that the SQL database has been updated correctly with the right tables.
        """
        insp = inspect(self.engine)
        table_names = insp.get_table_names()
        print("==============================")
        print("Available table nasmes in created SQL DB:", table_names)
        print("==============================")

    def run(self):
        """
        public method to execute the file processing pipeline.

        Includes steps for processing uploaded files and validating the database.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        input_txt, chatbot = self._process_uploaded_files()
        self._validate_db()
        return input_txt, chatbot


class UploadFile:
    """
    A class that acts as a controller to run various file processing pipelines
    based on the chatbot's current functionality when handling uploaded files.
    """
    @staticmethod
    def run_pipeline(files_dir: List, chatbot: List):
        """
        Run the appropriate pipeline based on chatbot functionality.

        Args:
            files_dir (List): List of paths to uploaded files.
            chatbot (List): The current state of the chatbot's dialogue.
            chatbot_functionality (str): A string specifying the chatbot's current functionality.

        Returns:
            Tuple: A tuple of an empty string and the updated chatbot list, or None if functionality not matched.
        """
        # if chatbot_functionality == "Process files":
        pipeline_instance = ProcessFiles(files_dir=files_dir, chatbot=chatbot)
        input_txt, chatbot = pipeline_instance.run()
        return input_txt, chatbot
        # else:
        #     pass # Other functionalities can be implemented here.
