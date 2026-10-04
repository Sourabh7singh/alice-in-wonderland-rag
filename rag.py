from langchain_community.document_loaders import TextLoader
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


class Rag:
    def __init__(
        self,
        file_path:str="data/alice_in_wonderland.md"
    ):
        
        self.chroma_path = file_path.split("/")[-1].replace(".md","_db")
        self.file_path = file_path
        
        self.client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
        
        self.load_database()   

    def load_data(
        self,
    ):
        loader = TextLoader(self.file_path,encoding="utf-8")
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s). Total character count: {len(documents[0].page_content)}")
        
        self.loader = loader
        self.documents = documents
        
    def break_data_to_markdown(
        self,
    ):
        headers_to_split_on = [
            ("CHAPTER", "Chapter")
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False
        )
        header_docs = markdown_splitter.split_text(self.documents[0].page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(header_docs)
        
        self.text_splitter = text_splitter
        self.chunks = chunks
        
    def save_database(self,):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        db = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=self.chroma_path
        )

        print("Vector database populated successfully!")
        self.embeddings = embeddings
        self.db = db
    
    def load_database(self):
        if not os.path.exists(self.chroma_path):
            # create db first
            self.load_data()
            self.break_data_to_markdown()
            self.save_database()
            
            return 
        
            
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=embeddings
        )

        print(f"Vector database loaded successfully! Total chunks: {db._collection.count()}")

        self.embeddings = embeddings
        self.db = db
    
    def get_gemini_response(self,prompt_text: str, model_name: str = "gemini-2.5-flash") -> str:
        """
        Sends the formatted RAG prompt to Gemini and returns the generated response.
        """
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt_text,
        )
        
        return response.text
        
    def ask_query(self,query):
        # BGE instruction prefix to maximize search precision
        bge_query = f"Represent this sentence for searching relevant passages: {query}"

        # Configure retriever using MMR
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "lambda_mult": 0.7}
        )

        retrieved_docs = retriever.invoke(bge_query)
        context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

        PROMPT = f"""Answer the question strictly based on the provided context below. If the premise of the question is incorrect according to the text, correct it.

            Context:
            {context_text}

            Question:
            {query}

            Answer:"""
        
        response_text = self.get_gemini_response(PROMPT)

        return response_text
        
        
        
import time

def main():
    start_time = time.perf_counter()
    
    rag = Rag()
    
    question = "What did the Drink Me bottle say on its label that made Alice instantly fall asleep?"

    response = rag.ask_query(question)
    print("\n\n-----\n\n",response)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"\nTotal time taken: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()