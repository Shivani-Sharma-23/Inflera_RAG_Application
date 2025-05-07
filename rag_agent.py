import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re
import argparse
import google.generativeai as genai
from typing import List, Dict, Any, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Loads and preprocess documents from a directory.
class DocumentLoader:
    def __init__(self, directory="documents"):
        self.directory = directory
        
    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory."""
        documents = []
        
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            logger.warning(f"Created empty documents directory: {self.directory}")
            return documents
            
        for filename in os.listdir(self.directory):
            if filename.endswith(('.txt', '.md', '.pdf', '.docx')):
                filepath = os.path.join(self.directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                        doc = Document(page_content=content, metadata={"source": filename})
                        documents.append(doc)
                        logger.info(f"Loaded document: {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
        
        return documents

#Splits documents into smaller chunks for better processing.
class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents to split.")
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

#Creates a vector store using FAISS for efficient similarity search.
class VectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        
    def add_documents(self, documents: List[Document]):#Add documents to the vector store.
        if not documents:
            logger.warning("No documents to add to vector store.")
            return
            
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        logger.info(f"Added {len(documents)} document chunks to vector store")
        
    def search(self, query: str, k=3) -> List[Document]:
        if not self.index:
            logger.error("Vector store is empty. Add documents first.")
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k=k
        )
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        
        logger.info(f"Retrieved {len(results)} documents for query: {query}")
        return results

class LLMClient:
    def __init__(self, model="gemini-2.0-flash"):
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY", "you_api_key")
        genai.configure(api_key=api_key)
        
    def generate_response(self, query: str, context: str = None) -> str:
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 500,
            }
            
            model = genai.GenerativeModel(model_name=self.model,
                                         generation_config=generation_config)
            
            if context:
                prompt = f"""You are a helpful assistant answering questions based on the provided context.
                Use the context below to answer the user's question. If the answer is not in
                the context, say that you don't have enough information.
                
                Context:
                {context}
                
                Question: {query}"""
            else:
                prompt = f"You are a helpful assistant. {query}"
            
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"

class Tools:
    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            cleaned_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            result = eval(cleaned_expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    @staticmethod
    def define(term: str) -> str:
        try:
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}") #Dictionary API
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    meanings = data[0].get('meanings', [])
                    if meanings:
                        definitions = [
                            f"{i+1}. ({meaning.get('partOfSpeech', 'n/a')}) {definition.get('definition', 'No definition available')}"
                            for i, meaning in enumerate(meanings[:3])
                            for definition in meaning.get('definitions', [])[:1]
                        ]
                        return f"Definition of '{term}':\n" + "\n".join(definitions)
            return f"Could not find definition for '{term}'"
        except Exception as e:
            return f"Error fetching definition: {str(e)}"

class Agent:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient, tools: Tools):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.tools = tools
        self.logs = []
        
    def log_step(self, step_name: str, details: Any):
        log_entry = {
            "step": step_name,
            "details": details
        }
        self.logs.append(log_entry)
        logger.info(f"Agent step: {step_name} - {details}")
        
    def determine_action(self, query: str) -> str:
        query_lower = query.lower()
        
        if re.search(r'\bcalculate\b|\bcompute\b|\bevaluate\b|\bsolve\b', query_lower):
            match = re.search(r'calculate\s+([\d+\-*/() .]+)', query_lower)
            if match:
                expression = match.group(1).strip()
                self.log_step("action_selection", f"Selected calculator tool for expression: {expression}")
                return "calculate", expression
        
        if re.search(r'\bdefine\b|\bmeaning\b|\bdefinition\b|\bwhat is\b|\bwhat are\b', query_lower):
            patterns = [
                r'define\s+(?:the\s+)?(?:word\s+)?["\']?([a-zA-Z]+)["\']?',
                r'meaning\s+of\s+["\']?([a-zA-Z]+)["\']?',
                r'definition\s+of\s+["\']?([a-zA-Z]+)["\']?',
                r'what\s+(?:is|are)\s+(?:a|an)?\s*["\']?([a-zA-Z]+)["\']?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    term = match.group(1).strip()
                    self.log_step("action_selection", f"Selected dictionary tool for term: {term}")
                    return "define", term
        
        self.log_step("action_selection", "Selected RAG pipeline")
        return "rag", query
    
    def process_query(self, query: str) -> Dict[str, Any]:
        self.logs = []
        
        action, action_input = self.determine_action(query)
        
        if action == "calculate":
            result = self.tools.calculate(action_input)
            return {
                "action": "calculate",
                "input": action_input,
                "result": result,
                "context": None,
                "logs": self.logs
            }            
        elif action == "define":
            result = self.tools.define(action_input)
            return {
                "action": "define",
                "input": action_input,
                "result": result,
                "context": None,
                "logs": self.logs
            }            
        else:
            if not hasattr(self.vector_store, 'index') or self.vector_store.index is None:
                self.log_step("retrieval_error", "Vector store is empty or not initialized")
                response = self.llm_client.generate_response(query)
                self.log_step("generation", "Generated response using LLM directly (no context)")
                return {
                    "action": "direct_llm",
                    "input": query,
                    "result": response,
                    "context": None,
                    "logs": self.logs
                }
            
            try:
                relevant_docs = self.vector_store.search(query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                self.log_step("retrieval", f"Retrieved {len(relevant_docs)} relevant documents")
            except Exception as e:
                self.log_step("retrieval_error", f"Error during retrieval: {str(e)}")
                context = ""
                relevant_docs = []
            
            response = self.llm_client.generate_response(query, context if context else None)
            self.log_step("generation", "Generated response using LLM")
            
            return {
                "action": "rag",
                "input": query,
                "result": response,
                "context": context,
                "retrieved_docs": relevant_docs,
                "logs": self.logs
            }

class RAGAssistant:
    def __init__(self, documents_dir="documents"):
        self.document_loader = DocumentLoader(documents_dir)
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        self.tools = Tools()
        self.agent = Agent(self.vector_store, self.llm_client, self.tools)
        
        self.initialize()
        
    def initialize(self):
        documents = self.document_loader.load_documents()
        if not documents:
            logger.warning("No documents found. Please add documents to the 'documents' directory or run create_documents.py")
            return
            
        chunks = self.text_chunker.split_documents(documents)
        if chunks:
            self.vector_store.add_documents(chunks)
        
    def ask(self, query: str) -> Dict[str, Any]:
        if not hasattr(self.vector_store, 'index') or self.vector_store.index is None:
            logger.warning("Vector store is empty. Answering without context.")
            result = self.llm_client.generate_response(query)
            return {
                "action": "direct_llm",
                "input": query,
                "result": result,
                "context": None,
                "logs": [{"step": "initialization_warning", "details": "No documents in vector store, answering directly"}]
            }
            
        return self.agent.process_query(query)

def run_cli():
    parser = argparse.ArgumentParser(description="RAG-Powered Multi-Agent Q&A Assistant")
    parser.add_argument("--docs", default="documents", help="Directory containing documents")
    parser.add_argument("--create-docs", action="store_true", help="Create sample documents before starting")
    args = parser.parse_args()
    
    if args.create_docs:
        try:
            if os.path.exists("create_documents.py"):
                print("Creating sample documents...")
                import Inflera_RAG_Application.create_documents as create_documents
                create_documents.main()
            else:
                print("Warning: create_documents.py not found. Continuing without creating sample documents.")
        except Exception as e:
            print(f"Error creating sample documents: {str(e)}")
    
    print("Initializing RAG Assistant...")
    assistant = RAGAssistant(args.docs)
    print("Assistant ready! Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
            
        result = assistant.ask(query)
        
        print("\n" + "="*50)
        print(f"Action: {result['action']}")
        
        if result.get('context'):
            print("\nRetrieved Context:")
            print("-"*50)
            print(result['context'][:300] + "..." if len(result['context']) > 300 else result['context'])
            
        print("\nAnswer:")
        print("-"*50)
        print(result['result'])
        
        print("\nDecision Steps:")
        print("-"*50)
        for log in result['logs']:
            print(f"- {log['step']}: {log['details']}")
        print("="*50)


if __name__ == "__main__":
    run_cli()