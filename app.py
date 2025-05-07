import os
import streamlit as st
import google.generativeai as genai
import logging
from typing import List, Dict, Any
from Inflera_RAG_Application.rag_agent import DocumentLoader, TextChunker, VectorStore, Tools, Agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlitLLMClient:
    def __init__(self, model="gemini-2.0-flash"):
        self.model = model
        
    def generate_response(self, query: str, context: str = None) -> str:
        try:
            api_key = st.session_state.get("api_key", "")
            if not api_key:
                return "Please provide a valid Gemini API key in the sidebar."
            
            genai.configure(api_key=api_key)
            
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
            
            # Generate content
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"

class StreamlitRAGAssistant:
    def __init__(self, documents_dir="documents"):
        self.document_loader = DocumentLoader(documents_dir)
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore()
        self.llm_client = StreamlitLLMClient()
        self.tools = Tools()
        self.agent = Agent(self.vector_store, self.llm_client, self.tools)
        
        self.initialize()
        
    def initialize(self):
        documents = self.document_loader.load_documents()
        if not documents:
            logger.warning("No documents found. Please add documents to the 'documents' directory.")
            return
            
        chunks = self.text_chunker.split_documents(documents)
        if chunks:
            self.vector_store.add_documents(chunks)
        
    def ask(self, query: str) -> Dict[str, Any]:
        if not st.session_state.get("api_key"):
            return {
                "action": "error",
                "input": query,
                "result": "Please provide a valid Gemini API key in the sidebar.",
                "context": None,
                "logs": [{"step": "api_key_missing", "details": "No API key provided"}]
            }
        
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

def main():
    st.set_page_config(
        page_title="Inflera RAG-Powered Assistant",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Inflera RAG-Powered Assistant")
    st.write("Ask questions about your documents using advanced retrieval and AI!")
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey",
        )
        
        if api_key:
            st.session_state["api_key"] = api_key
            st.success("API key set! You can now use the assistant.")

        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "pdf"])
        
        if uploaded_file is not None:
            if not os.path.exists("documents"):
                os.makedirs("documents")

            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File saved: {uploaded_file.name}")
            
            if "assistant" in st.session_state:
                del st.session_state["assistant"]
                st.info("Assistant reset to load new documents!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if "assistant" not in st.session_state and "api_key" in st.session_state:
        with st.spinner("Initializing assistant..."):
            st.session_state.assistant = StreamlitRAGAssistant()
    
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        if "api_key" not in st.session_state:
            with st.chat_message("assistant"):
                st.write("Please provide a Gemini API key in the sidebar to continue.")
                st.session_state.messages.append({"role": "assistant", "content": "Please provide a Gemini API key in the sidebar to continue."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.assistant.ask(prompt)
                
                st.write(result["result"])                
                st.session_state.messages.append({"role": "assistant", "content": result["result"]})
                
                with st.expander("See details"):
                    st.subheader("Action")
                    st.write(f"The assistant used: **{result['action']}**")
                    
                    if result.get('context'):
                        st.subheader("Retrieved Context")
                        st.text_area("Context", result['context'], height=200)
                    
                    st.subheader("Decision Steps")
                    for log in result['logs']:
                        st.write(f"- **{log['step']}**: {log['details']}")

if __name__ == "__main__":
    main()