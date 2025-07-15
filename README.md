# 🤖 Inflera RAG-Powered Q\&A Assistant

Welcome to **Inflera**, a Retrieval-Augmented Generation (RAG) based multi-agent Question Answering assistant. This project combines document-based retrieval, LLM-driven answering, and simple agentic decision-making to intelligently route user queries and provide accurate, contextual answers.

## 🧠 Overview

This assistant performs three major tasks:

1. **Retrieval-Augmented Generation (RAG)**: It fetches relevant context from uploaded documents.
2. **LLM Integration**: It uses a large language model to generate natural language responses.
3. **Agentic Workflow**: It decides whether to use a custom tool (calculator/dictionary) or the RAG + LLM pipeline.

## ✨ Try It Now
🔗 **Live Demo**: [Click here to try the app](https://inflera-rag-app.streamlit.app/)

## 🚀 Features

* Upload short `.txt` documents (FAQs, specs, etc.)
* Ask natural-language questions
* Intelligent routing to:

  * Calculator tool (if query involves math)
  * Dictionary tool (if query involves definitions)
  * RAG + LLM pipeline (for general queries)
    
* Streamlit-based intuitive interface
* Contextual snippets shown for transparency
* Decision logs for agent traceability


## 🛠️ Installation & Running the App

1. **Fork & then clone the Repo**

   ```bash
   git clone https://github.com/Shivani-Sharma-23/Inflera_RAG_Application.git
   cd Inflera_RAG_Application
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```


3. **Create Sample Documents**

      ```bash
   python create_documents.py
   ```
   
4. **Set Gemini API Key**

   In `rag_agent.py` add your `gemini-2.0-flash` api key

   ```bash
   api_key = os.environ.get("GOOGLE_API_KEY", "you_api_key")
   ```
5. **Run on CLI**

   ```bash
   python rag_agent.py
   ```
6. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```


## 💡 Example Prompts

* `Calculate 45 * 3 - 12` → Uses Calculator Tool
* `Define artificial intelligence` → Uses Dictionary Tool
* `When was Techcorp founded ?` → Uses RAG + LLM
