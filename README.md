# 🤖 Inflera RAG-Powered Multi-Agent Q&A Assistant

Welcome to **Inflera**, a Retrieval-Augmented Generation (RAG) based multi-agent Question Answering assistant. This project combines document-based retrieval, LLM-driven answering, and simple agentic decision-making to intelligently route user queries and provide accurate, contextual answers.

---

## 🧠 Overview

This assistant performs three major tasks:

1. **Retrieval-Augmented Generation (RAG)**: It fetches relevant context from uploaded documents.
2. **LLM Integration**: It uses a large language model (gemini-2.0-flash) to generate natural language responses.
3. **Agentic Workflow**: It decides whether to use a custom tool (calculator/dictionary) or the RAG + LLM pipeline.

---

## 🚀 Features

- 📄 Upload up to 3–5 short `.txt` documents (FAQs, specs, etc.) from sample_documents
- 🔍 Ask natural-language questions
- 🖥️ Streamlit-based intuitive interface
- 📝 Contextual snippets shown for transparency
- 🔧 Decision logs for agent traceability

---

## 🏗️ Architecture

```text
             ┌────────────┐
             │ User Query │
             └─────┬──────┘
                   ↓
          ┌──────────────────┐
          │  Keyword Router  │
          └─────┬────┬───────┘
        ┌──────┘    └───────────────┐
   ┌────────────┐           ┌──────────────┐
   │ Calculator │           │ Dictionary   │
   └────────────┘           └──────────────┘
                                │
                                ↓
                        ┌──────────────┐
                        │   RAG + LLM  │
                        └──────┬───────┘
                               ↓
                      ┌───────────────────┐
                      │ Final Answer + UI │
                      └───────────────────┘
