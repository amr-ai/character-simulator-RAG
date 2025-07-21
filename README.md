# RAG-Personas-Chat

*RAG-Personas-Chat* is a modular, web-based conversational AI application built with FastAPI. It leverages Retrieval-Augmented Generation (RAG) and supports customizable personas, enabling dynamic and contextually enriched responses tailored to different roles such as assistants, therapists, and educators.

---

## 🧠 Project Overview

This application enables users to interact with AI-powered personas that use both pretrained language models and a custom document knowledge base. Users can select a persona and ask questions or initiate conversations that blend personality-driven dialogue with document retrieval.

---

## 🚀 Features

- 🔍 *Retrieval-Augmented Generation (RAG):* Combines LLMs with document indexing for context-aware responses.
- 🎭 *Multi-Persona Support:* Each persona has a unique tone, objective, and conversational style.
- 🌐 *Web Interface:* Intuitive UI for selecting personas, uploading documents, and chatting.
- 📂 *Custom Indexing:* Users can index their own documents using FAISS via the rag_indexer.py.
- 🛠 *FastAPI Backend:* Lightweight and asynchronous server for scalable deployment.

---

## 🗂 Folder Structure


RAG-Personas-Chat/
│
├── __pycache__/         # Compiled Python files
├── data/                # Source documents for indexing
├── indexes/             # FAISS vector store index files
├── personas/            # JSON configuration files for different personas
├── static/              # CSS, JS, and assets for the frontend
├── templates/           # HTML templates (Jinja2)
├── venv/                # Python virtual environment
├── main.py              # FastAPI app entry point
└── rag_indexer.py       # Script for indexing documents


---

## 🖼 Screenshots

### 💬 Chat Interface
![Chat](https://raw.githubusercontent.com/ghreeb1/RAG-Personas-Chat/master/data/1.png)

### 🎭 Persona Selection
![Persona](https://raw.githubusercontent.com/ghreeb1/RAG-Personas-Chat/master/data/2.png)

### 🗂 Indexing Tool
![Indexer](https://raw.githubusercontent.com/ghreeb1/RAG-Personas-Chat/master/data/3.png)

---

## ⚙ Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:

bash
git clone https://github.com/your-username/RAG-Personas-Chat.git
cd RAG-Personas-Chat


2. Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


3. Install dependencies:

bash
pip install -r requirements.txt


---

## 🧾 Usage

### Step 1: Index Your Documents

Place your documents in the data/ folder, then run:

bash
python rag_indexer.py


This will generate FAISS indexes in the indexes/ directory.

### Step 2: Start the Server

bash
python main.py


Visit http://localhost:8000 in your browser.

---

## 👤 Persona Configuration

Create a new persona by adding a JSON file to the personas/ directory. Example:

json
{
  "name": "Therapist",
  "description": "A calm and empathetic mental health assistant.",
  "style": "Supportive and therapeutic"
}


---

## 📌 Technologies Used

- *FastAPI* – Web server framework
- *LangChain* – LLM orchestration
- *FAISS* – Document similarity search
- *OpenAI / Llama / Local Models* – Language model providers
- *HTML, CSS, JS* – Frontend interface

---

## 📄 License

This project is licensed under the *MIT License*. See the [LICENSE](LICENSE) file for details.

---

## 👥 Participated in This Project

<h3 align="left">Connect with Us:</h3>

<p align="left">
  <a href="https://linkedin.com/in/mohamed-khaled-3a9021263" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Mohamed%20Khaled-blue?style=flat&logo=linkedin" alt="LinkedIn - Mohamed Khaled" />
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/amr-mohamed-9ab096232/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Amr%20Mohamed-blue?style=flat&logo=linkedin" alt="LinkedIn - Amr Mohamed" />
  </a>
</p>

---
