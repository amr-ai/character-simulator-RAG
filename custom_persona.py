import wikipedia
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
import requests.exceptions
import streamlit as st

class CustomPersona:
    def __init__(self, name, description, prompt_template):
        self.name = name
        self.description = description
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "index")
        
        # Load the model and embeddings
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        
        # Set up the custom prompt
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Initialize the vector store (will be created later)
        self.vectorstore = None
        
    def create_index_from_wikipedia(self):
        """Create a vector index from Wikipedia data"""
        try:
            # Set Wikipedia language based on character name (Arabic or English)
            wikipedia.set_lang("ar" if any(ord(c) > 127 for c in self.name) else "en")
            search_results = wikipedia.search(self.name, results=3)
            
            if not search_results:
                return f"No information found about {self.name} on Wikipedia."
                
            # Try fetching the page for the character
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                # If multiple options, choose the first
                page = wikipedia.page(e.options[0], auto_suggest=False)
            
            # Get the content of the page
            content = page.content
            
            # Split text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_text(content)
            
            # Generate embeddings and store them
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
            
            # Set up retriever and chain
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            return f"Index successfully created for {self.name} using {len(chunks)} chunks."
            
        except Exception as e:
            return f"An error occurred while creating the index: {str(e)}"
    
    def ask(self, question: str):
        """Ask the custom persona a question"""
        if self.vectorstore is None:
            return "Index is not ready yet. Please create the index first."
            
        try:
            response = self.chain.invoke(question)
            # Trim long responses
            if len(response.split()) > 50:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: Service unavailable"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files (if needed)"""
        # Add cleanup code here if necessary
        pass

def create_prompt_template(name, characteristics):
    """Generate a custom prompt template based on character traits"""
    prompt = f"""You are {name}. Respond concisely (1-2 short paragraphs max) with:
    {characteristics}

    Context:
    {{context}}

    Question:
    {{question}}

    Answer briefly as {name}:"""
    
    return prompt
