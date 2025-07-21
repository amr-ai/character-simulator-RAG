import wikipedia
import os
import tempfile
import requests.exceptions
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

class CustomPersona:
    def __init__(self, name: str, characteristics: str, image_url: str):
        self.name = name
        self.image_url = image_url
        self.prompt_template = self._create_prompt_template(name, characteristics)
        
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.index_created = False
        
    def _create_prompt_template(self, name, characteristics):
        return f"""You are {name}. Respond concisely (1-2 short paragraphs max) with the following traits:
{characteristics}

Use the context provided from your life and works to answer the question.

Context:
{{context}}

Question:
{{question}}

Answer briefly as {name}:"""

    def create_index_from_wikipedia(self):
        """Fetches data from Wikipedia and builds the FAISS vector index."""
        try:
            wikipedia.set_lang("en") # Default to English, can be adjusted
            search_results = wikipedia.search(self.name, results=3)
            
            if not search_results:
                return f"No information found for {self.name} on Wikipedia."
            
            page = wikipedia.page(search_results[0], auto_suggest=False)
            content = page.content
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            chunks = text_splitter.split_text(content)
            
            if not chunks:
                 return f"Could not extract text content for {self.name}."
            
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self.prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )

            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            self.index_created = True
            return f"Knowledge base created for {self.name} with {len(chunks)} text fragments."

        except wikipedia.exceptions.PageError:
            return f"Could not find a specific Wikipedia page for '{self.name}'."
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Search for '{self.name}' is ambiguous. Please be more specific. Options: {e.options[:3]}"
        except Exception as e:
            return f"An error occurred during knowledge base creation: {str(e)}"

    def ask(self, question: str):
        if not self.index_created or not self.chain:
            return "The knowledge base is not ready. Please create it first."
            
        try:
            response = self.chain.invoke(question)
            if len(response.split()) > 60:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: The AI model service is unavailable. Please ensure Ollama is running."
        except Exception as e:
            return f"An error occurred: {str(e)}"