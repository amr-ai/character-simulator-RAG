import requests.exceptions
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings

class Persona:
    """Base class for all character personas."""
    
    def __init__(self, name: str, index_path: str, prompt_template: str, image_url: str):
        self.name = name
        self.image_url = image_url
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        
        try:
            self.vectorstore = FAISS.load_local(
                index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        except Exception as e:
            print(f"Could not load index for {name}: {e}")
            self.vectorstore = None
            self.retriever = None

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        if self.retriever:
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.chain = None

    def ask(self, question: str) -> str:
        """Invokes the chat chain with a question."""
        if not self.chain:
            return f"Error: The knowledge base for {self.name} is not available."
            
        try:
            response = self.chain.invoke(question)
            # Truncate if response is too long
            if len(response.split()) > 60:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: The AI model service is currently unavailable. Please ensure Ollama is running."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"