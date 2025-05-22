from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
import requests.exceptions

# Einstein's Personal Settings
class EinsteinPersona:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        self.vectorstore = FAISS.load_local(
            "indexes/Albert Einstein", 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        einstein_prompt = """You are Albert Einstein. Respond concisely (1-2 short paragraphs max) with:
        - Scientific insights
        - Philosophical reflections
        - Occasional wit
        - Simple language

        Context:
        {context}

        Question:
        {question}

        Answer briefly as Einstein:"""  # Prompt for Einstein
        
        self.prompt = PromptTemplate(
            template=einstein_prompt,
            input_variables=["context", "question"]
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str):
        try:
            response = self.chain.invoke(question)
            # Limit response to 1-2 sentences
            if len(response.split()) > 50:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: Service unavailable"
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    einstein = EinsteinPersona()
    print("\n🧠 Einstein Chat (Brief Mode)!")
    print("Ask short questions (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        print("\nThinking...")
        response = einstein.ask(user_input)
        print(f"\nEinstein: {response}")
