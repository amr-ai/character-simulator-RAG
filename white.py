from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
import requests.exceptions

class WhitePersona:
    def __init__(self):
        # Load the model and embeddings
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")
        
        # Load the prebuilt vector index for Walter White
        self.vectorstore = FAISS.load_local(
            "indexes/Walter White", 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Define a concise prompt tailored to Walter White
        white_prompt = """You are Walter White from Breaking Bad. Respond concisely (1-2 short paragraphs max) with:
        - Chemistry knowledge
        - Strategic thinking
        - Moments of pride or ego
        - Justified rationalization
        - References to your family or your "business"
        - Occasional use of your alter ego "Heisenberg"

        Context:
        {context}

        Question:
        {question}

        Answer briefly as Walter White:"""
        
        self.prompt = PromptTemplate(
            template=white_prompt,
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
            # Trim long responses
            if len(response.split()) > 50:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: Service unavailable"
        except Exception as e:
            return f"Error: {str(e)}"

# For direct CLI testing
if __name__ == "__main__":
    white = WhitePersona()
    print("\n🧪 Walter White Chat (Brief Mode)!")
    print("Ask short questions (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        print("\nThinking...")
        response = white.ask(user_input)
        print(f"\nWalter: {response}")
