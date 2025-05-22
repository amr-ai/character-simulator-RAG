from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
import requests.exceptions

# Kafka's Personal Settings
class KafkaPersona:
    def __init__(self):
        # Load the model and embeddings
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="all-minilm")

        # Load Kafka's prebuilt vector index
        self.vectorstore = FAISS.load_local(
            "indexes/Franz Kafka", 
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Define Kafka-style prompt template
        kafka_prompt = """You are Franz Kafka. Respond concisely (1-2 short paragraphs max) with:
        - Existential themes
        - Absurdist perspectives
        - Subtle dark humor
        - Melancholic undertones
        - References to alienation and bureaucracy

        Context:
        {context}

        Question:
        {question}

        Answer briefly as Kafka:"""
        
        self.prompt = PromptTemplate(
            template=kafka_prompt,
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
        # Handle the question and return a concise response
        try:
            response = self.chain.invoke(question)
            if len(response.split()) > 50:
                return ". ".join(response.split(". ")[:2]) + "."
            return response
        except requests.exceptions.ConnectionError:
            return "Error: Service unavailable"
        except Exception as e:
            return f"Error: {str(e)}"

# For direct CLI testing
if __name__ == "__main__":
    kafka = KafkaPersona()
    print("\n📚 Kafka Chat (Brief Mode)!")
    print("Ask short questions (type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        print("\nThinking...")
        response = kafka.ask(user_input)
        print(f"\nKafka: {response}")
