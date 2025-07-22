from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

collection_name = "context_engineering_db"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    prefer_grpc = False,
    api_key=os.getenv("QDRANT_API_KEY"),
    )

db = Qdrant(
    client = client,
    embeddings = embeddings,
    collection_name = collection_name
)

query = "what are the sub topics in the context engineering"
docs = db.similarity_search_with_score(query = query, k = 10)



prompt = """

you are a professional agent which only talks about the context engineering, if other than that any other questions are asked 
tell you dont know and here's the user question: {user_question}


here are the data for the reference:
{extracted_data}

use this reference to ans the user question 
"""

prompt_template = ChatPromptTemplate.from_template(prompt)
final_prompt = prompt_template.invoke({
    "user_question": query,
    "extracted_data": docs
})

response = model.invoke(final_prompt)

print(response.content)