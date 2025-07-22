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

You are a professional Context Engineering agent. You ONLY answer questions about Context Engineering.
If the user asks anything outside of Context Engineering, respond with "I don't know." without any additional commentary.

Use the following reference data to answer the user's question:
{extracted_data}

User Question:
{user_question}

"""

prompt_template = ChatPromptTemplate.from_template(prompt)
final_prompt = prompt_template.invoke({
    "user_question": query,
    "extracted_data": docs
})

response = model.invoke(final_prompt)

print(response.content)