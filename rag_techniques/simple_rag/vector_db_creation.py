from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
# Load the PDF document
loader = PyPDFLoader("resources/data.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDF")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  
    chunk_overlap=400,  
    separators=["\n\n", "\n", " ", ""]  
)

texts = text_splitter.split_documents(documents)


# load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print(" Embedding Model Loaded...................... ")

url = ""
collection_name = "context_engineering_db"

print(f"Number of text chunks: {len(texts)}")
print("Creating Qdrant index... This may take a while for large documents.")

try:
    # Process documents in smaller batches to avoid timeout
    batch_size = 10  
    
    if len(texts) <= batch_size:
        # If we have few documents, process all at once
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            collection_name=collection_name,
            timeout=120,  # Increase timeout to 2 minutes
        )
    else:
        # Process in batches for large document sets
        print(f"Processing {len(texts)} documents in batches of {batch_size}")
        
        # Create initial index with first batch
        first_batch = texts[:batch_size]
        qdrant = Qdrant.from_documents(
            first_batch,
            embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            collection_name=collection_name,
            timeout=120,
        )
        print(f"Processed batch 1/{(len(texts) + batch_size - 1) // batch_size}")
        
        # Add remaining documents in batches
        for i in range(batch_size, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            qdrant.add_documents(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print(f"Processed batch {batch_num}/{total_batches}")

    print("Qdrant index created successfully!")
    
except Exception as e:

    raise