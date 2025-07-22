# Simple RAG Example

This folder demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline using LangChain, Qdrant, and Gemini. The sample context is from a context engineering survey paper.

## Setup Steps

1. **Clone the repository** and navigate to this folder:
   ```sh
   git clone <repo-url>
   cd rag_techniques/simple_rag
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Create a `.env` file in this folder with the following keys:
     ```env
     QDRANT_URL=<your-qdrant-url>
     QDRANT_API_KEY=<your-qdrant-api-key>
     GOOGLE_API_KEY = <gemini api key>
     ```

4. **Prepare the vector database**:
   - Run the script to create the vector DB (if not already done):
     ```sh
     python vector_db_creation.py
     ```

5. **Run the RAG pipeline**:
   - Execute the main script:
     ```sh
     python simple_rag.py
     ```

## What it does
- Embeds the context engineering survey data into Qdrant.
- Runs a sample query: "what are the sub topics in the context engineering".
- Retrieves relevant chunks and generates an answer using Gemini.

---

**Note:**
- The code is intentionally kept minimal for easy reading and learning.
- Replace the context and query as needed for your own use case.
