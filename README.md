Local Hybrid RAG Chatbot
I built this RAG (Retrieval-Augmented Generation) system because I was frustrated with standard RAG pipelines. Too often, they would miss specific keywords or confidently hallucinate answers when the context wasn't there.

This project fixes those issues by running everything locally (privacy-first) and using a "Hybrid Search" approach—combining vector search with old-school keyword matching to make sure we actually find what you're looking for.

How it works
The pipeline is pretty straightforward:

Your Query comes in.

We search the documents using Vectors (meaning) and Keywords (exact text).

We merge those results and use a Re-ranker to pick the absolute best chunks.

If the best chunk isn't good enough (low score), we reject it.

If it is good, we pass it to Llama 3.2 to generate the final answer.

```
    graph LR
    A[User Query] --> B(Hybrid Search)
    B --> C{Re-ranker Check}
    C -- Score is Low --> D[Say 'I don't know']
    C -- Score is High --> E[Llama 3.2 generates answer]
```

Tech Stack
I kept this lightweight and local so you don't need API keys or cloud credits.

App: Python & Streamlit

LLM: Ollama (running Llama 3.2)

Database: ChromaDB (runs in memory, resets when you close the app)

Re-ranking: FlashRank (super fast, runs on CPU)

Getting Started
1. Prereqs
You need Python installed, and you need Ollama running in the background.

Download it at ollama.com

Run this in your terminal to grab the model: ollama pull llama3.2

2. Install
Clone the repo and grab the python requirements.

git clone https://github.com/yourusername/pdf-rag-chatbot.git

cd pdf-rag-chatbot

pip install -r requirements.txt

3. Run it
streamlit run app.py

Then just open your browser to http://localhost:8501, drop a PDF in the sidebar, and start chatting.

Project Structure
If you want to dig into the code:

app.py: The frontend UI (Streamlit).

src/hybrid_retrieval.py: Where the search magic happens (BM25 + Chroma).

src/generation.py: The logic for sending prompts to Ollama.

src/advanced_chunking.py: How we chop up the PDFs so the AI understands them better.

License
MIT License. Feel free to fork it and break things!