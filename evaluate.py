import os
import json
import time
import re
import chromadb
import config
import ollama

# Internal Modules
from src.embeddings import get_embedding_service
from src.generation import get_generation_service
from src.hybrid_retrieval import get_hybrid_retrieval_service, reformulate_query

# ==============================================================================
# 1. THE GOLDEN DATASET
# ==============================================================================
# Replace these with real questions and exact answers from your PDFs
# ==============================================================================
# 1. THE GOLDEN DATASET
# ==============================================================================
GOLDEN_DATA = [
    {
        "question": "What is the title of the fake review detection paper?",
        "expected_answer": "High performance fake review detection using pretrained DeBERTa optimized with Monarch Butterfly paradigm."
    },
    {
        "question": "What three datasets were used to test the proposed framework?",
        "expected_answer": "The proposed framework was tested on the Amazon, Fake Review, and Deceptive Opinion Spam datasets."
    },
    {
        "question": "Why is Monarch Butterfly Optimization (MBO) a good choice for fake review detection?",
        "expected_answer": "MBO is a good choice due to its efficient and adaptive search capabilities, which are crucial for optimizing hyperparameters and feature selection. Its ability to balance exploration and exploitation ensures robust performance even with noisy and variable review data."
    },
    {
        "question": "What classification accuracy did MBO-DeBERTa achieve for detecting fake reviews?",
        "expected_answer": "MBO-DeBERTa attained a classification accuracy of 98% for detecting fake reviews."
    }
]

# ==============================================================================
# 2. THE LLM JUDGE
# ==============================================================================
def judge_response(question: str, generated_answer: str, context: str) -> dict:
    """Uses the local LLM to grade the RAG pipeline's output."""
    
    prompt = (
        "You are an impartial AI judge evaluating a RAG (Retrieval-Augmented Generation) system. "
        "You will be given a User Question, the Retrieved Context, and the System's Answer.\n\n"
        "Grade the response on two metrics from 1 to 5:\n"
        "1. Faithfulness: Is the System's Answer derived completely from the Retrieved Context? (1 = Completely made up, 5 = Perfectly grounded in context)\n"
        "2. Relevance: Does the System's Answer directly address the User Question? (1 = Completely irrelevant, 5 = Perfectly answers the question)\n\n"
        "CRITICAL: You must output ONLY the scores in this exact format: [Faithfulness=X, Relevance=Y]\n\n"
        f"--- User Question ---\n{question}\n\n"
        f"--- Retrieved Context ---\n{context}\n\n"
        f"--- System's Answer ---\n{generated_answer}\n\n"
        "Scores:"
    )

    try:
        response = ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0.0} 
        )
        
        output = response['response'].strip()
        
        # --- NEW SMARTER PARSING LOGIC ---
        f_score, r_score = 0, 0
        
        # 1. Try to find explicit labels regardless of order
        f_match = re.search(r'Faithfulness\s*=?\s*(\d)', output, re.IGNORECASE)
        r_match = re.search(r'Relevance\s*=?\s*(\d)', output, re.IGNORECASE)
        
        if f_match and r_match:
            f_score = int(f_match.group(1))
            r_score = int(r_match.group(1))
        else:
            # 2. Fallback: Just grab the first two numbers inside brackets e.g. [5, 4]
            numbers = re.findall(r'\d', output)
            if len(numbers) >= 2:
                # Assuming the prompt order: Faithfulness first, Relevance second
                f_score = int(numbers[0])
                r_score = int(numbers[1])
                
        if f_score > 0 and r_score > 0:
            return {"faithfulness": f_score, "relevance": r_score}
        else:
            print(f"⚠️ Warning: LLM Judge returned unparseable format: {output}")
            return {"faithfulness": 0, "relevance": 0}
            
    except Exception as e:
        print(f"❌ LLM Judge failed: {e}")
        return {"faithfulness": 0, "relevance": 0}

# ==============================================================================
# 3. EVALUATION RUNNER
# ==============================================================================
def run_evaluation():
    print("🚀 Starting RAG Pipeline Evaluation...")
    
    # Initialize Services
    client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db_data"))
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        print("❌ Error: ChromaDB collection not found. Please upload documents via the UI first.")
        return

    embedding_service = get_embedding_service()
    embedding_service.load_model()
    retrieval_service = get_hybrid_retrieval_service()
    generation_service = get_generation_service()

    # Rebuild BM25 for the evaluation run
    all_docs = collection.get(include=['documents', 'metadatas'])
    if all_docs and all_docs.get('documents'):
        retrieval_service.build_bm25_index(all_docs['documents'], all_docs['metadatas'])

    results_log = []
    total_faithfulness = 0
    total_relevance = 0

    for i, item in enumerate(GOLDEN_DATA):
        print(f"\nEvaluating Question {i+1}/{len(GOLDEN_DATA)}: {item['question']}")
        
        # 1. Retrieve Context
        start_time = time.time()
        intent = generation_service.detect_intent(item['question'])
        query_emb = embedding_service.embed_query(item['question'])
        
        results = retrieval_service.retrieve(
            intent=intent,
            collection=collection,
            query_text=item['question'],
            query_emb=query_emb
        )
        
        context_chunks = []
        if results['scores']:
            top_score = results['scores'][0].get('rerank_score', results['scores'][0].get('confidence', 0))
            if top_score >= 0.2:
                # Use Parent text for generation
                context_chunks = [s.get('parent_text', s['text']) for s in results['scores']]
        
        context_str = "\n".join(context_chunks)

        # 2. Generate Answer
        generated_answer = generation_service.generate_answer(item['question'], context_chunks, [])
        latency = time.time() - start_time

        # 3. Grade Answer
        scores = judge_response(item['question'], generated_answer, context_str)
        
        total_faithfulness += scores['faithfulness']
        total_relevance += scores['relevance']

        # Log Result
        result_entry = {
            "question": item['question'],
            "expected": item['expected_answer'],
            "generated": generated_answer,
            "intent_detected": intent,
            "latency_seconds": round(latency, 2),
            "chunks_retrieved": len(context_chunks),
            "scores": scores
        }
        results_log.append(result_entry)
        
        print(f"   ⏱️ Latency: {result_entry['latency_seconds']}s | 📚 Chunks: {result_entry['chunks_retrieved']} | ⚖️ Scores: F={scores['faithfulness']}, R={scores['relevance']}")

    # ==============================================================================
    # 4. REPORTING
    # ==============================================================================
    avg_f = total_faithfulness / len(GOLDEN_DATA)
    avg_r = total_relevance / len(GOLDEN_DATA)
    
    print("\n" + "="*50)
    print("📊 EVALUATION COMPLETE")
    print("="*50)
    print(f"Average Faithfulness : {avg_f:.1f} / 5.0")
    print(f"Average Relevance    : {avg_r:.1f} / 5.0")
    print("="*50)

    # Save to file
    with open(config.EVALUATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": {"avg_faithfulness": avg_f, "avg_relevance": avg_r}, "details": results_log}, f, indent=4)
    print(f"💾 Full results saved to {config.EVALUATION_OUTPUT_PATH}")

if __name__ == "__main__":
    run_evaluation()