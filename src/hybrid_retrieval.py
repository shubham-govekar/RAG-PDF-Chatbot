"""
Retrieval Service using Strategy Pattern.
Handles: Metadata Lookup, Summary Extraction, and Hybrid Search.
"""
import chromadb
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import config
import numpy as np

# =========================================================
# 1. STRATEGY INTERFACE (The Contract)
# =========================================================
class RetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(self, collection, query_text: str, query_emb: List[float], filter_pdfs: List[str] = None) -> Dict[str, Any]:
        pass

# =========================================================
# 2. CONCRETE STRATEGIES
# =========================================================
class MetadataStrategy(RetrievalStrategy):
    """Fetches just the headers (Chunks 0-2) for author/title info."""
    def retrieve(self, collection, query_text, query_emb, filter_pdfs=None):
        results = {'scores': []}
        # Scan all metadata to find headers (Fast for local apps)
        all_docs = collection.get(include=['documents', 'metadatas'])
        
        for i, meta in enumerate(all_docs['metadatas']):
            if filter_pdfs and meta['pdf_name'] not in filter_pdfs:
                continue
            
            # Heuristic: Authors/Titles are always in the first 2 chunks
            if meta.get('chunk_index', 999) < 2:
                results['scores'].append({
                    'text': all_docs['documents'][i],
                    'confidence': 1.0,  # Max confidence for direct lookup
                    'pdf_name': meta['pdf_name'],
                    'chunk_index': meta['chunk_index'],
                    'parent_text': meta.get('parent_text', all_docs['documents'][i])
                })
        
        # Sort by PDF and Index
        results['scores'].sort(key=lambda x: (x['pdf_name'], x['chunk_index']))
        return results

class SummaryStrategy(RetrievalStrategy):
    """Fetches the Introduction/Abstract (Chunks 0-5)."""
    def retrieve(self, collection, query_text, query_emb, filter_pdfs=None):
        results = {'scores': []}
        all_docs = collection.get(include=['documents', 'metadatas'])
        
        for i, meta in enumerate(all_docs['metadatas']):
            if filter_pdfs and meta['pdf_name'] not in filter_pdfs:
                continue
                
            # Heuristic: Abstracts are usually within the first 6 chunks
            if meta.get('chunk_index', 999) < 6:
                results['scores'].append({
                    'text': all_docs['documents'][i],
                    'confidence': 1.0,
                    'pdf_name': meta['pdf_name'],
                    'chunk_index': meta['chunk_index'],
                    'parent_text': meta.get('parent_text', all_docs['documents'][i])
                })
        
        results['scores'].sort(key=lambda x: (x['pdf_name'], x['chunk_index']))
        return results

class HybridSearchStrategy(RetrievalStrategy):
    """
    Standard RAG: BM25 + Vector Search + Re-ranking.
    """
    def __init__(self):
        self.bm25 = None
        self.doc_map = {} # Map chunk_index -> chunk_data for BM25 lookup
        
    def build_bm25(self, documents: List[str], metadatas: List[Dict]):
        """Lazy load BM25 and build a mapping for fast lookup"""
        from rank_bm25 import BM25Okapi
        
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Store docs/meta by index so we can retrieve them by BM25 ID
        self.doc_map = {
            i: {'text': doc, 'meta': metadatas[i]} 
            for i, doc in enumerate(documents)
        }

    def retrieve(self, collection, query_text, query_emb, filter_pdfs=None):
        # --- 1. VECTOR SEARCH (Dense) ---
        params = {"n_results": config.TOP_K_RETRIEVAL}
        if filter_pdfs:
            params["where"] = {"pdf_name": {"$in": filter_pdfs}}
            
        vector_results = collection.query(
            query_embeddings=[query_emb],
            **params
        )
        
        # Normalize Vector Scores (0 to 1) using Distance
        # Chroma returns distance (lower is better), we need score (higher is better)
        vector_candidates = {}
        if vector_results['ids'] and vector_results['distances']:
            max_dist = max([max(d) for d in vector_results['distances']]) + 0.01
            
            for i, id_val in enumerate(vector_results['ids'][0]):
                dist = vector_results['distances'][0][i]
                meta = vector_results['metadatas'][0][i]
                
                # Check filter (Double check)
                if filter_pdfs and meta['pdf_name'] not in filter_pdfs:
                    continue

                # Convert distance to similarity score
                score = 1 - (dist / max_dist)
                
                # Create a unique key for deduplication
                key = f"{meta['pdf_name']}_{meta['chunk_index']}"
                vector_candidates[key] = {
                    'text': meta.get('parent_text', vector_results['documents'][0][i]),
                    'score': score,
                    'meta': meta,
                    'source': 'vector'
                }

        # --- 2. BM25 SEARCH (Sparse) ---
        bm25_candidates = {}
        if self.bm25:
            tokenized_query = query_text.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Get Top Indices
            top_n = config.TOP_K_RETRIEVAL
            # Sort indices by score descending
            top_indices = np.argsort(doc_scores)[::-1][:top_n]
            
            # Normalize BM25 scores
            max_bm25 = max(doc_scores) + 0.01 if len(doc_scores) > 0 else 1
            
            for idx in top_indices:
                if doc_scores[idx] <= 0: continue # Skip irrelevant
                
                chunk_data = self.doc_map.get(idx)
                if not chunk_data: continue
                
                meta = chunk_data['meta']
                if filter_pdfs and meta['pdf_name'] not in filter_pdfs:
                    continue
                    
                key = f"{meta['pdf_name']}_{meta['chunk_index']}"
                norm_score = doc_scores[idx] / max_bm25
                
                bm25_candidates[key] = {
                    'text': meta.get('parent_text', chunk_data['text']),
                    'score': norm_score,
                    'meta': meta,
                    'source': 'bm25'
                }

        # --- 3. HYBRID MERGE (Weighted Reciprocal Rank-ish) ---
        final_candidates = {}
        all_keys = set(vector_candidates.keys()) | set(bm25_candidates.keys())
        
        alpha = config.HYBRID_ALPHA # e.g. 0.5
        
        for key in all_keys:
            v_score = vector_candidates.get(key, {}).get('score', 0)
            k_score = bm25_candidates.get(key, {}).get('score', 0)
            
            # Weighted Combination
            hybrid_score = (v_score * (1 - alpha)) + (k_score * alpha)
            
            # Get the data object (prefer vector source for metadata richness if available)
            base_obj = vector_candidates.get(key) or bm25_candidates.get(key)
            
            final_candidates[key] = {
                'text': base_obj['text'],
                'confidence': hybrid_score, # Use hybrid score as confidence
                'pdf_name': base_obj['meta']['pdf_name'],
                'chunk_index': base_obj['meta']['chunk_index'],
                'parent_text': base_obj['meta'].get('parent_text', base_obj['text'])
            }

        # Convert to list and sort
        sorted_results = sorted(
            final_candidates.values(), 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        return {'scores': sorted_results[:config.TOP_K_RETRIEVAL]}

# =========================================================
# 3. RETRIEVAL SERVICE (The Context/Facade)
# =========================================================
from flashrank import Ranker, RerankRequest 

class RetrievalService:
    def __init__(self):
        self.strategies = {
            'METADATA': MetadataStrategy(),
            'SUMMARY': SummaryStrategy(),
            'SEARCH': HybridSearchStrategy()
        }
        self.ranker = Ranker() if config.ENABLE_RERANKING else None

    def build_bm25_index(self, docs, metas):
        # Helper to initialize the Hybrid strategy
        # FIX: We must pass 'metas' here so the strategy can map keywords to chunks
        self.strategies['SEARCH'].build_bm25(docs, metas)

    def retrieve(self, intent: str, collection, query_text, query_emb, filter_pdfs=None):
        print(f"\n🔍 DEBUG: Intent = {intent}")
        
        # 1. Select Strategy
        strategy = self.strategies.get(intent, self.strategies['SEARCH'])
        
        # 2. Guard Clause for Missing Index
        if intent == 'SEARCH' and not self.strategies['SEARCH'].bm25:
             print("⚠️ WARNING: BM25 index not found. Relying on Vector Search only.")
             
        # 3. Execute Basic Retrieval
        results = strategy.retrieve(collection, query_text, query_emb, filter_pdfs)
        
        # --- 4. DEDUPLICATION LAYER (The "Ghost Buster") ---
        # This removes chunks that have identical text content, 
        # ensuring Ref 1, 2, and 3 are unique.
        seen_content = set()
        unique_results = []
        
        for item in results['scores']:
            # Normalize text to catch duplicates
            content_signature = item['text'].strip()
            
            if content_signature not in seen_content:
                seen_content.add(content_signature)
                unique_results.append(item)
        
        results['scores'] = unique_results
        print(f"🔍 DEBUG: Deduplicated to {len(unique_results)} chunks")
        
        # --- 5. RE-RANKING LAYER (FlashRank) ---
        # This pushes the most relevant definition (Ref 4) to the top (Ref 1)
        if intent == 'SEARCH' and config.ENABLE_RERANKING and results['scores']:
            print("🔍 DEBUG: Starting FlashRank re-ranking...")
            
            try:
                # Prepare passages for FlashRank
                pass_ages = [
                    {"id": str(i), "text": r['text'], "meta": r} 
                    for i, r in enumerate(results['scores'])
                ]
                
                rerank_req = RerankRequest(query=query_text, passages=pass_ages)
                reranked = self.ranker.rerank(rerank_req)
                
                final_results = []
                for r in reranked:
                    # Find the original item metadata
                    item = next(p['meta'] for p in pass_ages if p['id'] == str(r['id']))
                    
                    # 1. Store Raw Score (0.0 to 1.0) for Logic Gate
                    # FlashRank returns a float between 0 and 1.
                    item['rerank_score'] = r['score']
                    
                    # 2. Update Confidence for UI Display
                    # We overwrite the "Vector Confidence" with the "Smart Reranker Confidence"
                    # so the UI shows the number that actually matters.
                    item['confidence'] = r['score'] 
                    
                    final_results.append(item)
                
                # Replace results with re-ranked list
                results['scores'] = final_results[:config.TOP_K_RERANK]
                
                # Debug print to confirm it worked
                if results['scores']:
                    print(f"✅ DEBUG: Re-ranking complete. Top score: {results['scores'][0].get('rerank_score', 0):.4f}")
                
            except Exception as e:
                print(f"⚠️ Re-ranking failed: {e}. Returning base results.")
                        # --- 6. CRITICAL RETURN STATEMENT ---
        return results  # <--- This prevents the 'NoneType' error

_retrieval_service = None
def get_hybrid_retrieval_service():
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service

# Keep this for backward compatibility with app.py imports
def reformulate_query(query, context):
    # This just needs to be importable; the logic is likely in utils or handled elsewhere
    # depending on your exact file structure. If you need the actual logic here:
    import ollama
    if not context: return query
    # ... (Your regex check) ...
    return query # Placeholder if you import the real one in app.py