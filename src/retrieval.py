from typing import List, Dict, Any, Optional
from flashrank import Ranker, RerankRequest
import config

class RetrievalService:
    def __init__(self):
        self.ranker = None

    def _load_ranker(self):
        """Lazy load re-ranker"""
        if self.ranker is None:
            # Use the config variable instead of hardcoding the string
            self.ranker = Ranker(
                model_name=config.RERANKER_MODEL, 
                cache_dir=config.RERANKER_CACHE_DIR # <--- UPDATED
            )

    def retrieve_and_rerank(
        self, 
        collection: Any, 
        query_emb: List[float], 
        query_text: str, 
        n_results: int = config.INITIAL_RETRIEVAL_COUNT,
        filter_pdfs: Optional[List[str]] = None
    ) -> Dict[str, List[Any]]:
        """
        Retrieve from Chroma, optionally filter and rerank.
        Returns a dict with 'scores' containing a list of result objects.
        """
        where_clause = {"pdf_name": {"$in": filter_pdfs}} if filter_pdfs else None
        
        # Query ChromaDB
        res = collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=where_clause
        )

        # Handle case where no results are found
        if not res['documents'] or not res['documents'][0]:
            return {"scores": []}

        docs = res['documents'][0]
        metas = res['metadatas'][0]
        ids = res['ids'][0]
        dists = res['distances'][0]

        # 1. Convert Chroma results to intermediate objects
        results_objs = []
        for i, doc in enumerate(docs):
            # Convert distance to similarity score (0-1 range)
            # Assuming cosine distance: 0=exact match, 2=opposite
            # Simple inversion: score = 1 - distance
            base_score = max(0.0, 1.0 - dists[i])
            
            item = {
                "id": ids[i],
                "text": doc,
                "meta": metas[i],
                "pdf_name": metas[i].get('pdf_name', 'Unknown'),
                "chunk_index": metas[i].get('chunk_index', 0),
                "score": base_score,      # Used for sorting
                "confidence": base_score, # Displayed in UI
                "rerank_score": None
            }
            results_objs.append(item)

        # 2. Rerank if enabled
        if config.ENABLE_RERANKING and results_objs:
            self._load_ranker()
            
            # Prepare request for FlashRank
            passages = [
                {"id": r["id"], "text": r["text"], "meta": r["meta"]} 
                for r in results_objs
            ]
            rerank_req = RerankRequest(query=query_text, passages=passages)
            ranked_results = self.ranker.rerank(rerank_req)
            
            # Create a lookup for rerank scores
            rerank_map = {r['id']: r['score'] for r in ranked_results}
            
            # Update objects with blended scores
            for r in results_objs:
                rr_score = rerank_map.get(r['id'], 0.0)
                r['rerank_score'] = rr_score * 100  # Convert 0-1 to percentage for UI
                
                # Blend: 70% reranker + 30% vector score
                # Note: rr_score is usually 0-1, r['score'] is 0-1
                blended = (rr_score * 0.7) + (r['score'] * 0.3)
                r['confidence'] = blended * 100     # Convert to percentage for UI logic
                r['score'] = blended                # Update sort key

        else:
            # If no reranking, just convert confidence to percentage for UI consistency
            for r in results_objs:
                r['confidence'] = r['score'] * 100

        # 3. Filter by threshold (config threshold is 0.40, likely meaning 40%)
        # Note: config.SIMILARITY_THRESHOLD is usually 0.4.
        # Since we converted confidence to %, we compare: confidence > threshold * 100
        threshold_pct = config.SIMILARITY_THRESHOLD * 100
        filtered = [r for r in results_objs if r['confidence'] >= threshold_pct]

        # 4. Sort and Limit
        filtered.sort(key=lambda x: x['score'], reverse=True)
        top_k = filtered[:config.FINAL_RESULTS_COUNT]

        # Return format expected by app.py: results['scores'] is the list of chunks
        return {
            "scores": top_k
        }

def reformulate_query(query: str, context: str) -> str:
    """Prepend context if it looks like a follow-up question."""
    if not context or len(query.split()) > 6:
        return query
    
    # Simple logic: if query is short, assume it relies on previous context
    return f"Context: {context}\nQuestion: {query}"

_retrieval_service = None

def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service