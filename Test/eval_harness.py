"""
Retrieval and faithfulness evaluation harness for the PDF-RAG pipeline.

Functionality:
1. Executes a predefined set of QA pairs through the retrieval pipeline 
   (reformulate -> retrieve -> grade), bypassing the UI.
2. Calculates retrieval metrics (Hit Rate@k and MRR) by verifying if the 
   expected source document is present in the final filtered results.
3. (Optional) Executes full generation and evaluates response faithfulness 
   (groundedness) using an LLM-as-a-judge approach.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure `src` is importable when executed from the project root or Test/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import GraphState  # noqa: E402
from src.nodes import (  # noqa: E402
    reformulate_query_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    big_llm,
)
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402


# --------------------------------------------------------------------------
# 1. EVALUATION SET
# --------------------------------------------------------------------------

@dataclass
class EvalCase:
    question: str
    expected_source: str
    expect_answer: str = ""  # Optional hint for manual review; not auto-graded


# Note: `expected_source` must exactly match the `source` metadata stored 
# in the Chroma docstore during ingestion (e.g., using raw strings for 
# Windows file paths to prevent escape character issues).

EVAL_SET: list[EvalCase] = [
    # --- data\Retrieval-Augmented_Generation_RAG.pdf (Klesel & Wittmann, BISE catchword) ---
    EvalCase(
        question="What is RAG?",
        expected_source=r"data\Retrieval-Augmented_Generation_RAG.pdf",
        expect_answer="Combines LLM generation with retrieval from an external/non-parametric database",
    ),
    EvalCase(
        question="What are the advantages and disadvantages of RAG?",
        expected_source=r"data\Retrieval-Augmented_Generation_RAG.pdf",
        expect_answer="Advantages: contextual understanding, reduced hallucination, grounding, guardrails, "
                      "lower cost. Challenges: data mgmt/MLOps, bias, blinkered chunk effect, retrieval effectiveness",
    ),
    EvalCase(
        question="What is grounding in the context of RAG?",
        expected_source=r"data\Retrieval-Augmented_Generation_RAG.pdf",
        expect_answer="Providing references/links to the source documents behind a generated answer",
    ),
    EvalCase(
        question="What is the blinkered chunk effect?",
        expected_source=r"data\Retrieval-Augmented_Generation_RAG.pdf",
        expect_answer="A retrieved chunk lacks full document context, limiting comprehensive understanding",
    ),
    EvalCase(
        question="What is the difference between parametric and non-parametric memory in RAG?",
        expected_source=r"data\Retrieval-Augmented_Generation_RAG.pdf",
        expect_answer="Parametric = knowledge stored in LLM weights; non-parametric = external database",
    ),

    # --- data\research paper.pdf (MBO-DeBERTa fake review detection) ---
    EvalCase(
        question="What is MBO-DeBERTa?",
        expected_source=r"data\research paper.pdf",
        expect_answer="A fake review detection model combining DeBERTa with Monarch Butterfly Optimization",
    ),
    EvalCase(
        question="What accuracy did the proposed model achieve on the Fake Review dataset?",
        expected_source=r"data\research paper.pdf",
        expect_answer="98% accuracy",
    ),
    EvalCase(
        question="What method was used to detect adversarial attacks on the fake review model?",
        expected_source=r"data\research paper.pdf",
        expect_answer="Fast Gradient Sign Method (FGSM), tested at 10/20/50% noise levels",
    ),
    EvalCase(
        question="Which three datasets were used to evaluate MBO-DeBERTa?",
        expected_source=r"data\research paper.pdf",
        expect_answer="Amazon fake review, Fake Review, and Deceptive Opinion Spam Corpus datasets",
    ),
    EvalCase(
        question="What is the Monarch Butterfly Optimization algorithm inspired by?",
        expected_source=r"data\research paper.pdf",
        expect_answer="Migratory patterns of monarch butterflies",
    ),

    # --- data\support paper 2.pdf (Jayasinghe & Dassanayaka, CNN fake review detection) ---
    EvalCase(
        question="What loss function do the authors propose for fake review detection?",
        expected_source=r"data\support paper 2.pdf",
        expect_answer="Hybrid Adaptive Cross-Entropy (HACE)",
    ),
    EvalCase(
        question="What ROC-AUC score did the CNN model achieve overall?",
        expected_source=r"data\support paper 2.pdf",
        expect_answer="0.961 ROC-AUC",
    ),
    EvalCase(
        question="How did the authors use verified purchase status to label fake reviews?",
        expected_source=r"data\support paper 2.pdf",
        expect_answer="Reviewers with over 50% non-verified-purchase reviews had all their reviews "
                      "labeled fake; remaining reviews labeled via cosine similarity threshold",
    ),
    EvalCase(
        question="What tokenizer was used to preprocess the Amazon reviews?",
        expected_source=r"data\support paper 2.pdf",
        expect_answer="NLTK TweetTokenizer",
    ),
    EvalCase(
        question="What word embeddings were used in the CNN model?",
        expected_source=r"data\support paper 2.pdf",
        expect_answer="Pre-trained GloVe embeddings (100 dimensions, 6B tokens)",
    ),
    
    # --- data\support paper.pdf (Chen et al. PosAtt-BiLSTM fake review detection) ---
    EvalCase(
        question="What does NN-SPE stand for in the context of the proposed fake review detection model?",
        expected_source=r"data\support paper.pdf",
        expect_answer="Non-Negative Sinusoidal Positional Encoding [cite: 14]",
    ),
    EvalCase(
        question="Which three public datasets were used to evaluate the PosAtt-BiLSTM model?",
        expected_source=r"data\support paper.pdf",
        expect_answer="Spam, Yelp Hotel, and Yelp Restaurant datasets [cite: 17]",
    ),
    EvalCase(
        question="What is the primary purpose of the Hybrid Attention Mechanism (HAM) in the PosAtt-BiLSTM architecture?",
        expected_source=r"data\support paper.pdf",
        expect_answer="It applies weights to both global and local features to ensure the accurate capture and transmission of key information [cite: 16]",
    ),
    EvalCase(
        question="What classification accuracy and F1 score did the PosAtt-BiLSTM model achieve on the Spam dataset?",
        expected_source=r"data\support paper.pdf",
        expect_answer="88.75% accuracy and 89.22% F1 score [cite: 516]",
    ),
    EvalCase(
        question="Why did the researchers opt for the Continuous Bag-of-Words (CBOW) model rather than Skip-gram for word vector representation?",
        expected_source=r"data\support paper.pdf",
        expect_answer="CBOW was chosen because it has fewer parameters, higher training efficiency, and stronger learning capabilities for rare word vectors [cite: 185]",
    ),

    # --- Negative control: should retrieve nothing from any ingested PDF ---
    EvalCase(
        question="What is the capital of France?",
        expected_source="",
    ),
]


# --------------------------------------------------------------------------
# 2. RETRIEVAL METRICS
# --------------------------------------------------------------------------

@dataclass
class CaseResult:
    question: str
    expected_source: str
    retrieved_sources: list[str] = field(default_factory=list)
    hit: bool = False
    rank_of_hit: int | None = None  # 1-indexed rank within filtered docs
    generation: str = ""
    faithfulness_verdict: str = ""
    latency_s: float = 0.0


def run_retrieval_only(case: EvalCase) -> CaseResult:
    """
    Executes the retrieval pipeline (reformulate -> retrieve -> grade) 
    and validates if the expected source is present in the filtered documents.
    """
    state: GraphState = {
        "messages": [],
        "raw_query": case.question,
        "search_query": "",
        "documents": [],
        "generation": "",
        "intent": "qa",
        "target_source": "",
    }
    config = {"configurable": {"relevance_threshold": 0.2}}

    t0 = time.time()
    state.update(reformulate_query_node(state, config))
    state.update(retrieve_node(state, config))
    state.update(grade_documents_node(state, config))
    latency = time.time() - t0

    docs = state.get("documents", [])
    sources = [d.metadata.get("source", "unknown") for d in docs]

    result = CaseResult(
        question=case.question,
        expected_source=case.expected_source,
        retrieved_sources=sources,
        latency_s=latency,
    )

    # Evaluate negative control cases where no documents should pass grading.
    if case.expected_source == "":
        result.hit = len(docs) == 0
        return result

    for i, s in enumerate(sources, start=1):
        if s == case.expected_source:
            result.hit = True
            result.rank_of_hit = i
            break

    return result


# --------------------------------------------------------------------------
# 3. FAITHFULNESS CHECK (LLM-as-a-judge)
# --------------------------------------------------------------------------

judge_prompt = ChatPromptTemplate.from_template(
    """
    You are grading whether an AI-generated answer is fully supported by
    the provided context. Respond with exactly one word: GROUNDED,
    UNGROUNDED, or REFUSAL.

    - GROUNDED: every factual claim in the answer is present in the context.
    - UNGROUNDED: the answer includes claims, facts, or details not found
      in the context (even if those claims happen to be true in general).
    - REFUSAL: the answer states it cannot find the information.

    Context:
    {context}

    Answer:
    {answer}

    Verdict (one word only):
    """
)
judge_chain = judge_prompt | big_llm | StrOutputParser()


def run_full_with_faithfulness(case: EvalCase, retrieval_result: CaseResult) -> CaseResult:
    """
    Executes generation using the graded documents and evaluates faithfulness. 
    Bypasses the evaluation if the grading stage returned no documents.
    """
    state: GraphState = {
        "messages": [],
        "raw_query": case.question,
        "search_query": "",
        "documents": [],
        "generation": "",
        "intent": "qa",
        "target_source": "",
    }
    config = {"configurable": {"relevance_threshold": 0.2}}
    state.update(reformulate_query_node(state, config))
    state.update(retrieve_node(state, config))
    state.update(grade_documents_node(state, config))

    if not state.get("documents"):
        retrieval_result.generation = "(no docs passed grading — generation skipped)"
        retrieval_result.faithfulness_verdict = "N/A"
        return retrieval_result

    state.update(generate_node(state, config))
    generation = state.get("generation", "")
    context = "\n\n".join(d.page_content for d in state["documents"])

    verdict = judge_chain.invoke({"context": context, "answer": generation}).strip()

    retrieval_result.generation = generation
    retrieval_result.faithfulness_verdict = verdict
    return retrieval_result


# --------------------------------------------------------------------------
# 4. REPORTING
# --------------------------------------------------------------------------

def compute_summary(results: list[CaseResult]) -> dict:
    n = len(results)
    hits = sum(r.hit for r in results)
    reciprocal_ranks = [
        (1.0 / r.rank_of_hit) for r in results if r.hit and r.rank_of_hit
    ]
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    grounded = sum(1 for r in results if r.faithfulness_verdict == "GROUNDED")
    ungrounded = sum(1 for r in results if r.faithfulness_verdict == "UNGROUNDED")
    judged = sum(1 for r in results if r.faithfulness_verdict in ("GROUNDED", "UNGROUNDED"))

    return {
        "n_cases": n,
        "hit_rate": hits / n if n else 0.0,
        "mrr": mrr,
        "faithfulness_rate": (grounded / judged) if judged else None,
        "ungrounded_count": ungrounded,
        "avg_latency_s": sum(r.latency_s for r in results) / n if n else 0.0,
    }


def write_report(results: list[CaseResult], summary: dict, out_path: Path):
    lines = ["# Retrieval & Faithfulness Eval Report", ""]
    lines.append(f"- Cases run: {summary['n_cases']}")
    lines.append(f"- Retrieval Hit Rate: {summary['hit_rate']:.1%}")
    lines.append(f"- Mean Reciprocal Rank (MRR): {summary['mrr']:.3f}")
    if summary["faithfulness_rate"] is not None:
        lines.append(f"- Faithfulness Rate (grounded / judged): {summary['faithfulness_rate']:.1%}")
        lines.append(f"- Ungrounded answers: {summary['ungrounded_count']}")
    lines.append(f"- Avg retrieval latency: {summary['avg_latency_s']:.2f}s")
    lines.append("")
    lines.append("## Per-case detail")
    lines.append("")
    lines.append("| Question | Expected | Hit | Rank | Faithfulness |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        q = r.question.replace("|", "\\|")
        lines.append(
            f"| {q} | {r.expected_source or '(none expected)'} | "
            f"{'✅' if r.hit else '❌'} | {r.rank_of_hit or '-'} | "
            f"{r.faithfulness_verdict or '-'} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(run_faithfulness: bool = True):
    print(f"Running {len(EVAL_SET)} eval cases...\n")
    results: list[CaseResult] = []

    for case in EVAL_SET:
        r = run_retrieval_only(case)
        if run_faithfulness:
            r = run_full_with_faithfulness(case, r)
        results.append(r)

        status = "HIT " if r.hit else "MISS"
        print(f"[{status}] rank={r.rank_of_hit} verdict={r.faithfulness_verdict or '-':<10} :: {case.question}")

    summary = compute_summary(results)
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    out_path = Path(__file__).resolve().parent / "eval_report.md"
    write_report(results, summary, out_path)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main(run_faithfulness=True)