"""Full hybrid RAG: dense + sparse retrieval + Groq generation."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval import HybridRetriever

load_dotenv(Path(__file__).parent / ".env")


# --- LLM client ---

def get_client() -> OpenAI:
    """Groq exposes an OpenAI-compatible endpoint."""
    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


MODEL = "openai/gpt-oss-20b"  # current free-tier model on Groq


# --- Generation ---

SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't have that information. "
    "Cite the source file for every fact you state. Be concise."
)


def generate(question: str, chunks: list[dict]) -> str:
    """Send retrieved chunks + question to Groq. Returns the answer."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content


# --- Main ---

def main(question: str | None = None) -> None:
    """Run the full hybrid RAG pipeline on a question."""
    if question is None:
        question = "How do I reset my password?"

    print(f"\nQuestion: {question}\n")

    # 1. Retrieve via hybrid search
    print("Loading hybrid retriever ...")
    retriever = HybridRetriever()

    print("Retrieving top candidates ...")
    results, sources = retriever.search(
        question,
        top_k=3,
        per_retriever_k=20,
        return_sources=True,
    )

    print(f"  Dense candidates : {len(sources['dense'])}")
    print(f"  Sparse candidates: {len(sources['sparse'])}")
    print(f"  Fused top-3      :")
    for i, r in enumerate(results, 1):
        print(f"    [{i}] rrf={r['rrf_score']:.5f}  {r['source']}")

    # 2. Generate with Groq
    print(f"\nGenerating with {MODEL} ...\n")
    answer = generate(question, results)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()