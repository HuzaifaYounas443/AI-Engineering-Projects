"""Step 9b: Full RAG — retrieval + DeepSeek generation."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion import Embedder, QdrantIndexer

load_dotenv(Path(__file__).parent / ".env")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    embedder = Embedder()
    vector = embedder.embed([question])[0]

    indexer = QdrantIndexer()
    response = indexer.client.query_points(
        collection_name="website_knowledge",
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": r.payload["text"], "source": r.payload["source"], "score": r.score}
        for r in response.points
    ]


def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    system = (
        "You are a helpful website support assistant. "
        "Answer the user's question using ONLY the context below. "
        "If the answer isn't in the context, say you don't have that information. "
        "Always cite the source file."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    client = get_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    question = "How do I reset my password?"

    print(f"Question: {question}\n")
    print("Retrieving ...")
    chunks = retrieve(question, top_k=3)
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['source']} (score: {c['score']:.4f})")

    print("\nGenerating with Groq ...\n")
    answer = generate(question, chunks)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)