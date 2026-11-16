"""Lesson 7 practice (starter): vector, BM25, hybrid and reranked search over the lesson 6 chunks.

Usage:
    python starter/search.py                  # evaluate all configurations on data/questions.json
    python starter/search.py "vpn backup"     # show the top results of every configuration for one query
    python starter/search.py --groups finance "operating costs"
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
QUESTIONS = ROOT / "data" / "questions.json"
OUT = ROOT / "output"
CACHE = OUT / "embeddings_cache.json"
MODELS = ROOT / ".models"   # downloaded once, ~300 MB

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, ~220 MB, uk + en
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"                                # ~80 MB cross-encoder
COLLECTION = "northwind"
K = 5                # how many results we evaluate (recall@K)
CANDIDATES = 20      # how many hybrid candidates go to the reranker
RRF_K = 60

_embedder = None
_reranker = None


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


def text_for_embedding(chunk: dict) -> str:
    """Contextual header: the section path in front of the text (lesson 6, chunking)."""
    return f"{chunk['section']}\n{chunk['text']}" if chunk["section"] else chunk["text"]


# ---------------------------------------------------------------- embeddings

# TODO 1-2 · Goal: turn every chunk into a vector, and do it only once.
#   Why: search compares vectors, not words; the model is slow, so unchanged chunks must come from the cache.
#   Done when: similar texts get closer vectors (check 1) and a second run prints "embedded now 0" (check 2).
def embed(texts: list[str]) -> list[list[float]]:
    """TODO 1: embed texts with EMBED_MODEL.
    TODO 2: cache vectors in output/embeddings_cache.json by sha256(model + text): a re-run embeds nothing."""
    # Hint: TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS)).embed(texts) yields numpy arrays; store them as lists of floats.
    # Set embed.last_missing = number of texts that were not in the cache.
    global _embedder

    # related to TODO 2: cache key = sha256(model + text). The model is part of the key because vectors of
    # different models are incompatible: change EMBED_MODEL and every lookup misses.
    keys = [hashlib.sha256(f"{EMBED_MODEL}\n{t}".encode("utf-8")).hexdigest() for t in texts]

    # related to TODO 2: read the cache from disk ({key: vector}); on the very first run there is no file yet.
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    # related to TODO 2: texts whose key is not in the cache; dict keeps the order and drops repeated texts.
    missing = {k: t for k, t in zip(keys, texts) if k not in cache}
    embed.last_missing = len(missing)

    if missing:
        # related to TODO 1: load the model once per process: the first call downloads it into .models/ and takes seconds,
        # later calls reuse the same object. With a warm cache the model is not loaded at all.
        if _embedder is None:
            _embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS))

        # related to TODO 1: model.embed() is a lazy generator of numpy arrays (384 floats each, same order as its input);
        # turn every array into a plain list of Python floats so it can be written to JSON.
        vectors = [[float(x) for x in vec] for vec in _embedder.embed(list(missing.values()))]

        # related to TODO 2: add only the new vectors to the cache and save it, so the next run finds them.
        cache.update(zip(missing.keys(), vectors))
        OUT.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")

    # related to TODO 2: return vectors in the order of the input texts, whether they came from the cache or were just embedded.
    return [cache[k] for k in keys]


embed.last_missing = 0


# ---------------------------------------------------------------- index

# TODO 3 · Goal: put all chunk vectors into Qdrant together with their metadata.
#   Why: the vector database finds the nearest vectors fast and returns the chunk (source, section, access) with them.
#   Done when: the collection has 69 points and each point carries its chunk as payload (check 3).
def build_index(chunks: list[dict]) -> QdrantClient:
    """TODO 3: create a Qdrant collection (cosine) and upsert one point per chunk with the chunk as payload."""
    # Hint: QdrantClient(":memory:"), create_collection(..., VectorParams(size, Distance.COSINE)), upsert(PointStruct(...))
    # related to TODO 3: embed every chunk with its section path in front (contextual header from lesson 6);
    # thanks to the cache (TODO 2) only new or changed chunks hit the model.
    vectors = embed([text_for_embedding(c) for c in chunks])

    # related to TODO 3: in-memory Qdrant: no server, the index lives only while the script runs.
    client = QdrantClient(":memory:")

    # related to TODO 3: the collection must know the vector length (384 for this model) and the metric;
    # cosine compares directions, so it does not matter that this model's vectors are not normalized.
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )

    # related to TODO 3: one point per chunk: id = position in the list, vector = its embedding,
    # payload = the whole chunk (text, source, section, access), so a search hit carries everything we need to show it.
    client.upsert(
        COLLECTION,
        points=[models.PointStruct(id=i, vector=vec, payload=chunk)
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))],
    )
    return client


# TODO 8 · Goal: a user only gets chunks of the groups they belong to.
#   Why: access rules from lesson 6 must work inside the search; a prompt "do not show finance data" is not protection.
#   Done when: the Q3 budget is hidden from all_employees but found for finance (check 8, "access filter: OK").
def access_filter(groups: list[str]) -> models.Filter:
    """TODO 8: only chunks whose `access` shares at least one group with the user."""
    return None   # no filter yet: every user sees every chunk


def allowed(chunk: dict, groups: list[str]) -> bool:
    return bool(set(chunk["access"]) & set(groups))


# ---------------------------------------------------------------- search

# TODO 4 · Goal: semantic search: the query becomes a vector, Qdrant returns the k nearest chunks.
#   Why: it finds the answer even when the user's words differ from the document (another language included).
#   Done when: "How many days a week can I work from home?" finds the remote work rule (check 4).
def vector_search(client: QdrantClient, query: str, groups: list[str], k: int = K) -> list[dict]:
    """TODO 4: embed the query with the same model and return the top-k chunks (with "score")."""
    # Hint: client.query_points(COLLECTION, query=<vector>, limit=k, query_filter=access_filter(groups)).points
    # related to TODO 4: embed the query with the same embed() as the chunks, so both live in one vector space;
    # a vector from another model would be compared with chunk vectors meaninglessly.
    query_vector = embed([query])[0]

    # related to TODO 4: Qdrant returns the k points closest to the query by cosine;
    # access_filter(groups) runs inside the search, so chunks the user may not read never come back (TODO 8).
    points = client.query_points(
        COLLECTION, query=query_vector, limit=k, query_filter=access_filter(groups),
    ).points

    # related to TODO 4: a hit = the chunk from the payload plus its similarity "score" (higher = closer);
    # the score is not a probability of a right answer, 0.4-0.6 is normal for this model.
    return [{**p.payload, "score": p.score} for p in points]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# TODO 6 · Goal: classic keyword search (BM25) over the same chunks.
#   Why: embeddings miss exact codes and names like "BV-2013-02"; BM25 finds them at once.
#   Done when: "agreement BV-2013-02" returns the manual chunk 6.3 first (check 6).
class BM25Index:
    def __init__(self, chunks: list[dict]):
        """TODO 6: build BM25 over the tokenized texts (section path included, like for embeddings)."""
        self.chunks = chunks
        self.bm25 = None

    def search(self, query: str, groups: list[str], k: int = K) -> list[dict]:
        # Hint: self.bm25.get_scores(tokenize(query)); skip zero scores and chunks the user may not read (allowed())
        return []


# TODO 7 · Goal: merge the vector and BM25 rankings into one list.
#   Why: their scores are on different scales (0.36 vs 13.0), so we combine ranks, not scores.
#   Done when: a chunk found by both searches beats one found by a single search (check 7),
#   and hybrid recall@5 is higher than vector-only (check "TODO 6-7").
def rrf(rankings: list[list[dict]], k: int = K) -> list[dict]:
    """TODO 7: Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank), rank starts at 1."""
    return rankings[0][:k] if rankings else []   # for now: just the first ranking


# TODO 9 · Goal: re-order the top-20 hybrid candidates with a cross-encoder.
#   Why: it reads the query and the chunk together, so the right chunk moves to the top (higher MRR).
#   Done when: the right chunk goes from last to first place (check 9). Then compare MRR and ms/query in the table.
def rerank(query: str, candidates: list[dict], k: int = K) -> list[dict]:
    """TODO 9: score every (query, chunk) pair with the cross-encoder and keep the top-k."""
    # Hint: TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODELS)).rerank(query, texts) yields one score per text
    return candidates[:k]   # for now: no reranking


def make_configs(client: QdrantClient, bm25: BM25Index) -> dict:
    return {
        "vector": lambda q, g: vector_search(client, q, g),
        "bm25": lambda q, g: bm25.search(q, g),
        "hybrid": lambda q, g: rrf([vector_search(client, q, g, CANDIDATES), bm25.search(q, g, CANDIDATES)]),
        "hybrid+rerank": lambda q, g: rerank(q, rrf([vector_search(client, q, g, CANDIDATES),
                                                     bm25.search(q, g, CANDIDATES)], CANDIDATES)),
    }


# ---------------------------------------------------------------- evaluation

# TODO 5 · Goal: measure the search with numbers instead of "looks fine".
#   Why: without recall@k and MRR we cannot tell whether hybrid or the reranker actually helps.
#   Done when: the fake test gives recall 0.67 and MRR 0.50 (check 5) and the table shows real numbers per config.
def evaluate(search, questions: list[dict], k: int = K) -> dict:
    """TODO 5: recall@k (a relevant chunk is in the top-k) and MRR (1 / rank of the first relevant chunk)."""
    hits, reciprocal_ranks, misses, elapsed = 0, [], [], 0.0
    for q in questions:
        # related to TODO 5: run the search and time it; only the top-k chunk ids matter for the metrics.
        start = time.perf_counter()
        top = [h["chunk_id"] for h in search(q["question"], q["groups"])][:k]
        elapsed += time.perf_counter() - start

        # related to TODO 5: 1-based position of the first relevant chunk in the top-k, None if there is none.
        rank = next((i for i, cid in enumerate(top, start=1) if cid in q["relevant"]), None)

        # related to TODO 5: recall counts a question as found; MRR adds 1/rank (1st place = 1, 2nd = 0.5, ...),
        # a miss adds 0 to MRR and its id goes to "misses" so we can see which questions fail.
        if rank is None:
            reciprocal_ranks.append(0.0)
            misses.append(q["id"])
        else:
            hits += 1
            reciprocal_ranks.append(1 / rank)

    # related to TODO 5: average everything over the questions; "ms" is the mean search time per query.
    n = len(questions)
    return {"recall": hits / n, "mrr": sum(reciprocal_ranks) / n, "misses": misses, "ms": elapsed / n * 1000}


def check_access(client: QdrantClient, bm25: BM25Index) -> bool:
    """A user without the finance group must never get the Q3 budget, whatever they ask."""
    q, groups = "What were the total operating costs in Q3?", ["all_employees"]
    hits = vector_search(client, q, groups) + bm25.search(q, groups)
    return all("finance" not in h["access"] for h in hits)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


# TODO 10 (optional) · Goal: find chunks from different documents that say almost the same.
#   Why: embeddings catch duplicates in other words, which the shingles of lesson 6 miss.
#   Done when: --pairs prints 5 pairs with cosine between -1 and 1 (check 10).
def similar_pairs(chunks: list[dict], top: int = 5) -> list[tuple]:
    """TODO 10 (optional): the most similar chunk pairs from different documents (semantic near-duplicates).
    This model does not normalize its vectors, so a plain dot product is not a cosine."""
    return []


# ---------------------------------------------------------------- main

def show(configs: dict, query: str, groups: list[str]):
    for name, search in configs.items():
        print(f"\n== {name}")
        for h in search(query, groups):
            print(f"  {h['score']:8.4f}  {h['chunk_id']:<50} {h['section'][:45]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--groups", default="all_employees,warehouse")
    parser.add_argument("--pairs", action="store_true", help="show semantic near-duplicates (TODO 10)")
    args = parser.parse_args()

    chunks = load_chunks()
    client = build_index(chunks)
    bm25 = BM25Index(chunks)
    configs = make_configs(client, bm25)
    print(f"chunks {len(chunks)} | embedded now {embed.last_missing} (rest from cache) | model {EMBED_MODEL}")

    if args.pairs:
        for score, a, b in similar_pairs(chunks):
            print(f"  {score:.3f}  {a}  ~  {b}")
        return
    if args.query:
        show(configs, args.query, args.groups.split(","))
        return

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"\n{'config':<15}{'recall@' + str(K):>10}{'MRR':>8}{'ms/query':>10}   misses")
    results = {}
    for name, search in configs.items():
        r = evaluate(search, questions)
        results[name] = r
        print(f"{name:<15}{r['recall']:>10.2f}{r['mrr']:>8.2f}{r['ms']:>10.0f}   {' '.join(r['misses']) or '-'}")
    access_ok = check_access(client, bm25)
    print("\naccess filter: " + ("OK, finance chunk hidden from all_employees" if access_ok
                                   else "LEAK, all_employees can see the finance chunk (TODO 8)"))
    line = " | ".join(f"{n} {r['recall']:.2f}/{r['mrr']:.2f}" for n, r in results.items())
    print(f"\nFOR CHAT: recall@{K}/MRR on {len(questions)} questions | {line}")
    (OUT / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as e:
        sys.exit(f"Not implemented yet: {e}")
