# Chapter 7: Search In-Depth

## Topic 7: Reranking — Cross-Encoders (Cohere Rerank, BGE-reranker)

---

### 1. Concept, Intuition, and Why It Exists

**The fundamental limitation every prior topic in this chapter has been working around:**

- BM25 (Topic 1), dense retrieval (Topic 2), and hybrid RRF (Topic 4) are all **bi-encoder** approaches — query and document are encoded *independently* into vectors, and similarity is computed afterward via a simple operation (dot product, BM25 score sum)
- This independence is exactly what makes them fast enough to search a full corpus: document vectors are precomputed once at ingest time, and only the query needs encoding at search time
- The cost of this independence: the model never actually sees the query and document *together*. It cannot model fine-grained interactions between specific words in the query and specific words in the document — it can only compare two fixed summaries that were computed without knowledge of each other
- A **cross-encoder** removes this limitation by taking the query and a candidate document as a single joint input, letting the model's attention mechanism directly compare every query token against every document token — producing a single, highly accurate relevance score
- The cost: a cross-encoder cannot precompute anything. Every (query, document) pair requires a full forward pass through the model at query time. This is far too slow to run against an entire corpus — which is exactly why reranking is always a **second stage**, applied only to a small candidate pool a cheaper method has already narrowed down

**Why this is the natural next step after Topics 1–6:**

- Topics 1–4 built increasingly good candidate retrieval (sparse, dense, hybrid)
- Topic 6 (MMR) re-ordered that candidate pool for diversity
- This topic adds the missing piece: a much more *accurate* relevance judgment, applied to a short list, as the final quality gate before chunks reach the generation layer (Chapter 8)
- The architecture pattern across this entire chapter has been: cheap-and-broad first, expensive-and-precise last, only on what's already been narrowed down — reranking is the purest expression of that principle

---

### 2. Internal Working — Step by Step

**Bi-encoder scoring (what you already have):**

```text
query  -> Encoder -> vector_q   (computed once per query)
doc    -> Encoder -> vector_d   (computed once per document, at ingest time)
score  = similarity(vector_q, vector_d)   (cheap: one dot product)
```

**Cross-encoder scoring (this topic):**

```text
[query, doc] -> concatenated as one input -> Encoder -> single relevance score

Example input to the model:
  "[CLS] premature withdrawal penalty FD [SEP] Premature withdrawal of FD
   incurs a 1 percent penalty on the applicable interest rate. [SEP]"

Output: a single scalar (often passed through a sigmoid), e.g. 0.94
```

**Step-by-step reranking pipeline:**

1. Run cheap retrieval first (BM25 + dense + RRF from Topic 4) to get a candidate pool — typically top-20 to top-50
2. Optionally apply MMR (Topic 6) for diversity within that pool, or apply MMR *after* reranking (see Section 5 — order matters)
3. For each candidate document in the pool, construct the `[query, document]` pair
4. Pass every pair through the cross-encoder model, one batch at a time
5. Each pair receives its own independent relevance score — unlike bi-encoder scores, cross-encoder scores are directly comparable across different documents for the *same* query (though not necessarily across different queries)
6. Sort candidates by cross-encoder score, descending
7. Return the top-k (typically top-3 to top-5) as the final result set passed to generation

**Why cross-encoder scores are more accurate — the mechanism:**

- Self-attention inside the model can directly compute "how well does the word 'penalty' in the query align with the word 'penalty' in this specific document, in this specific surrounding context?"
- A bi-encoder's single fixed vector for the document was computed without ever knowing what the query would be — it's a general-purpose summary, not a query-aware one
- This is the same fundamental trade-off discussed for ColBERT in Topic 3 (late interaction vs no interaction) — a cross-encoder takes this to its logical extreme: *full* interaction, every token attending to every other token, at the cost of needing to redo this computation for every single candidate on every single query

---

### 3. How It Is Implemented in This Project

**Where reranking fits in the pipeline built across this chapter:**

1. Customer email arrives (Hinglish or English, ~31 words average)
2. Hybrid retrieval (Topic 4: BM25 + Dense + RRF) → top-20 candidate pool
3. Cross-encoder reranking (this topic) → re-scores all 20, returns top-5 by cross-encoder relevance
4. MMR (Topic 6) → optionally applied on the reranked top-5 to ensure diversity within the final set actually passed to generation
5. Final chunks handed to Chapter 8's generation layer

**Two practical model choices for this project:**

- **BGE-reranker** (`BAAI/bge-reranker-v2-m3` or `BAAI/bge-reranker-base`): open-source, runs locally on the RTX 4060, multilingual (relevant for the 64.4% Hinglish corpus), no per-query API cost
- **Cohere Rerank** (`rerank-multilingual-v3.0`): managed API, strong multilingual reranking quality, no local GPU needed, but introduces a per-query external API cost and network latency, and sends customer email content to a third-party API — a genuine consideration given the PII concerns already raised in Chapter 6 Topic 9

**Code pattern — local BGE-reranker:**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

# Pairs: [query, candidate_text] for every candidate in the pool
pairs = [[query, candidate["text"]] for candidate in candidate_pool]

# One batched forward pass -- much more efficient than one call per pair
scores = reranker.predict(pairs)

# Attach scores back and sort
for candidate, score in zip(candidate_pool, scores):
    candidate["rerank_score"] = float(score)

reranked = sorted(candidate_pool, key=lambda c: c["rerank_score"], reverse=True)
top_k = reranked[:5]
```

**Code pattern — Cohere Rerank API (shown for completeness, not run in this project's offline pipeline):**

```python
import cohere

co = cohere.Client(api_key="...")

response = co.rerank(
    model="rerank-multilingual-v3.0",
    query=query,
    documents=[c["text"] for c in candidate_pool],
    top_n=5,
)

# response.results is already sorted by relevance, with .relevance_score per result
```

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

**Latency — the central trade-off of this entire topic:**

- BM25 + dense + RRF for this project's 17-page corpus: well under 5ms total (Topics 1, 2, 4 measurements)
- Cross-encoder reranking of 20 candidates: each pair is a full transformer forward pass — on a local BGE-reranker-base on CPU, expect roughly 20-80ms per pair depending on hardware, or a few milliseconds per pair when batched on the RTX 4060 GPU
- Batching all 20 pairs into a single `predict()` call (as shown above) is essential — one call per pair would multiply latency by roughly the batch size
- At production volume (8,000-12,000 emails/day, ~93/minute), reranking 20 candidates per email on GPU is comfortably within latency budget; on CPU alone it becomes a meaningful fraction of total response time and should be measured explicitly, not assumed acceptable

**Cost:**

- Local BGE-reranker: zero marginal cost per query beyond compute already available on the RTX 4060, same cost profile as the dense embedding model from Topic 2
- Cohere Rerank API: priced per search unit (per query, per up-to-N documents reranked) — at production volume this is a real, ongoing operational cost that scales linearly with email volume, unlike the local model's fixed infrastructure cost
- This is a genuine build-vs-buy decision, not a default choice — Section 6 covers this trade-off explicitly

**Security and PII (direct continuation of Chapter 6 Topic 9):**

- Using Cohere's hosted Rerank API means the customer's email text and the candidate policy chunks are transmitted to a third-party service on every query — this has real data residency and compliance implications for an NBFC handling financial customer data, and should be evaluated against the same access-scoping principles established in Chapter 6
- A local BGE-reranker keeps all data on your own infrastructure, avoiding this exposure entirely — for a regulated financial domain, this is a strong argument in favor of the local option unless the managed API's quality advantage is measured to be worth the compliance review

**Debugging a bad reranking result:**

- If the cross-encoder consistently reorders results in a way that looks wrong, check the raw score distribution first — cross-encoder scores from different model checkpoints are not universally calibrated (a score of 0.7 from one model isn't necessarily "70% confident" in a meaningful absolute sense), so compare *relative* rankings, not absolute score thresholds, unless the specific model's calibration has been separately verified
- If reranking seems to add no value over the Topic 4 hybrid ranking, verify the candidate pool actually contains meaningfully different orderings to begin with — reranking cannot improve on a candidate pool where the top-20 by BM25+dense already perfectly matches the true relevance order; its value is proportional to how often the cheap retrieval stage's ranking is imperfect

**Monitoring:**

- Track how often the reranker's top-1 differs from the pre-rerank (hybrid RRF) top-1 — this quantifies how much value reranking is actually adding in production, directly comparable to the "agreement rate" monitoring pattern established in Topic 4 for BM25 vs dense
- Track reranking latency percentiles (p50, p95, p99) separately from retrieval latency — reranking is the new, most expensive step in the pipeline and deserves its own SLA tracking
- If using Cohere's API, track API error rates and timeouts as a distinct failure mode requiring graceful degradation (falling back to the pre-rerank ranking if the API call fails)

**Scaling:**

- Reranking cost scales with candidate pool size, not corpus size — this is the entire point of the two-stage design. Growing the underlying 17-page knowledge base to 17,000 pages does not change reranking cost, because reranking only ever touches the fixed-size candidate pool (e.g. top-20) that the cheap retrieval stage already narrowed down
- If the candidate pool size itself needs to grow (e.g. widening from top-20 to top-50 for better recall), reranking cost grows linearly with that — a deliberate, measurable trade-off, not an accidental one

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

**How large should the candidate pool be before reranking?**

- Too small (e.g. top-5 into reranking): if the true best document was ranked 7th by hybrid retrieval, reranking never sees it — reranking cannot recover recall the earlier stage already lost
- Too large (e.g. top-200 into reranking): reranking latency and cost grow proportionally for diminishing returns, since documents ranked very low by the cheap stage are unlikely to be truly relevant
- Common practical range: top-20 to top-50 into reranking, tuned against measured Recall@K (Topic 9) rather than picked arbitrarily

**Reranking before or after MMR (Topic 6)?**

- As already flagged in Topic 6's design decisions: reranking should generally run *before* MMR, so that MMR's relevance term (`Sim(d, Query)`) uses the more accurate cross-encoder score rather than the cruder bi-encoder or RRF-fused score
- Running MMR first means the diversity-aware selection is working from a less accurate relevance signal, potentially discarding a genuinely more relevant document in favor of a merely-different one before the more accurate judge ever gets to weigh in

**Local model vs managed API — the build-vs-buy decision:**

- Local (BGE-reranker): no per-query cost, no data leaves your infrastructure, requires GPU capacity planning and model maintenance (Chapter 6 Topic 8's embedding model migration concerns apply equally here — a reranker model version change also requires re-validation)
- Managed API (Cohere Rerank): typically higher out-of-the-box quality on general benchmarks, zero infrastructure to maintain, but ongoing per-query cost and the PII/data-residency concern raised in Section 4
- For this project specifically — an NBFC handling financial customer data with an already-established local-model-first philosophy (Chapter 3's embedding model choice) — the local BGE-reranker is the more consistent choice unless a measured quality gap justifies the compliance review needed to adopt a third-party API

**Is reranking always worth the added latency?**

- Not unconditionally — this is the same evidence-before-complexity principle applied throughout this chapter (Topics 1, 4, 6 all flagged this)
- The correct test: measure Recall@K and NDCG@K (Topic 9) with and without the reranking stage on a labeled evaluation set. If the improvement is marginal relative to the added latency and cost, reranking may not be justified for this specific corpus and query distribution — particularly relevant for a small, 17-page knowledge base where the candidate pool from hybrid retrieval may already be quite accurate

---

### 6. Alternatives and When to Use Each

**No reranking (hybrid RRF output used directly, Topic 4 as-is):**
- Best for: small, low-redundancy corpora where hybrid retrieval's ranking is already measured to be highly accurate; latency-critical applications where the extra reranking hop cannot be afforded
- Use when: Topic 9's evaluation shows reranking's Recall@K/NDCG@K improvement is not meaningfully better than hybrid alone

**Local cross-encoder reranking (BGE-reranker, this topic's primary recommendation):**
- Best for: this project's actual constraints — data residency concerns for financial customer data, existing local-model infrastructure (RTX 4060) already used for embeddings, need for multilingual (Hinglish) support
- Use when: quality improvement over hybrid-only is confirmed valuable and local GPU capacity is available

**Managed reranking API (Cohere Rerank):**
- Best for: teams without GPU infrastructure, or where absolute best-in-class reranking quality matters more than per-query cost and data residency constraints
- Use when: the compliance review for sending customer data to a third party has been completed and approved, and the API's latency profile fits the SLA

**ColBERT-style late interaction (Topic 3) as a middle ground:**
- Sits between bi-encoder speed and cross-encoder accuracy — token-level MaxSim scoring without the full joint-attention cost of a true cross-encoder
- Use when: cross-encoder reranking latency is measured to be a genuine bottleneck, but pure bi-encoder accuracy is insufficient

---

### 7. Common Mistakes and Production Failures

- **Running the cross-encoder one pair at a time instead of batching** — turns a fast, GPU-efficient operation into a slow, serialized one; always batch all candidate pairs into a single `predict()` call
- **Applying reranking to the entire corpus instead of a pre-narrowed candidate pool** — defeats the entire two-stage design; cross-encoders are computationally infeasible at full-corpus scale, which is exactly why Topics 1-4 exist first
- **Treating cross-encoder scores as calibrated absolute probabilities across different queries** — a score of 0.6 for one query's top result and 0.6 for a different query's top result do not necessarily mean the same thing; use scores for *relative* ranking within a single query's candidate pool, not for cross-query comparisons or fixed absolute thresholds without separate calibration work
- **Skipping reranking's contribution to the evaluation harness** — deploying reranking without measuring its actual Recall@K/NDCG@K improvement (Topic 9) means flying blind on whether the added latency and cost are justified
- **Sending PII-containing customer emails to a third-party reranking API without a compliance review** — a direct violation of the access-scoping principles established in Chapter 6 Topic 9, easy to overlook when integrating a new external API
- **Not falling back gracefully when a managed reranking API times out or errors** — the pipeline should degrade to the pre-rerank (hybrid RRF) ranking rather than failing the entire request when an external API call fails

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: What is the fundamental architectural difference between a bi-encoder and a cross-encoder, and why does it make cross-encoders unsuitable for first-stage retrieval?**

A: A bi-encoder encodes query and document independently into separate vectors, computed without knowledge of each other, then compares them via a cheap operation like a dot product — document vectors can be precomputed once and reused for every future query. A cross-encoder takes query and document together as a single joint input, letting the model's attention directly compare every token pair, producing a much more accurate relevance score — but this requires a full model forward pass for every single (query, document) pair, with nothing precomputable. Running a cross-encoder against an entire corpus for every query would require re-running the model N times per query, where N is the corpus size — computationally infeasible at scale, which is why cross-encoders are always applied as a second-stage reranker over a small candidate pool a cheaper method has already narrowed down.

**Q: Why must candidate pairs be batched into a single call to the cross-encoder rather than scored one at a time?**

A: Modern transformer inference is significantly more efficient when processing multiple inputs in parallel as a batch, especially on GPU hardware, due to how matrix operations parallelize across the batch dimension. Scoring pairs one at a time forfeits this parallelism, multiplying wall-clock latency by roughly the number of candidates compared to a single batched call.

**Intermediate:**

**Q: You have a hybrid RRF pipeline (Topic 4) already producing reasonable results. How would you decide whether adding a reranking stage is actually worth the additional latency and cost?**

A: I would build a labeled evaluation set of (query, correct/relevant document) pairs covering the actual query distribution, then measure Recall@K and NDCG@K (Topic 9) both with and without the reranking stage applied to the hybrid RRF candidate pool. If reranking produces a measurable, meaningful improvement in these metrics — not just a different ordering, but placing genuinely more relevant documents higher — that improvement is weighed against the measured latency and cost increase. For a small corpus like this project's 17 pages, hybrid retrieval alone may already be quite accurate, making reranking's marginal value smaller than it would be for a much larger, noisier corpus; the decision should be evidence-based, not assumed.

**Q: For an NBFC handling customer financial data, what specific risk does using a managed reranking API like Cohere Rerank introduce that a local BGE-reranker does not?**

A: Using a managed API means transmitting the customer's email content and candidate policy document text to a third-party service on every query. This raises the same data residency and access-control concerns established in Chapter 6 Topic 9 for PII in the vector store — except now the exposure extends beyond your own infrastructure to an external vendor. This requires a compliance review (data processing agreements, regional data residency requirements for financial data, audit trail requirements) before adoption, whereas a locally-run BGE-reranker keeps all data within infrastructure you already control, avoiding this review entirely.

**Advanced:**

**Q: Design the full candidate-pool-size and reranking strategy for this project, explaining your reasoning at each stage.**

A: Start with hybrid RRF (Topic 4) retrieving a top-20 candidate pool from the 17-page knowledge base — wide enough that Recall@20 is measured to be very high (the true relevant chunk is almost always somewhere in this pool), narrow enough that reranking 20 candidates has negligible latency cost on the RTX 4060. Apply the BGE-reranker (chosen over Cohere for data residency reasons specific to financial customer data) to re-score all 20 candidates in a single batched call. Take the reranked top-5. Optionally apply MMR (Topic 6) on this reranked top-5 if a diversity check against the corpus's known redundancy pattern (the same fact restated across FAQ/Policy/SOP) shows it's still needed after reranking — reranking's more accurate scoring may already reduce redundancy incidentally, since a cross-encoder is better at recognizing when two candidates say essentially the same thing relative to the query. Validate the entire pipeline's Recall@K and NDCG@K (Topic 9) against a labeled evaluation set before and after adding each stage, to confirm each addition is earning its complexity cost.

**Q: A teammate proposes using an LLM (e.g. asking Claude "which of these 5 documents is most relevant to this query?") as the reranker instead of a dedicated cross-encoder model. Evaluate this proposal.**

A: This is a legitimate and increasingly common pattern — "LLM-as-reranker" — but it has different trade-offs than a dedicated cross-encoder. A dedicated cross-encoder (BGE-reranker) is purpose-trained specifically for relevance scoring, typically faster and cheaper per candidate than an LLM API call, and produces a genuine numeric score suitable for sorting. An LLM-based reranker can potentially reason more flexibly about nuanced relevance (e.g. understanding that a document technically contains the right keywords but doesn't actually answer the question), but at meaningfully higher latency and cost per candidate, and with output that may need careful prompt engineering to produce consistently sortable scores rather than free-text judgments. For this project's latency and cost constraints (production volume of 8,000-12,000 emails/day), a dedicated cross-encoder is the more practical default; LLM-as-reranker becomes worth considering specifically for the hardest, most ambiguous cases that a dedicated reranker still gets wrong — which connects directly to this project's broader cascade design philosophy (cheap methods first, GenAI reserved for the genuinely hard tail).

**Scenario-based:**

**Q: After deploying BGE-reranker in production, you notice p95 latency for the reranking stage has grown significantly over the past month, even though email volume has stayed flat. Walk through your diagnosis.**

A: First check whether the candidate pool size fed into reranking has changed — if the upstream hybrid retrieval stage (Topic 4) is retrieving more candidates than before (e.g. due to a configuration change or the knowledge base growing), reranking latency grows proportionally since it must score every candidate in the pool. Second, check whether the reranker model itself changed versions without a corresponding infrastructure review — a larger or different checkpoint could have a meaningfully different latency profile per pair. Third, check GPU utilization and contention — if other processes are now sharing the RTX 4060, or if batch sizes have shrunk for some reason, per-batch latency would increase even with the same total candidate volume. Fourth, confirm batching is still happening correctly in code — a regression that accidentally reverted to one-at-a-time scoring would produce exactly this kind of latency growth with flat volume.

---

### 9. Hidden Concepts and Prerequisites

**Cross-encoder score calibration is a separate problem from ranking accuracy:**

- A cross-encoder can produce a *correct relative ranking* (document A is genuinely more relevant than document B) while producing scores that are poorly calibrated in an absolute sense (the numeric gap between A's and B's scores may not reflect a meaningful, interpretable confidence difference)
- This matters if the pipeline ever needs an absolute relevance threshold (e.g. "only pass chunks to generation if reranker score > 0.5") rather than just a relative top-k selection — calibration would need to be separately verified or trained for, which most off-the-shelf rerankers do not guarantee out of the box

**Knowledge distillation connects rerankers back to Topic 3:**

- Many production cross-encoder rerankers, and the bi-encoders they're often used to train (mentioned in Topic 3's SPLADE v2 discussion), use knowledge distillation — a cross-encoder's more accurate relevance judgments are used as training signal to improve a faster bi-encoder or sparse model
- This creates a virtuous cycle: cross-encoders are too slow for first-stage retrieval, but their judgments can make first-stage retrievers better over time, reducing how much reranking needs to correct for at inference time

**The reranking stage is where "search" and "ranking" as separate ML disciplines most directly meet:**

- Classical information retrieval (BM25, TF-IDF) and modern learning-to-rank (LTR) research are historically somewhat separate fields; reranking with a trained cross-encoder is where LTR techniques most directly enter a RAG pipeline
- Worth knowing that LTR as a discipline has its own vocabulary (pointwise, pairwise, and listwise ranking loss functions) — a cross-encoder trained with a pointwise loss (score each document independently) behaves differently from one trained with a pairwise or listwise loss (explicitly trained to get relative orderings right) — useful context when evaluating which pretrained reranker checkpoint to adopt

---

### 10. Revision Summary

> Reranking applies a cross-encoder — a model that takes query and document as a single joint input, rather than separately encoding them like a bi-encoder — to re-score a small candidate pool (typically top-20 to top-50) that cheaper retrieval methods (BM25, dense, hybrid RRF from Topics 1-4) have already narrowed down from the full corpus. Cross-encoders are far more accurate because attention can directly compare query and document tokens, but far too slow to run against a full corpus, making them strictly a second-stage step. For this project, the local BGE-reranker is the more consistent choice over Cohere's managed API given the financial domain's PII and data-residency concerns already established in Chapter 6, and given the project's existing local-model-first infrastructure. Reranking should generally run before MMR (Topic 6) so MMR's diversity correction uses the most accurate relevance signal available. As with every other technique in this chapter, reranking's value must be measured — via Recall@K and NDCG@K (Topic 9) — rather than assumed, since its added latency and cost are only justified when the underlying candidate pool's cheap ranking is measurably imperfect.

---
