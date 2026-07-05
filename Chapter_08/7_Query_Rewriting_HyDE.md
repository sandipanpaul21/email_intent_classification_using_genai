# Chapter 8: RAG Generation — The Answer Layer

## Topic 7: Query Rewriting and HyDE (Hypothetical Document Embeddings)

---

### 1. Concept, Intuition, and Why It Exists

- Chapter 7's entire retrieval pipeline (BM25, dense, hybrid, reranking, MMR) operates on whatever query it receives — but the raw customer query itself is often a poor retrieval input, independent of the multi-turn contextualization problem covered in Topic 6. This topic addresses **single-turn** query quality problems: a self-contained but poorly-phrased, vague, or vocabulary-mismatched query, exactly the kind already measured as a real weakness in this project's corpus (Chapter 7 Topic 1's Hinglish vocabulary mismatch findings, Topic 2's 0.0393 discrimination gap).
- **Query rewriting** (general term): transforming the original query into a better retrieval input before it hits the retrieval pipeline — could mean expanding it with synonyms, correcting phrasing, or restructuring it as a clearer question.
- **HyDE (Hypothetical Document Embeddings)**: a specific, clever query rewriting technique — instead of embedding the query directly for dense retrieval, ask an LLM to generate a *hypothetical answer* to the query (even though it might be wrong or incomplete), and embed *that hypothetical answer* instead of the original query. The intuition: a hypothetical answer, even an imperfect one, is written in the same style and vocabulary as real answer documents in the knowledge base — making its embedding a better match for genuinely relevant documents than the original, often terser and differently-worded, query.
- Why this matters specifically for this project: your emails average ~31 words, are 64.4% Hinglish, and often phrase things very differently from how the formal English policy documents phrase the same information ("FD band karna hai" vs. "premature closure is allowed subject to applicable penalty charges") — exactly the vocabulary and register mismatch HyDE is designed to bridge.

---

### 2. Internal Working — Step by Step

**Query rewriting (general techniques):**

1. **Spelling/grammar normalization**: correcting obvious typos or informal phrasing before retrieval — a lightweight, often rule-based or small-model step.
2. **Query expansion**: adding synonyms or related terms to the query — directly connects to Chapter 7 Topic 1's discussion of Hinglish-to-English term mapping (byaj→interest, machurity→maturity) as a fix for BM25's vocabulary mismatch; this topic formalizes that as a standard, general technique rather than a corpus-specific patch.
3. **Decomposition**: breaking a complex, multi-part query into several simpler sub-queries, each retrieved independently, then combined — useful for compound customer questions ("what's the penalty AND how long does the refund take").

**HyDE, step by step:**

1. Take the original query — e.g. "FD band karna hai penalty kitni hogi"
2. Prompt an LLM (this project's `claude-haiku-4-5-20251001` is a reasonable, fast choice) to generate a plausible, hypothetical answer to this query — *without* access to the actual knowledge base, purely from the model's general knowledge or even just plausible-sounding domain language: "Premature closure of a Fixed Deposit typically incurs a penalty of around 1% on the applicable interest rate, and the account is closed within a few working days of the request."
3. Embed this hypothetical answer (not the original query) using the project's embedding model (`paraphrase-multilingual-MiniLM-L12-v2`).
4. Use *that* embedding to perform dense retrieval against the real knowledge base — the hypothetical answer's embedding, being written in answer-style formal English, is much closer in vector space to the real policy document chunks than the original terse Hinglish query would have been.
5. Retrieve real chunks as normal; the hypothetical answer is discarded after its embedding is computed — it's never shown to the customer and never cited, since it might be factually wrong (it was never grounded in the real knowledge base to begin with).

---

### 3. How It Is Implemented in This Project

```python
def hyde_rewrite_and_embed(query: str, client, model_id: str, embed_model) -> "np.ndarray":
    """HyDE: generate a hypothetical answer, embed IT instead of the raw query.
    The hypothetical answer is discarded after embedding -- never shown to the
    customer, never cited, since it's ungrounded and may be factually wrong."""

    hyde_prompt = f"""Write a brief, plausible-sounding answer to this customer question
about Fixed Deposits, as if you were an FD policy document. Do not worry about being
factually precise -- just write in the style and vocabulary a real policy answer would use.

Question: {query}

Hypothetical answer:"""

    response = client.messages.create(
        model=model_id, max_tokens=150,
        messages=[{"role": "user", "content": hyde_prompt}],
    )
    hypothetical_answer = response.content[0].text.strip()

    # Embed the HYPOTHETICAL ANSWER, not the original query
    hyde_embedding = embed_model.encode(hypothetical_answer, normalize_embeddings=True)
    return hyde_embedding


def hyde_dense_retrieve(query: str, knowledge_base: list, client, model_id: str,
                         embed_model, top_k: int = 5) -> list:
    """Dense retrieval using HyDE's hypothetical-answer embedding instead of
    the raw query embedding. Reuses the exact same cosine-similarity search
    logic from Chapter 7 Topic 2 -- only the QUERY VECTOR construction changes."""
    import numpy as np

    hyde_vec = hyde_rewrite_and_embed(query, client, model_id, embed_model)

    scored = [(doc, float(np.dot(hyde_vec, doc["embedding"]))) for doc in knowledge_base]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def simple_query_expansion(query: str, hinglish_map: dict) -> str:
    """Lightweight alternative to HyDE: expand known Hinglish terms with
    their English equivalents before retrieval -- the Chapter 7 Topic 1
    fix, formalized as a reusable query rewriting step."""
    words = query.lower().split()
    expanded = list(words)
    for word in words:
        if word in hinglish_map and hinglish_map[word] not in expanded:
            expanded.append(hinglish_map[word])
    return " ".join(expanded)
```

- HyDE's dense retrieval integrates as a drop-in alternative query-vector source within Chapter 7's existing hybrid RRF pipeline — BM25 continues to use the original raw query (HyDE specifically helps dense retrieval, not BM25, since BM25's exact-term matching doesn't benefit from a paraphrased hypothetical answer the same way), and the fused ranking combines BM25-on-raw-query with dense-on-HyDE-embedding.

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

- **HyDE adds a full extra LLM call before retrieval even starts**: this is a genuine latency and cost trade-off — an additional `claude-haiku-4-5-20251001` round-trip per query, on top of everything else in the pipeline. For this project's production volume (8,000-12,000 emails/day), this is a meaningful, measurable added cost that must be weighed against the retrieval-quality improvement it provides.
- **The hypothetical answer can be confidently wrong, and that's acceptable — but only because it's never surfaced**: HyDE's entire safety depends on the hypothetical answer being strictly an internal retrieval-aid artifact, discarded immediately after embedding. If any part of the pipeline accidentally leaks the hypothetical answer to the customer or into the citation/faithfulness pipeline (Topics 2 and 4), it would be a serious, hard-to-detect bug — the hypothetical answer was never grounded in real context and must never be treated as if it were.
- **HyDE quality depends on the generating model's domain familiarity**: if the model has poor general knowledge of FD terminology (unlikely for a well-known Claude model, but a real consideration for smaller or less capable models), the hypothetical answer could be so far off-topic or off-register that its embedding is actually a worse retrieval signal than the raw query — this should be validated empirically, not assumed to always help.
- **Monitoring**: track Recall@K and NDCG@K (Chapter 7 Topic 9's methodology) separately for HyDE-based dense retrieval versus raw-query dense retrieval on the same evaluation set — this is the only way to confirm HyDE is actually improving retrieval for this specific corpus, rather than assuming the technique's general reputation transfers automatically.
- **Cost**: an additional model call per query at this project's volume is a non-trivial addition to the overall per-request cost profile established across Chapter 7 and this chapter — should be explicitly budgeted and tracked as its own cost line, not folded silently into "retrieval cost."
- **Security**: the HyDE generation prompt should be scoped narrowly enough that it can't be manipulated by adversarial input in the original customer query into generating harmful or off-policy content, even though the hypothetical answer is never shown to anyone — an injected instruction in the customer query could still influence what gets embedded and therefore what gets retrieved, a subtler injection vector than directly manipulating a visible output.
- **Deployment**: HyDE should be implemented as a swappable retrieval-query-construction strategy, not hardwired — this project's hybrid pipeline should be able to toggle between raw-query dense retrieval and HyDE-based dense retrieval (or run both and compare) without restructuring the rest of the pipeline, supporting the evaluation-driven adoption decision described below.

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

- **HyDE for every query vs. selectively applied**: running HyDE on every query maximizes potential benefit but maximizes cost and latency uniformly; a more targeted approach could apply HyDE selectively — e.g. only for queries where the raw-query dense retrieval score is unusually low (a signal the query itself may be a poor retrieval input), falling back to standard dense retrieval otherwise. This adds complexity but concentrates the extra cost where it's most likely to help.
- **HyDE vs. simple query expansion (Chapter 7 Topic 1's Hinglish mapping)**: query expansion is cheap (no extra LLM call, just a dictionary lookup) and directly targets this project's known Hinglish vocabulary mismatch; HyDE is more general-purpose and can help with a wider range of phrasing problems beyond just Hinglish-English mismatch, but at meaningfully higher cost. For this project specifically, given that Hinglish vocabulary mismatch is the *measured*, dominant retrieval weakness (established repeatedly across Chapter 7), the cheaper, more targeted query-expansion fix may deliver most of the value at a fraction of HyDE's cost — this should be validated head-to-head via Chapter 7 Topic 9's evaluation methodology before committing to HyDE as a production default.
- **Combining HyDE and query expansion**: these aren't mutually exclusive — a Hinglish query could first be expanded with English equivalents, then used as the basis for HyDE's hypothetical-answer generation, potentially combining both techniques' benefits. Worth testing as a combined configuration in the evaluation harness, not just as two separate alternatives.

---

### 6. Alternatives and When to Use Each

- **No query rewriting (raw query used directly, Chapter 7 as originally built)**: the simplest baseline, and may already be sufficient once hybrid retrieval (BM25 + dense + RRF) is in place, since BM25 already handles exact-term cases and dense retrieval handles some vocabulary bridging on its own — always the first configuration to measure against before adding rewriting complexity.
- **Simple query expansion / Hinglish term mapping (cheap, targeted)**: the most directly applicable fix given this project's specifically identified Hinglish vocabulary mismatch problem — low cost, easy to build and maintain, directly targets a known, measured weakness.
- **HyDE (general-purpose, higher cost)**: worth adopting if evaluation shows it provides meaningfully better retrieval quality than cheaper alternatives, particularly for queries that are poorly-phrased or vague in ways that simple term-substitution can't fix.
- **Query decomposition for multi-part questions**: reserved for the subset of customer queries that genuinely bundle multiple distinct questions together — a narrower, specific-purpose technique rather than a general default.

---

### 7. Common Mistakes and Production Failures

- Leaking HyDE's hypothetical, ungrounded answer to the customer or into the citation/faithfulness pipeline — a serious correctness and trust failure, since the hypothetical answer was never verified against real context.
- Adopting HyDE as a default without measuring it against cheaper alternatives (query expansion) on this project's actual evaluation set — given the specifically diagnosed Hinglish vocabulary mismatch problem, a cheaper fix may already solve most of the issue.
- Applying HyDE to BM25 retrieval, where it provides little benefit since BM25's exact-term matching doesn't gain from a paraphrased hypothetical answer's embedding the way dense retrieval does — HyDE is a dense-retrieval-specific technique, not a universal query-rewriting fix for every retriever.
- Running HyDE on every single query uniformly without considering a selective, cost-targeted application strategy, when volume and cost constraints (this project's 8,000-12,000 emails/day) make uniform application expensive.
- Not testing whether the HyDE-generating model actually has good enough domain familiarity with FD terminology to produce useful hypothetical answers — assuming it will "just work" rather than validating it empirically.

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: What is HyDE, and why would embedding a hypothetical (possibly wrong) answer produce better retrieval than embedding the original query?**
A: HyDE (Hypothetical Document Embeddings) generates a plausible hypothetical answer to the query using an LLM, then embeds that hypothetical answer instead of the original query for dense retrieval. The intuition is that real documents in a knowledge base are written in answer-style prose, and a hypothetical answer — even if factually imprecise — shares that same style and vocabulary, making its embedding a closer match to genuinely relevant real documents than a terse, differently-phrased original query would be.

**Q: Why must the hypothetical answer generated by HyDE never be shown to the customer or used for citation?**
A: The hypothetical answer is generated without access to the real knowledge base — it's the model's own guess at what a plausible answer might look like, purely for the purpose of producing a better embedding. It was never grounded or verified, so it can be factually wrong. If it were surfaced to the customer or cited as if it were real, this would undermine everything the citation (Topic 2) and hallucination detection (Topic 4) mechanisms exist to guarantee.

**Intermediate:**

**Q: This project already identified Hinglish vocabulary mismatch as a specific, measured retrieval weakness (Chapter 7 Topic 1). Would you recommend HyDE or simple query expansion as the first fix to try, and why?**
A: Simple query expansion (mapping known Hinglish terms to their English equivalents before retrieval) is cheaper, faster, and directly targets the specifically diagnosed problem — no additional LLM call is needed, just a dictionary lookup, and it's easy to build, maintain, and validate against Chapter 7 Topic 9's evaluation methodology. HyDE is more general-purpose and could help with a broader range of phrasing issues, but adds meaningful cost and latency at this project's production volume. Given the problem is already specifically diagnosed, I'd try the cheaper, targeted fix first, measure its actual impact on Recall@K/NDCG@K, and only add HyDE if query expansion alone proves insufficient.

**Q: Why does HyDE specifically help dense retrieval but not meaningfully help BM25?**
A: BM25 (Chapter 7 Topic 1) scores based on exact term overlap and corpus-wide term statistics — a hypothetical answer's paraphrased vocabulary doesn't help BM25 find exact-term matches any better than the original query would, and could actually dilute BM25's precision if used naively. Dense retrieval, in contrast, matches based on semantic similarity in embedding space, where a hypothetical answer's answer-style phrasing genuinely does produce a closer match to real answer documents than a terse original query — this is why HyDE is applied specifically to the dense retrieval leg of a hybrid pipeline, leaving BM25 to operate on the original raw query.

**Advanced:**

**Q: Design an evaluation methodology to determine whether HyDE is worth adopting as a production default for this project, given its added cost.**
A: Using Chapter 7 Topic 9's labeled evaluation set (extended to include the project's known-weak query types — Hinglish phrasings, vague follow-ups, vocabulary-mismatched queries), run three configurations head-to-head: raw-query dense retrieval, query-expansion-then-dense-retrieval, and HyDE-based dense retrieval, all combined identically with BM25 via the existing hybrid RRF pipeline. Compute Recall@K, MRR, and NDCG@K for each configuration. If HyDE's improvement over query expansion alone is small or within noise given the evaluation set size, the added cost isn't justified given a cheaper alternative already captures most of the benefit. If HyDE shows a meaningful, statistically credible improvement specifically on the harder, more ambiguous query subset, it may be worth adopting selectively — applied only when a cheap upstream signal (e.g. low raw-query retrieval confidence) suggests the query is likely to benefit, rather than uniformly on every request, to control the added cost.

**Q: A teammate proposes using HyDE's hypothetical answer directly as a cache — if a very similar query comes in later, serve the cached hypothetical answer as a starting point for a faster response. What's the risk?**
A: This conflates HyDE's role (an internal, ungrounded, retrieval-aid artifact) with a legitimate answer-generation shortcut, which is a serious risk — HyDE's hypothetical answer was never verified against real context and could be factually wrong. Caching and reusing it as if it were a validated answer, even as a "starting point," creates a path for ungrounded, unverified content to influence what a customer eventually sees, bypassing the citation and hallucination-detection guarantees established in Topics 2 and 4. If caching is desired for latency/cost reasons, the correct target for caching is the *final, verified* answer for genuinely repeated queries (or repeated retrieved-context-plus-answer pairs), not the intermediate, ungrounded HyDE artifact.

**Scenario-based:**

**Q: After adopting HyDE for dense retrieval, you observe that retrieval quality improved for English queries but showed no improvement — or slightly worse results — for Hinglish queries, which was the original motivating problem. Diagnose.**
A: This suggests the HyDE-generating model may be producing hypothetical answers primarily in English regardless of the Hinglish input query (a plausible behavior, since it was prompted to write "in the style a real policy answer would use," and this project's actual policy documents are in English) — meaning HyDE isn't actually addressing the Hinglish-specific vocabulary bridging problem any differently than a straightforward English-only dense retrieval already does; it's solving a different problem (general phrasing/register mismatch) that happens to not be this project's dominant weakness. This points back toward the simpler, more directly targeted query-expansion approach (explicit Hinglish-to-English term mapping) as the more appropriate fix for the *specific* problem this project actually has, with HyDE potentially still useful as a secondary technique for other, non-Hinglish-specific phrasing issues — reinforcing the value of measuring each technique against the actual, specific failure modes rather than adopting a generally-reputed technique and assuming it addresses whatever problem prompted looking into query rewriting in the first place.

---

### 9. Hidden Concepts and Prerequisites

- **HyDE's original research context**: HyDE (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels") was originally motivated by zero-shot retrieval — cases with no labeled query-document relevance data to fine-tune a retrieval model on. This project has been building exactly such labeled data via Chapter 7 Topic 9's evaluation harness, which is itself a useful lens: HyDE is most valuable exactly in the label-scarce regime, and its relative value diminishes as this project accumulates more labeled data that could instead be used to fine-tune a retrieval or embedding model directly (Chapter 7 Topic 2's discussion of embedding model fine-tuning) — worth knowing this trade-off exists rather than treating HyDE as a permanent architectural fixture.
- **Query rewriting connects forward to Chapter 9's "Query Transformation" topic**: this project's syllabus explicitly earmarks Chapter 9 Topic 2 for query transformation specific to the Hinglish, short, code-mixed email problem — this Chapter 8 topic is the general RAG-generation-layer introduction to the concept; Chapter 9 revisits and deepens it specifically in the context of wiring retrieval into the full agent tool-use architecture.
- **The generating model for HyDE need not be the same model used for the final answer**: since the hypothetical answer is discarded and never shown to anyone, it's reasonable to use a smaller, cheaper, or even different-provider model specifically for HyDE generation, decoupling this cost from the final answer-generation model choice — a genuine cost-optimization lever separate from anything else covered in this chapter.

---

### 10. Revision Summary

> Query rewriting transforms a poor retrieval input into a better one before Chapter 7's retrieval pipeline runs, addressing single-turn phrasing and vocabulary problems distinct from Topic 6's multi-turn contextualization concern. HyDE specifically generates a hypothetical, ungrounded answer via an LLM call, then embeds that hypothetical answer (discarded immediately after) instead of the original query for dense retrieval — exploiting the fact that answer-style prose embeds closer to real knowledge-base documents than terse original queries do. This technique specifically helps dense retrieval, not BM25. For this project, given the already-diagnosed, specific Hinglish vocabulary mismatch problem (Chapter 7 Topic 1), a cheaper, targeted query-expansion fix (Hinglish-to-English term mapping) should be tried and measured first via Chapter 7 Topic 9's evaluation methodology before adopting HyDE's added per-query LLM-call cost — HyDE's more general-purpose phrasing correction may add less value here than its reputation suggests, since the corpus's dominant weakness is a specific, already-identified vocabulary gap rather than general phrasing ambiguity. The hypothetical answer must never be shown to customers or used for citation, since it was never grounded in real context.

---
