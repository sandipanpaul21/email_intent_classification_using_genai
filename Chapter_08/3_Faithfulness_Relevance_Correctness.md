# Chapter 8: RAG Generation — The Answer Layer

## Topic 3: Faithfulness vs. Relevance vs. Correctness — Three Distinct Failure Modes

---

### 1. Concept, Intuition, and Why It Exists

- "The answer is wrong" is not a single failure mode — it conflates at least three genuinely different problems that require different diagnoses and different fixes. This topic exists to give each one a precise name, because Topic 4's hallucination detection, Chapter 7's retrieval evaluation, and general answer-quality review all target *different* ones of these three, and conflating them wastes debugging effort chasing the wrong layer of the pipeline.
- **Faithfulness**: does the generated answer accurately reflect what the *retrieved context* actually says, regardless of whether that context itself is correct or complete? A faithfulness failure is a generation-layer problem — the model said something the provided documents don't support.
- **Relevance**: does the generated answer actually address what the *user asked*, regardless of whether it's faithful to the context? A relevance failure can happen even with perfect faithfulness — the model can accurately summarize retrieved content that simply doesn't answer the question asked (a retrieval-layer symptom surfacing at generation time).
- **Correctness**: is the answer actually *true*, according to ground truth (the real world, or an authoritative source), regardless of what the retrieved context said? A correctness failure can happen even with perfect faithfulness and relevance — if the retrieved context itself was wrong, outdated, or incomplete, a perfectly faithful summary of it is still factually wrong.
- The critical insight this topic exists to teach: these three can fail independently, in any combination, and a single "the answer was wrong" bug report gives no information about which one broke without a structured way to distinguish them.

---

### 2. Internal Working — Step by Step

**Distinguishing the three in practice, using this project's FD domain as the running example:**

1. **Faithfulness check**: compare each claim in the generated answer against the specific text of the retrieved chunks that were actually sent in context (the same `context_sources` from Topic 2). If the answer states "the penalty is 2%" but the retrieved chunk says "1%", this is an unfaithful answer — a generation-layer failure, independent of whether 1% or 2% is the actual correct real-world rate.
2. **Relevance check**: compare the generated answer against the user's actual question, independent of the retrieved context's content. If a customer asks "what is the penalty for early withdrawal" and the answer discusses senior citizen interest rate bonuses (faithfully summarizing a retrieved chunk that happened to be about that topic, due to a Chapter 7 retrieval miss), the answer is faithful to what was retrieved but irrelevant to what was asked.
3. **Correctness check**: compare the generated answer against ground truth — is the actual policy rate 1%, and does the underlying knowledge base correctly and currently reflect that? If the knowledge base itself contains an outdated policy document (a Chapter 4 ingestion/versioning issue, or Chapter 6 Topic 8's embedding model migration and staleness concerns), a perfectly faithful, perfectly relevant answer can still be factually wrong.
4. **The four combinations that matter most in debugging**:
   - Faithful + Relevant + Correct: the target state, everything working.
   - Faithful + Relevant + Incorrect: retrieved context itself was wrong or stale — a Chapter 4/6 knowledge-base problem, not a Chapter 8 generation problem.
   - Faithful + Irrelevant: retrieval found the wrong chunks (Chapter 7 problem) and generation faithfully but uselessly summarized them.
   - Unfaithful (regardless of relevance/correctness): a generation-layer hallucination — Topic 4's specific target.

---

### 3. How It Is Implemented in This Project

- **Faithfulness** is checked using the citation mechanism from Topic 2 as its foundation: once an answer's claims are attributed to specific source chunks, faithfulness checking becomes "does this claim actually match what the cited chunk says" — an entailment-style comparison, covered operationally in Topic 4.
- **Relevance** is checked independently of the retrieved context entirely — comparing the generated answer's content against the original query, which can reuse the same embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, established in Chapter 3/6) to compute a semantic similarity between query and answer as a coarse relevance signal, or an LLM-as-judge call for a more nuanced assessment (forward reference to Chapter 14).
- **Correctness** cannot be fully automated within this project's own pipeline — it fundamentally requires either human-labeled ground truth (the same evaluation-set discipline established in Chapter 7 Topic 9, extended to generated answers rather than just retrieved chunks) or an external authoritative source to check against. This project's MuRIL baseline (0.97 FD Recall) and the labeled dev sets already used in Chapter 2's prompt evaluation are the closest existing analogs — correctness evaluation for generation is the same evidence-based discipline applied one layer further downstream.

```python
def check_faithfulness(answer_claims: list, cited_context_map: dict) -> dict:
    """Coarse faithfulness check: does each claim's cited source text
    actually contain content consistent with the claim? Full semantic
    entailment is expensive (Topic 4); this is the cheap structural layer --
    confirms every claim maps to SOME cited source before deeper checking."""
    unsupported = []
    for claim in answer_claims:
        source = claim.get("cited_source")
        if source not in cited_context_map:
            unsupported.append(claim)
    return {"all_claims_have_valid_source": len(unsupported) == 0, "unsupported_claims": unsupported}


def check_relevance(query: str, answer: str, embed_model) -> float:
    """Coarse relevance signal: semantic similarity between the ORIGINAL
    QUERY and the GENERATED ANSWER -- independent of what was retrieved.
    A low score suggests the answer may not address what was actually asked,
    regardless of how faithful it is to whatever context was retrieved."""
    import numpy as np
    query_vec = embed_model.encode(query, normalize_embeddings=True)
    answer_vec = embed_model.encode(answer, normalize_embeddings=True)
    return float(np.dot(query_vec, answer_vec))


# Correctness: cannot be checked automatically without ground truth.
# Requires either:
#   (a) a human-labeled evaluation set (Chapter 7 Topic 9's discipline,
#       extended to (query, correct_answer) pairs), or
#   (b) cross-referencing against an authoritative, versioned source of
#       truth outside this pipeline entirely (e.g. the actual current
#       RBI-compliant policy document, verified by a domain expert)
```

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

- **The most common real-world confusion**: treating a correctness failure (stale or wrong knowledge base content) as if it were a faithfulness failure (model misrepresenting correct content) — these require completely different fixes. Faithfulness fixes belong in prompting or generation-layer tooling (Topic 4); correctness fixes belong in knowledge base freshness and Chapter 4/6 ingestion pipelines. Debugging effort spent tuning prompts to fix what is actually a stale-document problem is wasted effort.
- **Relevance failures often masquerade as retrieval failures, and usually are**: an irrelevant-but-faithful answer is the generation layer accurately reporting on the wrong input — the actual bug lives in Chapter 7's retrieval or reranking, not in Chapter 8's generation. This is exactly why this topic insists on decomposing "wrong answer" reports before assigning them to a pipeline stage.
- **Cost and latency of each check**: faithfulness checking (comparing claims to cited sources) is relatively cheap if built on top of Topic 2's citation infrastructure. Relevance checking via embedding similarity is cheap (a single additional embedding call). Correctness checking is the most expensive and slowest, since it fundamentally requires either human review or an external, authoritative check — it cannot be fully automated within the pipeline's own components, since the pipeline cannot validate its own knowledge base's truthfulness against itself.
- **Monitoring**: track faithfulness and relevance scores as continuous production metrics (both are automatable); track correctness only via periodic sampled human review or explicit customer feedback signals, since it cannot run on every request at production volume.
- **Security**: none of these three checks are security controls per se, but a systematic faithfulness failure pattern (the model consistently fabricating claims not supported by context) is a useful early signal that could also indicate prompt injection is succeeding — the checks built for quality assurance double as a partial defense-in-depth signal.
- **Deployment**: faithfulness and relevance checks are cheap enough to run synchronously in the request path for a regulated domain; correctness review is inherently asynchronous (human-in-the-loop or periodic audit), and should never block a live customer response.

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

- **How much automated checking is worth building vs. accepting some human review load**: faithfulness and relevance are automatable at reasonable cost and should be built into the production path; correctness fundamentally cannot be fully automated without an external ground truth, so a real-time dilemma is how much of the correctness burden to accept as an ongoing human-review cost versus how much to invest in better ground-truth infrastructure (e.g. a more rigorously maintained, versioned knowledge base with clear provenance and freshness guarantees, connecting back to Chapter 4/6's ingestion discipline).
- **Where to draw the line between "relevance failure" and "faithfulness failure" when both retrieval AND generation contributed**: in practice, a single bad answer can have contributions from both layers simultaneously — a mediocre retrieval result combined with a generation layer that stretches to make the mediocre context seem more relevant than it is. Attribution in these mixed cases requires the disciplined layer-by-layer check described in Section 2, not an either/or judgment.

---

### 6. Alternatives and When to Use Each

- **A single, undifferentiated "answer quality" score**: simpler to implement and communicate, but actively harmful for debugging — it gives no signal about which pipeline stage to fix, and different failure modes require entirely different remediation. Not recommended once a system is past early prototyping.
- **Three separate, explicitly-named metrics (this topic's approach)**: the correct choice once a RAG system needs to be debugged and improved iteratively by different specialists (retrieval engineers fixing relevance-driving-issues, prompt engineers fixing faithfulness, domain experts fixing correctness/knowledge-base staleness).
- **RAGAS's formal metric suite (Faithfulness, Answer Relevancy, Context Precision, Context Recall — forward reference to Chapter 14)**: a more rigorous, standardized version of exactly this topic's distinctions, worth adopting once the project moves to systematic, ongoing RAG evaluation rather than ad hoc debugging.

---

### 7. Common Mistakes and Production Failures

- Treating every "wrong answer" bug report identically without first determining which of the three failure modes actually occurred, leading to wasted effort fixing the wrong pipeline stage.
- Assuming a faithful answer is automatically correct — faithfulness only guarantees consistency with the retrieved context, not truth, and a stale or wrong knowledge base can produce faithful-but-incorrect answers indefinitely without any faithfulness check ever catching it.
- Not distinguishing relevance failures from retrieval failures organizationally — a relevance metric dropping in production should route the investigation to Chapter 7's retrieval pipeline, not to Chapter 8's prompting, but teams without clear metric separation often default to prompt-tuning as the first response regardless of root cause.
- Building expensive, real-time correctness checking that assumes it can be fully automated — correctness fundamentally requires ground truth external to the pipeline, and treating it as solvable purely through better prompting or better models is a category error.

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: Define faithfulness, relevance, and correctness, and give an example where each could fail independently.**
A: Faithfulness: does the answer match the retrieved context? A model stating "2%" when the retrieved chunk says "1%" is unfaithful. Relevance: does the answer address the question asked? A faithful summary of a retrieved chunk about senior citizen rates, in response to a question about withdrawal penalties, is irrelevant despite being faithful. Correctness: is the answer actually true? A perfectly faithful, perfectly relevant answer can still be wrong if the underlying retrieved document itself is outdated or incorrect.

**Q: Why can't correctness be fully automated the way faithfulness and relevance can?**
A: Faithfulness and relevance can be checked using tools already inside the pipeline — comparing the answer to context (faithfulness) or to the query (relevance) via embeddings or entailment models. Correctness requires comparing the answer to ground truth external to the pipeline — the pipeline has no way to validate its own knowledge base's accuracy against itself, so correctness checking fundamentally requires either human review or an authoritative external source.

**Intermediate:**

**Q: A customer complains that an answer about FD withdrawal penalties was wrong. Your faithfulness and relevance checks both pass. Where do you look next?**
A: If faithfulness (the answer matches what was retrieved) and relevance (the answer addresses the question asked) both check out, the remaining possibility is a correctness failure — the retrieved context itself was wrong or outdated. This points investigation toward the knowledge base's freshness and provenance (Chapter 4 ingestion, Chapter 6 embedding/document versioning) rather than toward the generation prompt or retrieval ranking, since both of those layers performed correctly given what they had to work with.

**Q: How would you build a lightweight, production-viable faithfulness check without a full entailment model?**
A: Reuse Topic 2's citation infrastructure as the foundation — require the model to attribute each factual claim to a specific cited source, then perform an initial cheap structural check (does the cited source actually exist in the context that was sent — Topic 2's `verify_citations`), followed by a coarser faithfulness signal such as lexical or embedding-similarity overlap between the claim and the cited passage. This won't catch subtle semantic contradictions as reliably as a dedicated entailment model, but it's cheap enough to run on every production request, reserving more expensive entailment-based checking (Topic 4) for sampled or flagged cases.

**Advanced:**

**Q: Design an escalation and routing policy for a production RAG system that distinguishes these three failure modes automatically where possible, and routes appropriately when it can't.**
A: Faithfulness and relevance checks run synchronously on every request — faithfulness via the citation-and-claim-matching approach described above, relevance via query-answer embedding similarity. A low faithfulness score triggers an automatic fallback (route to human review, or regenerate with a stricter grounding prompt) since this is a clear generation-layer signal the pipeline can act on directly. A low relevance score routes the request back through Chapter 7's retrieval diagnostics rather than regenerating with the same context, since relevance failures are usually retrieval-layer symptoms. Correctness cannot be checked per-request in real time — instead, implement periodic sampled review (a percentage of daily traffic, weighted toward low-confidence or customer-flagged cases) by a domain expert, feeding confirmed correctness failures back into both the knowledge base (fixing or flagging stale source documents) and the Chapter 7 Topic 9 evaluation set (adding the case as a new labeled example).

**Q: A teammate proposes using a single RAGAS-style aggregate score to represent overall RAG quality on a dashboard. What's your concern?**
A: An aggregate score is useful for tracking overall trend direction over time, but it collapses exactly the distinction this topic exists to preserve — a dropping aggregate score gives no signal about which of faithfulness, relevance, or correctness degraded, and therefore no signal about which team or pipeline stage should investigate. I'd keep the aggregate for executive-level trend reporting but insist the underlying dashboard break out the three (or more, per RAGAS's actual metric set) components separately for anyone actually debugging or improving the system — collapsing them for convenience defeats the purpose of measuring them separately in the first place.

**Scenario-based:**

**Q: Over several weeks, faithfulness scores remain stable and high, but customer satisfaction with FD-related answers is declining. Correctness spot-checks by a domain expert reveal the knowledge base's premature withdrawal penalty documentation is six months out of date following an undocumented policy change. Walk through the fix, and what process gap allowed this to go undetected for so long.**
A: The immediate fix is updating the knowledge base with the current policy document and re-ingesting (Chapter 4's pipeline) — a content problem, not a model or prompting problem, confirmed precisely because faithfulness stayed high throughout (the model was accurately reflecting what it was given, the whole time). The process gap: this project's correctness checking relies on periodic, sampled human review rather than any systematic trigger for knowledge base staleness — there's no automated signal that fires when an underlying source document becomes outdated, only a downstream, lagging signal via customer dissatisfaction or spot-checks. A structural fix is establishing a knowledge base freshness/versioning discipline at the ingestion layer (a review or expiry date attached to policy-sensitive documents, similar in spirit to Chapter 6 Topic 8's model-migration staleness tracking, applied here to content rather than embeddings) so correctness risk from stale content is caught proactively rather than discovered reactively through declining customer satisfaction.

---

### 9. Hidden Concepts and Prerequisites

- **This three-way decomposition maps directly onto RAGAS's formal metrics** (Chapter 14): Faithfulness (RAGAS Faithfulness), Relevance (RAGAS Answer Relevancy), and the retrieval-side analogs (Context Precision, Context Recall) that determine whether correctness failures trace back to retrieval quality specifically — this topic is the conceptual foundation the later, more formal evaluation chapter builds on.
- **Correctness is fundamentally an epistemics problem, not an engineering one**: no amount of pipeline sophistication can substitute for having an actual, current, authoritative source of truth — this is why knowledge base governance (who updates policy documents, how quickly, with what review process) is as much a part of "RAG system design" as anything covered in Chapters 4-8, even though it's an organizational process, not a technical component.
- **The three failure modes are not mutually exclusive and can compound**: a system can simultaneously have a relevance problem (Chapter 7 retrieval issues) and a correctness problem (stale knowledge base) contributing to the same bad answer — disentangling compound failures requires checking each dimension independently rather than assuming a single root cause.

---

### 10. Revision Summary

> "Wrong answer" is not one failure mode — it decomposes into faithfulness (does the answer match the retrieved context?), relevance (does the answer address the question asked?), and correctness (is the answer actually true?), and each can fail independently. Faithfulness and relevance are automatable within this project's own pipeline — faithfulness via Topic 2's citation infrastructure plus claim-to-source matching, relevance via query-answer embedding similarity — and can run synchronously on every production request. Correctness cannot be fully automated, since it requires ground truth external to the pipeline itself; it depends on human review and knowledge base governance, connecting back to Chapter 4's ingestion and Chapter 6's document freshness concerns. Debugging a bad answer requires checking these three independently, in order, since a faithful-and-relevant-but-incorrect answer, a faithful-but-irrelevant answer, and an unfaithful answer each point to entirely different pipeline stages and require entirely different fixes.

---
