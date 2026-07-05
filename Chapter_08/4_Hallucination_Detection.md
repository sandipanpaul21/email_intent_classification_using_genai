# Chapter 8: RAG Generation — The Answer Layer

## Topic 4: Hallucination as a Measured Failure Mode — Detection Beyond "Instruct the Model Not To"

---

### 1. Concept, Intuition, and Why It Exists

- Every RAG system prompt in this project family says some version of "only use the provided context" and "say you don't know if the context is insufficient." This is necessary — but Topic 3 already established that faithfulness failures (the specific technical name for a hallucination in RAG) can and do still occur despite explicit instructions, because instruction-following is probabilistic, not guaranteed.
- This topic exists to move past "we told it not to" into "here is how we actually detect when it did it anyway" — a measured, structural safety net that doesn't depend on the model's compliance being perfect.
- The core reframe: hallucination detection is not a prompting problem, it's a **verification** problem — given a generated answer and the context it was supposed to be grounded in, can you independently check whether the grounding actually held, after the fact, using a method separate from the generation process itself?

---

### 2. Internal Working — Step by Step

**Three concrete, increasingly rigorous detection techniques:**

1. **Claim extraction**: break the generated answer into discrete, checkable factual claims — "the penalty is 1%", "it applies to premature withdrawal", "senior citizens get an additional 0.5%" are three separate claims, not one blob of text. This decomposition is what makes the rest of the pipeline tractable; checking one long paragraph against context holistically is much less reliable than checking each atomic claim.
2. **Source mapping (built on Topic 2's citation mechanism)**: for each extracted claim, identify which cited source chunk it should be checked against — this is exactly the `sources_used` structured output already established in Topic 2.
3. **Entailment checking**: for each (claim, source chunk) pair, determine whether the source chunk actually **entails** the claim — does the source text logically support this specific statement, not just share some vocabulary with it. Three practical methods, in increasing cost order:
   - **Lexical/embedding overlap**: cheap, coarse — compute similarity between the claim and the source passage; low similarity is a red flag, but high similarity doesn't guarantee true entailment (a source can be topically similar while actually contradicting the claim).
   - **Dedicated NLI (Natural Language Inference) model**: a model specifically trained to classify a (premise, hypothesis) pair as entailment, contradiction, or neutral — more reliable than raw similarity, moderate cost, can run locally.
   - **LLM-as-judge**: prompt a separate model call (or a separate call to the same model, at lower temperature, with a narrowly-scoped judging prompt) to explicitly answer "does this source passage support this specific claim, yes or no" — most reliable, most expensive, and introduces the LLM-judging-LLM concerns covered in Chapter 16.
4. **Aggregation into an answer-level faithfulness verdict**: an answer with even one unsupported claim among several supported ones is not simply "50% hallucinated" in any usable sense — the aggregation policy (does one bad claim invalidate the whole answer, or just get flagged) is a real design decision, covered in Section 5.

---

### 3. How It Is Implemented in This Project

- Claim extraction and source mapping reuse Topic 2's structured citation tool output directly — `finalize_answer_with_citations` already requires the model to declare `sources_used`; extending this schema to require per-claim source attribution (rather than one list for the whole answer) is the natural refinement this topic motivates.
- For entailment checking, this project's existing local-model-first philosophy (established with `paraphrase-multilingual-MiniLM-L12-v2` for embeddings, and the local-vs-API build-vs-buy discussion in Chapter 7 Topic 7 for reranking) suggests starting with the cheap embedding-overlap tier for every request, escalating to LLM-as-judge only for flagged or sampled cases — not running the most expensive check on every single production request.

```python
import numpy as np

ANSWER_TOOLS_WITH_PER_CLAIM_CITATION = [
    {
        "name": "finalize_answer_with_claims",
        "description": "Call this when ready to answer. Break your answer into discrete factual claims, each attributed to one source.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "cited_source": {"type": "string"},
                        },
                        "required": ["text", "cited_source"],
                    },
                },
            },
            "required": ["claims"],
        },
    }
]


def check_entailment_embedding(claim_text: str, source_text: str, embed_model,
                                threshold: float = 0.5) -> dict:
    """Tier 1 (cheap): embedding similarity as a coarse entailment proxy.
    Runs on EVERY production request."""
    claim_vec = embed_model.encode(claim_text, normalize_embeddings=True)
    source_vec = embed_model.encode(source_text, normalize_embeddings=True)
    similarity = float(np.dot(claim_vec, source_vec))
    return {"similarity": similarity, "flagged": similarity < threshold}


def check_entailment_llm_judge(claim_text: str, source_text: str, client, model_id: str) -> dict:
    """Tier 2 (expensive): LLM-as-judge, explicit entailment question.
    Runs only on Tier 1-flagged claims, or a random sample."""
    judge_prompt = f"""Source passage: "{source_text}"
Claim: "{claim_text}"

Does the source passage support this claim? Answer with exactly one word: YES, NO, or PARTIAL."""
    response = client.messages.create(
        model=model_id, max_tokens=10,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    return {"verdict": verdict, "entailed": verdict == "YES"}


def evaluate_answer_faithfulness(claims: list, context_map: dict, embed_model,
                                  client, model_id: str, embedding_threshold: float = 0.5) -> dict:
    """Full pipeline: cheap tier for every claim, expensive tier only for flagged claims."""
    results = []
    for claim in claims:
        source_text = context_map.get(claim["cited_source"], "")
        tier1 = check_entailment_embedding(claim["text"], source_text, embed_model, embedding_threshold)
        result = {"claim": claim["text"], "tier1": tier1}
        if tier1["flagged"]:
            tier2 = check_entailment_llm_judge(claim["text"], source_text, client, model_id)
            result["tier2"] = tier2
        results.append(result)

    unsupported = [r for r in results if r.get("tier2", {}).get("verdict") == "NO"
                   or (r["tier1"]["flagged"] and "tier2" not in r)]
    return {"all_supported": len(unsupported) == 0, "claim_results": results, "unsupported": unsupported}
```

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

- **Embedding similarity is a weak entailment proxy — the specific failure it misses**: two passages can be highly similar in embedding space while one directly contradicts the other (e.g. "the penalty is 1%" vs. "the penalty is not 1%, it is 2%" — these can embed closely due to shared vocabulary despite being contradictory). This is why the two-tier design exists — embedding similarity is a cheap *filter* to catch obviously unsupported claims and reduce how often the expensive tier needs to run, not a standalone reliable entailment check.
- **Cost of the tiered approach**: for this project's production volume (8,000-12,000 emails/day), running Tier 1 (embedding check) on every claim is negligible cost — the same order of magnitude as the embedding calls already happening for retrieval. Tier 2 (LLM-as-judge) should be reserved for the fraction of claims Tier 1 flags, keeping the expensive check's volume proportional to actual risk, not total volume.
- **Latency**: Tier 1 adds negligible latency (a couple of embedding calls). Tier 2, when triggered, adds a full additional LLM round-trip — for a customer-facing synchronous response, this should be budgeted for explicitly (does hallucination checking happen before the customer sees the answer, adding latency, or after, as an async audit that can trigger a follow-up correction?) — a genuine design trade-off, not a default.
- **Monitoring**: track the Tier 1 flag rate (how often the cheap check catches something) and the Tier 2 confirmation rate (how often the expensive check agrees a flagged claim really is unsupported) — a high flag rate with a low confirmation rate suggests Tier 1's threshold is too aggressive (many false positives, wasted Tier 2 calls); a low flag rate with occasional customer-reported hallucinations that Tier 1 never caught suggests the threshold is too lenient.
- **Security**: hallucination detection is a quality control, not a security control by itself, but a spike in unsupported claims can be an early signal of successful prompt injection (Chapter 3's existing concern) causing the model to generate content not actually grounded in the intended context — worth cross-referencing hallucination spikes against injection-attempt monitoring.
- **Deployment**: the two-tier design should be built as a modular, swappable pipeline stage — Tier 2's LLM-as-judge implementation specifically should be interchangeable (different model, different provider) without changing Tier 1 or the aggregation logic, since judge-model choice and cost will likely need tuning independently over time.

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

- **Synchronous (block the answer) vs. asynchronous (audit after the fact) hallucination checking**: synchronous checking (as shown in the code above) guarantees no unverified answer reaches the customer, at the cost of added latency, especially when Tier 2 fires. Asynchronous checking lets answers go out immediately and flags problems after the fact for correction or follow-up — lower latency, but a customer may briefly receive an unverified (possibly hallucinated) answer. For this project's regulated domain, synchronous checking is the more defensible default, with asynchronous, sampled deeper auditing as a supplementary layer, not a replacement.
- **One unsupported claim invalidating the whole answer vs. partial correction**: the simplest policy blocks the entire answer if any claim fails Tier 2 — safe, but can discard an otherwise-good answer over one flawed detail. A more sophisticated policy could regenerate only the unsupported portion, or explicitly flag the specific claim to the customer as unverified rather than discarding the whole response — added complexity that should be justified by measured need (how often does exactly one claim fail while the rest are fine?) before being built.
- **Threshold tuning for Tier 1's embedding similarity cutoff**: exactly the same evidence-based tuning discipline established throughout Chapter 7 (BM25's k1/b, RRF's k, MMR's lambda) applies here — the threshold should be validated against a labeled set of known-good and known-hallucinated claim/source pairs, not picked by intuition.

---

### 6. Alternatives and When to Use Each

- **Prompting alone ("only use the provided context"), no structural verification**: the minimum viable starting point, and necessary even with verification in place (a good prompt reduces how often verification needs to catch problems), but insufficient alone for a regulated domain given that instruction-following is probabilistic.
- **Two-tier embedding-then-LLM-judge (this project's approach)**: the right balance of cost and reliability for this project's volume and domain sensitivity.
- **Dedicated NLI model as a middle tier**: worth adding between Tier 1 and Tier 2 if the embedding-similarity tier's false-positive rate proves too high in practice — a trained NLI model is more reliable than raw embedding similarity at meaningfully lower cost than an LLM-judge call, and is a natural refinement once volume justifies the added infrastructure.
- **Full LLM-as-judge on every claim, no tiering**: the most reliable single-tier approach, but the most expensive at scale — reasonable only for a much lower-volume, higher-stakes use case than this project's 8,000-12,000 emails/day.

---

### 7. Common Mistakes and Production Failures

- Treating a good system prompt as sufficient protection against hallucination without any structural, post-hoc verification.
- Using embedding similarity as a standalone entailment check without understanding that it cannot reliably distinguish support from contradiction when vocabulary overlaps.
- Running the expensive LLM-as-judge tier on every claim regardless of volume, creating an unnecessary cost and latency burden that a cheap first-tier filter could have avoided.
- Not distinguishing hallucination detection (Topic 4, a faithfulness check) from relevance or correctness checking (Topic 3) — a hallucination detector correctly reports "supported by context" even when the context itself is wrong (a correctness failure) or doesn't answer the question (a relevance failure); conflating these leads to false confidence.
- Blocking every answer with any flagged claim without measuring whether this creates an unacceptable customer-experience cost (too many legitimate answers held up for review) versus the risk being mitigated.

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: Why is "instruct the model not to hallucinate" insufficient as a sole mitigation?**
A: Instruction-following in LLMs is probabilistic, not a hard guarantee — a well-crafted system prompt reduces the *rate* of hallucination but cannot eliminate it structurally. A production system in a regulated domain needs an independent, post-hoc verification mechanism that doesn't rely on the same model's compliance being perfect, which is what hallucination detection provides.

**Q: What is entailment, and why is a two-tier detection approach used instead of a single method?**
A: Entailment is whether a source passage logically supports a specific claim — not just shares vocabulary with it. A single cheap method (embedding similarity) is fast but unreliable at distinguishing genuine support from superficial topical overlap or even direct contradiction. A single expensive method (LLM-as-judge on every claim) is reliable but too costly to run on every claim at production volume. Tiering uses the cheap method as a filter to reduce how often the expensive method needs to run, balancing cost against reliability.

**Intermediate:**

**Q: Walk through why embedding similarity can fail to distinguish a supporting claim from a contradicting one.**
A: Embeddings primarily capture topical and lexical similarity, not logical polarity. "The penalty is 1%" and "the penalty is not 1%, it is 2%" share almost all the same vocabulary and topic, so they embed closely together despite being logically contradictory. A high embedding-similarity score in this case would falsely suggest the claim is well-supported, when the source actually contradicts it — this is precisely why Tier 1 is used only as a coarse filter, not a final verdict.

**Q: How would you decide the threshold for Tier 1's embedding-similarity flag rate?**
A: Build a labeled evaluation set of (claim, source, ground-truth entailment label) pairs — some genuinely supported, some contradicted, some unrelated — following the same evidence-based tuning discipline established in Chapter 7 for BM25's k1/b and RRF's k. Sweep the threshold and measure how well it separates known-good from known-bad pairs, specifically checking the false-negative rate (genuinely unsupported claims that Tier 1 fails to flag, letting them skip Tier 2 entirely) since that's the more costly failure mode for a regulated domain.

**Advanced:**

**Q: Design the full production hallucination-detection pipeline for this project, including the synchronous/asynchronous trade-off and the answer-blocking policy.**
A: Every generated answer's claims (via the per-claim citation schema) pass through Tier 1 (embedding similarity) synchronously, adding negligible latency. Any claim flagged by Tier 1 escalates to Tier 2 (LLM-as-judge) synchronously as well, given the regulated domain's need to avoid surfacing unverified answers — this does add latency for the fraction of requests that trigger it, which should be monitored and budgeted for explicitly. If Tier 2 confirms a claim is unsupported, the full answer is blocked from customer-facing surfacing and routed to human review (mirroring the `refuse_out_of_scope` philosophy), with the specific unsupported claim, its cited source, and the full context logged for review and for potential addition to the Chapter 7 Topic 9 evaluation set. Separately, an asynchronous, lower-priority audit samples a percentage of *passed* answers for deeper review, catching cases where Tier 1 and Tier 2 both missed a subtle issue — this doesn't block anything but feeds back into ongoing threshold tuning and judge-prompt refinement.

**Q: A teammate proposes using the same model (`claude-haiku-4-5-20251001`) for both generation and Tier 2 judging. What's the concern, and how would you address it?**
A: Using the same model for generation and judging introduces a correlated-failure risk — if the model has a systematic blind spot or bias that causes it to generate an unsupported claim in the first place, that same blind spot may cause it to also judge that claim as supported, since the same underlying weaknesses apply to both tasks. This is a milder version of the "LLM judging LLM" concern covered formally in Chapter 16. A stronger design uses a different model (or at minimum a very differently-scoped, adversarially-worded judging prompt) for Tier 2, and periodically validates the judge's own accuracy against a small human-labeled sample, treating the judge itself as a component requiring its own evaluation, not an infallible oracle.

**Scenario-based:**

**Q: After deploying this hallucination detection pipeline, Tier 2 confirmation rate (the fraction of Tier-1-flagged claims that Tier 2 confirms are truly unsupported) is very low — most flagged claims turn out to be fine. What does this tell you, and what would you change?**
A: A low confirmation rate indicates Tier 1's threshold is too aggressive, generating many false positives — flagging claims that are actually well-supported but happen to have lower embedding similarity to their source (perhaps due to paraphrasing, or Hinglish-English translation effects consistent with this project's known embedding weaknesses from Chapter 7 Topic 2). This is both a cost problem (unnecessary Tier 2 calls) and a latency problem (unnecessary customer-facing delay). The fix is re-tuning the Tier 1 threshold against a labeled evaluation set, likely raising the similarity threshold required to trigger a flag (making Tier 1 less trigger-happy) — validated by confirming this reduces Tier 2 call volume without meaningfully increasing the false-negative rate (genuinely unsupported claims slipping through Tier 1 unflagged).

---

### 9. Hidden Concepts and Prerequisites

- **Natural Language Inference (NLI) as a distinct, named ML task**: entailment/contradiction/neutral classification is a long-studied NLP task with dedicated datasets (SNLI, MultiNLI) and pretrained models — worth knowing this exists as a named field, since it directly informs the middle-tier alternative discussed in Section 6, rather than being an ad hoc invention specific to RAG.
- **Hallucination detection connects forward to RAGAS's Faithfulness metric** (Chapter 14): RAGAS formalizes exactly this claim-decomposition-and-entailment-checking pattern into a standardized, automatable metric — this topic's manual two-tier pipeline is the hand-built precursor to adopting a more standardized evaluation framework later.
- **The claim-decomposition step is itself a non-trivial NLP problem**: reliably breaking a generated answer into discrete, checkable factual claims (rather than one holistic blob) is harder than it looks — ambiguous claim boundaries, compound sentences with multiple facts, and implicit claims (things implied but not directly stated) are all edge cases that a naive sentence-splitting approach handles poorly; the structured tool-output approach used in this project's code (having the model itself declare its claims via the tool schema) sidesteps much of this difficulty by making claim decomposition part of generation rather than a separate post-hoc parsing step.

---

### 10. Revision Summary

> Hallucination detection moves beyond prompting ("only use the provided context") into structural, post-hoc verification — because instruction-following is probabilistic, not guaranteed. The technique: decompose the generated answer into discrete claims (via structured tool output, extending Topic 2's citation schema), map each claim to its cited source, and check entailment using a two-tier approach — cheap embedding similarity as a first-pass filter (which cannot reliably distinguish support from contradiction, only flag likely problems), escalating to expensive but reliable LLM-as-judge checking only for flagged claims. This tiering balances cost against reliability at this project's production volume. Hallucination detection specifically targets faithfulness (Topic 3) — it correctly reports "supported by context" even when that context is itself wrong (a correctness failure) or off-topic (a relevance failure), so it must not be conflated with those other checks. For a regulated financial domain, synchronous blocking of unverified answers, with logging and human-review routing, is the more defensible default over asynchronous auditing alone.

---
