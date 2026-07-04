# Chapter 7: Search In-Depth

## Topic 9: Evaluating Retrieval — Recall@K, MRR, NDCG@K

---

### 1. Concept, Intuition, and Why It Exists

**The gap this topic closes:**

- Every single design decision across Topics 1-8 has ended with some version of the same sentence: "this should be validated against a labeled evaluation set, not chosen by intuition" — k1/b in BM25 (Topic 1), the k constant in RRF (Topic 4), lambda in MMR (Topic 6), whether reranking is worth its cost (Topic 7), how aggressively to filter (Topic 8)
- This topic is where that repeated deferral finally gets resolved — it builds the actual evaluation harness every prior topic assumed would eventually exist
- Without this, every comparison made so far in this chapter (BM25 vs dense, hybrid vs individual retrievers, MMR vs plain top-k) has been *qualitative* — "this result set looks more diverse" or "this seems to handle Hinglish better" — genuinely useful for building intuition, but not a substitute for a number you can track over time and defend in a design review

**What "evaluating retrieval" actually requires:**

- A labeled evaluation set: a collection of (query, set of truly relevant documents) pairs — for this project, this means (customer email or representative query, correct policy chunk ID(s)) pairs
- A set of metrics that turn "did the retrieval system return the right documents, in a good order" into comparable numbers
- A consistent process for running every retrieval configuration (BM25-only, dense-only, hybrid, hybrid+MMR, hybrid+rerank) against the same evaluation set, so comparisons are apples-to-apples

**The three metrics this topic covers, and what distinct question each one answers:**

- **Recall@K**: out of all the truly relevant documents for this query, what fraction appear somewhere in the top-K results? Answers: "did we find the right stuff at all, within a reasonable list length?"
- **MRR (Mean Reciprocal Rank)**: for queries with one correct answer, how high up the ranked list did the first correct result appear, averaged across all queries? Answers: "when there's one right answer, how quickly does the user find it?"
- **NDCG@K (Normalized Discounted Cumulative Gain)**: accounting for the fact that some relevant documents are more relevant than others, and that position in the ranking matters (a relevant document at rank 1 is worth more than the same document at rank 5), what is the overall quality of the ranking? Answers: "is the *whole ordering* good, not just whether relevant stuff appears somewhere in it?"

---

### 2. Internal Working — Step by Step

**Recall@K — the formula and its mechanics:**

```text
Recall@K = (number of relevant documents appearing in top-K results) / (total number of relevant documents for this query)

Example: a query has 2 truly relevant chunks in the corpus. The top-5 retrieved
results contain 1 of those 2 relevant chunks.
Recall@5 = 1 / 2 = 0.5
```

- Averaged across all queries in the evaluation set to get a single overall Recall@K number for a given retrieval configuration
- Recall@K is monotonically non-decreasing as K grows — Recall@20 can never be lower than Recall@5, since a larger result list can only contain the same or more relevant documents
- This is why Recall@K is always reported *for a specific K* — Recall@5 and Recall@20 measure genuinely different things, and K should match the actual number of chunks that will realistically be passed to the generation layer

**MRR — the formula and its mechanics:**

```text
Reciprocal Rank (single query) = 1 / (rank position of the FIRST relevant document)

If no relevant document appears in the returned results at all, Reciprocal Rank = 0

MRR = average of Reciprocal Rank across all queries in the evaluation set
```

- Example: for a query, the first genuinely relevant document appears at rank 3 in the results. Reciprocal Rank = 1/3 = 0.333
- MRR is most meaningful when each query has essentially one clearly correct answer (or one answer that matters most) — for this project, a query like "what is my FD reference number's status" has a single correct customer record, making MRR directly interpretable
- MRR is less informative for queries with several equally valid relevant documents, since it only cares about the position of the *first* one, ignoring how the rest of the list performs — this is exactly the gap NDCG@K fills

**NDCG@K — the formula and its mechanics, built up piece by piece:**

```text
Step 1 -- Relevance grades, not just binary relevant/irrelevant:
  Each document in the evaluation set gets a graded relevance score, e.g.:
    2 = highly relevant (directly answers the query)
    1 = partially relevant (related but not a complete answer)
    0 = not relevant

Step 2 -- DCG (Discounted Cumulative Gain) at rank K:
  DCG@K = sum over i=1 to K of:  relevance_i / log2(i + 1)

  The log2(i+1) term is the "discount" -- a relevant document at rank 1
  contributes almost its full relevance score; the same document at rank 10
  contributes much less, because log2(11) is much larger than log2(2).

Step 3 -- IDCG (Ideal DCG): the DCG you would get if the results were
  ranked in the PERFECT order (most relevant document first, down to least
  relevant) -- this is the maximum possible DCG@K for this specific query's
  relevance labels.

Step 4 -- NDCG@K = DCG@K / IDCG@K

  Normalizing by the ideal score makes NDCG@K comparable ACROSS different
  queries, even queries with different numbers of relevant documents or
  different relevance grade distributions -- NDCG@K is always in [0, 1],
  where 1.0 means the retrieval system achieved the perfect possible ordering.
```

- Worked example: a query has relevance grades [2, 0, 1] for the top-3 retrieved documents (in that order — first result is highly relevant, second is irrelevant, third is partially relevant)
  - `DCG@3 = 2/log2(2) + 0/log2(3) + 1/log2(4) = 2/1 + 0/1.585 + 1/2 = 2.0 + 0 + 0.5 = 2.5`
  - The ideal order for these same 3 documents would be [2, 1, 0]: `IDCG@3 = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2.0 + 0.631 + 0 = 2.631`
  - `NDCG@3 = 2.5 / 2.631 = 0.950`
- This single number captures both "did we find relevant stuff" (like Recall) and "is it in the right order, weighted by how relevant it actually is" (which neither Recall@K nor MRR captures)

---

### 3. How It Is Implemented in This Project

**Building the labeled evaluation set — the actual prerequisite work this topic requires:**

- For this project's 17-page knowledge base, a realistic evaluation set consists of representative queries (drawn from real customer email patterns — both English and Hinglish, matching the corpus's 64.4% Hinglish composition) paired with the specific chunk ID(s) that genuinely answer each query
- Graded relevance (for NDCG@K) requires going one step further than binary relevant/not-relevant: for each query, labeling not just *which* chunks are relevant but *how* relevant each one is (2 = directly answers it, 1 = related context, 0 = irrelevant)
- This labeling work is manual and must be done carefully — an evaluation set built from arbitrary or low-quality labels produces metrics that look precise but measure nothing meaningful, a classic case of false rigor

**Code pattern — the three metrics implemented from scratch, run against this project's knowledge base:**

```python
import math

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """retrieved_ids: ranked list of document IDs returned by the retrieval system
    relevant_ids: the SET of truly relevant document IDs for this query (ground truth)"""
    top_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """Returns 1/rank of the FIRST relevant document found, or 0 if none found."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: list, relevance_grades: dict, k: int) -> float:
    """relevance_grades: dict mapping doc_id -> graded relevance (0, 1, 2, ...)
    Documents not in relevance_grades are treated as relevance 0."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_grades.get(doc_id, 0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list, relevance_grades: dict, k: int) -> float:
    """Normalizes DCG@K by the ideal DCG@K (perfect ordering of the same relevance grades)."""
    actual_dcg = dcg_at_k(retrieved_ids, relevance_grades, k)
    # Ideal ordering: sort all graded documents by relevance, descending
    ideal_order = sorted(relevance_grades.keys(), key=lambda d: relevance_grades[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, relevance_grades, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg
```

**Running the same evaluation set across every retrieval configuration built in this chapter — the actual point of this topic:**

- BM25-only (Topic 1) vs Dense-only (Topic 2) vs Hybrid RRF (Topic 4) vs Hybrid+MMR (Topic 6) vs Hybrid+Rerank (Topic 7): every one of these configurations gets run against the identical evaluation set, producing a Recall@K, MRR, and NDCG@K number for each
- This is the mechanism that finally turns every "should be validated" deferral from Topics 1-8 into an actual answer, and is exactly what a Lead-level retrieval system design review would expect to see before any of those configurations is adopted in production

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

**The evaluation set is only as good as its labels — and labeling is the actual bottleneck, not the metric math:**

- Computing Recall@K, MRR, and NDCG@K is straightforward once ground truth exists; producing trustworthy ground truth for a 64.4% Hinglish, ambiguous, real-world email corpus is the genuinely hard and time-consuming part
- A common failure: building an evaluation set too small (5-10 queries) to be statistically meaningful, then over-interpreting small metric differences between configurations that are actually just noise — for this project's scale, dozens to low hundreds of labeled queries, covering the actual proportional mix of query types (Hinglish vs English, exact-reference-number lookups vs general policy questions), is a more defensible starting point than a handful of hand-picked "nice" examples

**Recall@K, MRR, and NDCG@K can disagree with each other — and when they do, that disagreement is informative, not a bug:**

- A retrieval configuration might have high Recall@10 (the relevant documents are somewhere in the top 10) but poor MRR (the *first* relevant document is buried at rank 8, not rank 1) — this tells you the system finds the right information eventually, but a user or downstream LLM reading only the top few results might still miss it
- A configuration might have good MRR but poor NDCG@K if it gets the single most relevant document to rank 1 correctly, but badly orders the remaining, still-somewhat-relevant documents behind it — NDCG@K cares about the whole ranking, MRR only cares about the first hit
- Reporting all three, not just one, is what surfaces this kind of nuance — a Lead-level evaluation practice never relies on a single aggregate metric

**Statistical significance and evaluation set drift:**

- As this project's corpus and query distribution evolve (Chapter 6 Topic 10's "honest migration trigger" discussion applies equally here), an evaluation set built once and never revisited becomes stale — it stops representing the actual query distribution the system faces in production
- A mature practice re-samples and re-labels the evaluation set periodically, and explicitly version-controls it, so metric comparisons over time are comparing against a consistent (or explicitly-versioned) ground truth, not silently drifting

**Cost and effort:**

- Unlike every other topic in this chapter, this topic's primary cost is *human labeling time*, not compute or API cost — worth explicitly flagging in any project planning discussion, since it's easy to under-budget the labeling effort required to make the rest of this chapter's design decisions genuinely evidence-based rather than qualitative

**Monitoring in production — evaluation doesn't stop once a configuration is deployed:**

- Production monitoring (implicit relevance signals: did the generated answer get a thumbs-up, did the customer's follow-up email indicate the first answer was wrong, did a human reviewer flag the retrieved context as insufficient) is a weaker but continuously-available substitute for full offline evaluation-set metrics, and should feed back into periodically refreshing the labeled evaluation set itself

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

**How many queries does the evaluation set need before metric differences are trustworthy?**

- There's no universal magic number — it depends on how large a metric difference you need to detect reliably; a very small evaluation set can only reliably detect very large differences between configurations, while subtle improvements (e.g. MMR's diversity benefit, which may only matter for a specific subset of redundancy-prone queries) require proportionally more evaluation queries covering that specific pattern to detect confidently
- A practical compromise for this project: start with tens of queries covering the clearest, highest-value patterns (exact reference lookups, common Hinglish phrasings, the known redundancy pattern MMR targets), and grow the set over time as specific configuration decisions need finer-grained validation

**Binary relevance (for Recall@K, MRR) vs graded relevance (required for NDCG@K) — is the extra labeling effort worth it?**

- Binary labeling is faster and often sufficient for early-stage comparisons (is hybrid clearly better than BM25-only, yes or no) — graded relevance's extra nuance mainly pays off once configurations are close enough that ranking quality, not just presence/absence, is the differentiator
- For this project, a reasonable sequencing: start with binary relevance and Recall@K/MRR to make the big, obvious calls (hybrid vs single-method retrieval) quickly, then invest in graded labels and NDCG@K once comparing more similar configurations (e.g. reranking vs no reranking) where ordering nuance actually matters

**Which K to report Recall@K and NDCG@K at?**

- K should match the actual number of chunks the generation layer (Chapter 8) will realistically receive — reporting Recall@50 is not useful if only the top-5 chunks are ever passed to the LLM; the metric should reflect the system's real operating point, not an arbitrarily generous cutoff that makes numbers look better than the deployed behavior

---

### 6. Alternatives and When to Use Each

**Recall@K:**
- Best for: early-stage, coarse comparisons — "does this retrieval configuration find the right information at all, within a reasonable list length" — the simplest metric to compute and explain to non-technical stakeholders
- Use when: the priority question is coverage, not fine-grained ranking quality

**MRR:**
- Best for: query distributions with one clearly correct answer per query, such as exact-reference-number lookups or single-fact FAQ questions
- Use when: user experience specifically depends on the correct answer appearing as close to the top as possible, and secondary relevant documents matter less

**NDCG@K:**
- Best for: query distributions where multiple documents can be relevant to varying degrees, and the full ordering of the result set matters — the most information-rich of the three metrics, at the cost of requiring graded relevance labels
- Use when: comparing configurations that are already both reasonably good at Recall@K and MRR, and the remaining question is which one produces the genuinely better-ordered result set

**Precision@K (a related metric not detailed above, worth knowing):**
- Measures what fraction of the top-K results are actually relevant, rather than what fraction of all relevant documents were found — the complementary question to Recall@K
- Use when: the cost of irrelevant results in the top-K matters directly (e.g. context window budget in Chapter 8's generation layer, where every irrelevant chunk consumes space a relevant one could have used)

**RAGAS-style end-to-end metrics (forward reference to Chapter 14 — RAG Evaluation End-to-End):**
- Metrics like Context Precision and Context Recall extend these classical IR metrics specifically to the RAG use case, and Faithfulness/Answer Relevancy go further to evaluate the *generated answer*, not just retrieval
- Use when: evaluating the full pipeline end-to-end, not just the retrieval stage in isolation — this topic's metrics are the retrieval-specific foundation those end-to-end metrics build on

---

### 7. Common Mistakes and Production Failures

- **Building an evaluation set too small or too hand-picked to be representative** — produces metrics that look precise but don't generalize to real production query patterns, especially risky for this project's Hinglish-heavy, ambiguous corpus where edge cases are common, not rare
- **Comparing metrics computed at different K values as if they were comparable** — Recall@5 from one experiment and Recall@10 from another are not directly comparable; always fix K consistently across configurations being compared
- **Relying on a single metric to make a configuration decision** — as discussed in Section 4, Recall@K, MRR, and NDCG@K can disagree, and each surfaces a different failure mode; a Lead-level evaluation practice reports and reasons about all three together
- **Letting the evaluation set go stale as the corpus and query distribution evolve** — a set built once and never revisited silently loses relevance to what the system actually faces in production
- **Treating small numeric differences between configurations as meaningful without checking statistical significance or evaluation set size** — a 0.02 difference in NDCG@K on a 10-query evaluation set is very likely noise, not a genuine signal
- **Using binary relevance labels to compute NDCG@K without actually grading relevance** — NDCG@K's entire value proposition depends on graded relevance; computing it with only binary labels collapses much of its advantage over Recall@K
- **Skipping this topic entirely and shipping retrieval configuration decisions based on qualitative impression alone** — the central anti-pattern this whole topic exists to correct, given how many times Topics 1-8 explicitly deferred a decision to "future evaluation"

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: Explain Recall@K and why it's reported at a specific K rather than as a single overall number.**

A: Recall@K measures the fraction of all truly relevant documents for a query that appear within the top-K retrieved results. It's reported at a specific K because Recall@K is monotonically non-decreasing as K grows — a longer result list can only find the same or more relevant documents, never fewer — so Recall@50 will always be greater than or equal to Recall@5 for the same system. Reporting it at the K that matches the system's actual operating point (e.g. the number of chunks genuinely passed to the generation layer) is what makes the metric meaningful rather than artificially inflated by an unrealistically generous cutoff.

**Q: What does MRR measure, and what kind of query distribution is it best suited for?**

A: MRR (Mean Reciprocal Rank) measures, on average across queries, the reciprocal of the rank position at which the *first* relevant document appears — a relevant document at rank 1 contributes 1.0, at rank 2 contributes 0.5, and so on, with 0 if no relevant document is found at all. It's best suited for query distributions where each query has essentially one clearly correct answer, since it only cares about how quickly that first correct answer is found and ignores how the rest of the ranked list performs.

**Intermediate:**

**Q: Why does NDCG@K require graded relevance labels, and what specific failure mode does this let it catch that Recall@K and MRR cannot?**

A: NDCG@K weighs each retrieved document's contribution both by its position (via a logarithmic discount — documents ranked higher contribute more) and by its graded relevance (not just relevant/irrelevant, but how relevant — a 2 for a directly-answering document versus a 1 for merely related context). This lets NDCG@K distinguish between a ranking that puts a highly-relevant document first and a mediocre-but-related one second, versus a ranking that puts a mediocre-but-related document first and the highly-relevant one second — both of these might score identically on Recall@K (both relevant documents are present in the top-K) and even similarly on MRR (if both have a "relevant" document at rank 1 under binary labeling), but NDCG@K correctly penalizes the second ordering for burying the more valuable document lower in the list.

**Q: You're comparing hybrid RRF (Topic 4) against hybrid RRF plus reranking (Topic 7) on this project's evaluation set, and Recall@10 is identical between the two configurations, but NDCG@10 for the reranked version is meaningfully higher. What does this tell you, and is reranking worth adopting based on this result?**

A: Identical Recall@10 means both configurations are finding the same set of relevant documents somewhere within the top 10 — reranking isn't improving *what* gets found, only *how it's ordered*. Higher NDCG@10 for the reranked version means it's placing the more relevant documents higher in that same set — the ordering is genuinely better, which matters because downstream, only a handful of the top results (not all 10) are typically passed to the generation layer, so a better ordering within the top 10 directly translates to better content reaching the LLM. Whether this is worth adopting depends on weighing this measured NDCG@10 improvement against reranking's added latency and cost (Topic 7) — the evaluation confirms reranking is doing something real and positive, but the final adoption decision is a cost-benefit call informed by, not fully determined by, this one metric result.

**Advanced:**

**Q: Design the evaluation methodology you would use to validate every major retrieval design decision made across this entire chapter (BM25 parameters, RRF's k constant, MMR's lambda, whether to rerank, filter selectivity) before finalizing this project's production retrieval configuration.**

A: First, build a labeled evaluation set with graded relevance, deliberately sampling queries proportional to the corpus's real query distribution — specifically ensuring adequate representation of Hinglish queries (64.4% of the corpus), exact-reference-number lookups, and known-redundant topics (like the premature withdrawal penalty restated across FAQ/Policy/SOP) since these are the specific patterns earlier topics identified as differentiating. Second, establish a fixed K matching the system's real operating point for the generation layer. Third, run a systematic sweep: BM25-only at several k1/b combinations, dense-only, hybrid RRF at several k values, hybrid+MMR at several lambda values, hybrid+rerank, and combinations of these, computing Recall@K, MRR, and NDCG@K for every configuration against the identical evaluation set. Fourth, since some parameters (BM25's k1/b, RRF's k, MMR's lambda) interact with each other rather than being independently optimal, I would not tune them in isolation — a reasonable practical approach is a staged search: fix reasonable defaults for the parameters least likely to interact strongly, tune the ones with the largest expected impact first (informed by which topics measured the largest problems — e.g. the Topic 2 finding that dense retrieval alone has only a 0.0393 discrimination gap suggests hybrid's relative weighting matters more than fine BM25 tuning). Fifth, report all three metrics together for the finalist configurations, and make the final production choice based on the full picture — including latency and cost from Topic 7 — not metric numbers alone.

**Q: A colleague argues that since the evaluation set is manually labeled and therefore subjective, these metrics shouldn't be trusted for making production decisions, and intuition-based qualitative review is more reliable. How do you respond?**

A: Manual labeling does introduce subjectivity, and that's a real, legitimate limitation worth acknowledging explicitly — but the alternative isn't actually more objective, it's less transparent about its own subjectivity. Qualitative review by reading a handful of example outputs is also a human judgment process, just one that isn't systematically recorded, isn't reproducible, isn't comparable across configurations at scale, and is far more susceptible to being swayed by a few memorable examples rather than the full query distribution. A labeled evaluation set, even an imperfect one, makes the judgment criteria explicit, auditable, and consistently re-applicable across every configuration being compared — the same standard is applied identically to BM25-only and to hybrid+rerank, which qualitative review cannot guarantee. The right response to labeling subjectivity concerns isn't to abandon systematic evaluation, but to improve labeling quality (multiple labelers, documented labeling guidelines, periodic label review) and to combine offline evaluation-set metrics with the production monitoring signals discussed in Section 4, rather than relying on either qualitative impression or a single static evaluation set alone.

**Scenario-based:**

**Q: After finalizing a retrieval configuration based on strong Recall@K, MRR, and NDCG@K numbers from your evaluation set, production users report that answers about a newly-launched product consistently feel poor, even though the evaluation metrics looked good at launch. Diagnose this discrepancy.**

A: This is very likely an evaluation set staleness problem, exactly the risk flagged in Section 4 — the evaluation set was built and labeled before the new product existed, so it contains zero queries or ground-truth labels covering that product, meaning the strong offline metrics never actually measured retrieval quality for this specific new content at all. The evaluation numbers weren't wrong, they were simply answering a question ("how good is retrieval for the query patterns that existed when the evaluation set was built") that no longer fully represents current production reality. The fix: treat evaluation set maintenance as an ongoing process tied to corpus and product changes, not a one-time setup step — every significant new product launch or knowledge base addition should trigger new labeled queries covering that addition before trusting the existing metrics to represent current system quality, echoing the same "honest migration trigger" discipline from Chapter 6 Topic 10 applied specifically to evaluation data rather than infrastructure.

---

### 9. Hidden Concepts and Prerequisites

**The relationship between offline evaluation metrics and actual user satisfaction is itself an assumption, not a guarantee:**

- Recall@K, MRR, and NDCG@K all measure how well retrieval matches *labeled ground truth* — but ground truth relevance labels are a human's judgment about what *should* be relevant, which may not perfectly predict what actually produces a satisfying generated answer or a satisfied customer downstream
- This is precisely why RAGAS-style end-to-end metrics (forward reference to Chapter 14) exist as a complementary, not a replacement, layer of evaluation — they get closer to measuring the thing that ultimately matters (a good generated answer), at the cost of being more expensive to compute and harder to attribute specifically to the retrieval stage versus the generation stage

**Metric gaming and Goodhart's Law apply here as much as anywhere else in ML:**

- Once Recall@K, MRR, or NDCG@K becomes a target that configurations are explicitly optimized against, there's a risk of overfitting retrieval tuning specifically to the quirks of the evaluation set rather than genuinely improving retrieval for the full production query distribution — a reminder that the evaluation set itself needs to be representative and periodically refreshed (Section 4), not treated as a fixed, permanent ground truth to chase indefinitely

**Inter-annotator agreement is a real, measurable concept worth knowing by name:**

- When more than one person labels relevance for the same queries, their agreement rate (often measured with Cohen's Kappa or similar statistics) quantifies how subjective or consistent the labeling task actually is
- Low inter-annotator agreement on a specific query type is itself useful information — it suggests that query type has inherently ambiguous relevance (multiple genuinely defensible correct answers), which should inform how strictly metrics for that query type are interpreted, rather than assuming all evaluation queries are equally clear-cut

---

### 10. Revision Summary

> This topic builds the evaluation harness that every earlier topic in this chapter implicitly assumed would eventually exist. Recall@K measures whether relevant documents are found at all within a result list of length K; MRR measures how quickly the first relevant document appears, best suited to single-correct-answer query distributions; NDCG@K measures the quality of the full ranking, accounting for both position (via logarithmic discounting) and graded relevance, making it the most information-rich but also the most labeling-intensive of the three. None of these metrics matter without a genuinely representative, carefully labeled evaluation set — for this project, that means queries proportionally covering the 64.4% Hinglish composition, exact-reference lookups, and the corpus's known redundancy patterns, not a small set of hand-picked convenient examples. Every retrieval configuration decision deferred throughout this chapter (BM25's k1/b, RRF's k constant, MMR's lambda, whether reranking or metadata filtering are worth their cost) should be resolved by running that configuration against this same evaluation set and comparing Recall@K, MRR, and NDCG@K together, since they can disagree and each surfaces a different failure mode. Evaluation sets go stale as the corpus and query distribution evolve — periodic re-labeling, particularly after new product launches or knowledge base additions, is a required ongoing practice, not a one-time setup task.

---
