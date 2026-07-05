# Chapter 8: RAG Generation — The Answer Layer

## Topic 6: Multi-Turn RAG — Conversation History + Retrieval

---

### 1. Concept, Intuition, and Why It Exists

- Every topic so far in this chapter has implicitly assumed a single-turn interaction: one customer email in, one retrieved context, one generated answer. Real customer interactions are often multi-turn — a customer asks a follow-up question, references something said earlier, or clarifies an ambiguous initial request.
- Multi-turn RAG exists because retrieval itself needs to account for conversation history, not just the latest message in isolation. "What about for senior citizens?" as a follow-up to "what is the FD premature withdrawal penalty?" is meaningless as a standalone retrieval query — it has no reference to FDs, penalties, or withdrawal at all. Without conversation-aware retrieval, this follow-up would retrieve poorly or not at all.
- The core new problem this topic introduces beyond single-turn RAG: **query contextualization** — reformulating a follow-up turn into a retrieval query that carries forward the necessary context from earlier turns, before retrieval even runs. This is a genuinely different problem from anything covered in Chapter 7, which assumed each query arrived already well-formed and self-contained.

---

### 2. Internal Working — Step by Step

1. **Conversation history accumulation**: this project's existing agent loop pattern (`messages = [...]`, appended to across turns, from Chapter 3's `run_agent()`) already provides the mechanical scaffolding — a growing list of role/content dicts representing the full conversation so far.
2. **Query contextualization (the new step)**: before retrieval runs on a new turn, the raw follow-up message must be rewritten into a self-contained query using the conversation history. Two practical approaches:
   - **Rule-based/template contextualization**: prepend the previous turn's topic or key entities to the new query — simple, fast, but brittle for anything beyond straightforward follow-ups.
   - **LLM-based query rewriting**: a dedicated (often small, fast) model call that reads the conversation history and the raw follow-up, and outputs a self-contained, retrieval-ready query — e.g. "What about for senior citizens?" + prior turn about FD withdrawal penalty → "What is the FD premature withdrawal penalty for senior citizens?" This is a direct preview of Topic 7 (Query Rewriting and HyDE), applied specifically to the multi-turn case.
3. **Retrieval using the contextualized query**: once contextualized, retrieval proceeds exactly as in Chapter 7 — hybrid BM25+dense+RRF, reranking, MMR — no changes to the retrieval pipeline itself, only to what query it receives.
4. **Context window management across turns**: as conversation history grows, Topic 1's token budgeting must now account for accumulated prior turns *in addition to* the newly retrieved chunks — a genuinely tighter budget problem than the single-turn case, since history competes with retrieved context for the same fixed window.
5. **Deciding what history to keep**: not every prior turn needs to be sent in full on every subsequent call — summarization, truncation, or selective inclusion of only relevant prior turns are all real strategies, covered in Section 5.

---

### 3. How It Is Implemented in This Project

- This project's `run_agent()` pattern from Chapter 3 already threads `messages` through the loop turn by turn — multi-turn RAG extends this by adding a query-contextualization step immediately before the retrieval call, using the accumulated `messages` list as the source of conversational context.

```python
def contextualize_query(raw_query: str, conversation_history: list, client, model_id: str) -> str:
    """Rewrites a follow-up query into a self-contained retrieval query,
    using conversation history. Uses a fast, cheap model call -- this is
    NOT the main answer-generation call, just a lightweight rewriting step."""

    if not conversation_history:
        return raw_query  # first turn -- nothing to contextualize against

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation_history[-4:]  # last few turns only
    )

    rewrite_prompt = f"""Conversation so far:
{history_text}

Latest customer message: "{raw_query}"

Rewrite the latest message as a single, self-contained question that includes
any necessary context from the conversation above. Output ONLY the rewritten
question, nothing else."""

    response = client.messages.create(
        model=model_id, max_tokens=100,
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    return response.content[0].text.strip()


def multi_turn_rag_turn(raw_query: str, conversation_history: list, client, model_id: str,
                         retrieval_fn, token_budget: int) -> dict:
    """One full turn of multi-turn RAG: contextualize, retrieve, budget, generate."""

    contextualized_query = contextualize_query(raw_query, conversation_history, client, model_id)

    ranked_chunks = retrieval_fn(contextualized_query)  # Chapter 7's full pipeline, unchanged

    # Token budget must now account for accumulated history, not just system prompt
    history_tokens = sum(estimate_tokens(m["content"]) for m in conversation_history)
    available_for_chunks = token_budget - history_tokens

    context_block = build_context_block(ranked_chunks, available_for_chunks)

    return {
        "contextualized_query": contextualized_query,
        "context_block": context_block,
        "history_tokens_used": history_tokens,
    }
```

---

### 4. Real-World Issues, Edge Cases, Debugging, Monitoring, Scaling, Latency, Cost, Security, Deployment

- **Contextualization can fail silently in ambiguous cases**: "what about the other one" as a follow-up to a message discussing two different FD products has genuine ambiguity even for a human reading the transcript — an LLM-based rewriter can guess wrong, producing a confidently-wrong contextualized query that then drives retrieval toward the wrong content entirely, with no obvious signal anything went wrong until the final answer is reviewed.
- **Cost compounds with turn count**: every additional turn requires (a) an extra contextualization model call, (b) a full Chapter 7 retrieval pass, and (c) growing history tokens competing with retrieved context in Topic 1's budget — a 5-turn conversation costs meaningfully more than 5 independent single-turn queries, both in API calls and in the increasing pressure on the fixed context budget.
- **Latency**: contextualization adds a full extra LLM round-trip before retrieval even starts — for a customer-facing synchronous flow, this is added latency on every turn beyond the first, and should use the fastest available model (this project's `claude-haiku-4-5-20251001` is already the fast tier) rather than a larger, slower model for this specific step.
- **History truncation and the "forgotten context" bug**: truncating conversation history (e.g. keeping only the last N turns, as shown in the example code) risks losing genuinely relevant earlier context — a customer who established an important detail in turn 1 (e.g. "I'm a senior citizen") and asks a related question in turn 6 may get an answer that's silently missing that earlier-established context if it fell outside the truncation window.
- **Monitoring**: track contextualization-query-differs-from-raw-query rate (how often the rewriter actually changes anything, vs. passing the raw query through unchanged) as a coarse health signal, and specifically sample and review cases with high conversational complexity (many turns, topic switches) where contextualization is most likely to fail.
- **Security**: conversation history is an additional injection surface — a customer could embed adversarial instructions in an early turn, hoping they persist in context and influence a much later turn's behavior; this extends Chapter 3's existing "treat email content as data, never as commands" principle across the full accumulated history, not just the current turn.
- **Deployment**: multi-turn state (the accumulated `messages` list) must be persisted per-conversation across requests in a real deployment — this introduces session/state management infrastructure that a single-turn system doesn't need (connecting forward to Chapter 10's Memory in Agents topic, which covers this more formally).

---

### 5. Design Decisions, Trade-offs, and Real-Time Dilemmas

- **Full history inclusion vs. truncation vs. summarization**: including the full conversation history every turn is simplest and safest against the "forgotten context" bug, but doesn't scale — token cost and budget pressure grow linearly with turn count, unsustainably for long conversations. Truncating to the last N turns bounds cost but risks losing early, still-relevant context (as in the senior-citizen example above). Summarizing older turns into a compact representation preserves relevant information at lower token cost than full inclusion, at the cost of implementation complexity and a new failure mode (the summary itself losing or distorting important detail). For this project's likely mostly-short conversations (given the short, focused nature of individual FD-related emails per Chapter 1's EDA), truncation with a reasonably generous window is probably sufficient; summarization becomes worth the complexity specifically if longer, more complex support conversations turn out to be common in practice.
- **Rule-based vs. LLM-based contextualization**: rule-based approaches (e.g. always prepend the previous turn's detected topic/entity) are cheap and fast but brittle beyond simple, predictable follow-up patterns. LLM-based rewriting handles a much broader range of conversational patterns correctly but costs an extra model call and adds latency on every turn. Given this project's existing philosophy of using the fast, cheap model tier where possible, LLM-based contextualization with `claude-haiku-4-5-20251001` is a reasonable default, reserving rule-based shortcuts only for detected simple/unambiguous cases if cost or latency become measured bottlenecks.
- **Should contextualization and the final answer-generation call ever be merged into one call to save cost/latency?**: technically possible (ask the model to both contextualize and directly answer in one pass, skipping the separate retrieval step), but this defeats the purpose of RAG entirely for the follow-up turn — the model would be answering from its own parametric knowledge instead of the project's grounded, retrieved, verifiable context, undermining everything Topics 2-4 established. The two-step separation (contextualize, then retrieve-and-generate) is necessary, not just a convenient default.

---

### 6. Alternatives and When to Use Each

- **Single-turn only, no conversation history (this project's baseline through Chapter 7)**: appropriate when each customer interaction genuinely is self-contained (a single email with a single question, no back-and-forth) — likely true for a meaningful fraction of this project's actual email volume, given the corpus's structure.
- **Rule-based contextualization**: appropriate for a constrained, predictable set of common follow-up patterns identified from real production data, where the cost/latency savings over LLM-based rewriting are measured to matter.
- **LLM-based contextualization (this topic's primary recommendation)**: the right default for genuinely open-ended multi-turn conversations, given this project's access to a fast, cheap model tier.
- **Full conversation-state summarization for long conversations**: reserved for cases where truncation is measured to cause real context loss — not needed as a default given this project's likely short-conversation profile.

---

### 7. Common Mistakes and Production Failures

- Running retrieval on the raw follow-up message without any contextualization, silently retrieving poor or irrelevant results for any query that depends on prior conversational context.
- Truncating conversation history without ever validating whether important early context is being lost for genuinely long conversations.
- Forgetting that conversation history itself competes with retrieved context for the same fixed token budget (Topic 1), leading to budget miscalculation as conversations grow.
- Not extending Chapter 3's prompt-injection defense ("treat content as data, not commands") across the full accumulated conversation history, leaving early-turn-embedded adversarial content unguarded in later turns.
- Merging contextualization and generation into a single call in a way that skips genuine retrieval for follow-up turns, silently reverting to non-grounded, unverifiable answers exactly where this chapter's entire architecture (Topics 1-4) was built to prevent that.

---

### 8. Lead-Level Interview Questions

**Basic:**

**Q: Why can't a follow-up question simply be passed to the existing Chapter 7 retrieval pipeline unchanged?**
A: A follow-up question like "what about for senior citizens?" is often not self-contained — it references context established in an earlier turn (e.g. the topic being FD withdrawal penalties) without repeating that context explicitly. Retrieval, which matches query text against document content, would perform poorly or fail entirely on such a fragment, since it has no lexical or semantic connection to the actual topic without the missing context.

**Q: What is query contextualization, and where does it sit in the pipeline relative to retrieval?**
A: Query contextualization rewrites a raw follow-up message into a self-contained, retrieval-ready query using conversation history — it runs *before* retrieval, transforming the input retrieval receives, without changing retrieval's own logic (Chapter 7's hybrid BM25+dense+RRF, reranking, and MMR all remain unchanged; they simply receive a better-formed query).

**Intermediate:**

**Q: How does multi-turn RAG affect Topic 1's token budgeting, and why is this a genuinely new problem compared to single-turn RAG?**
A: In single-turn RAG, the token budget only needs to account for the system prompt and reserved output space, leaving the rest for retrieved chunks. In multi-turn RAG, accumulated conversation history also consumes budget, and that budget grows with every turn — meaning the space available for retrieved context shrinks as a conversation gets longer, requiring either history truncation/summarization or accepting a smaller retrieved-context budget on later turns, a trade-off that doesn't exist in the single-turn case.

**Q: A customer establishes they're a senior citizen in turn 1, then asks an unrelated FD question in turn 6. Your history-truncation window only keeps the last 4 turns. What goes wrong, and how would you fix it?**
A: The senior-citizen detail from turn 1 falls outside the 4-turn truncation window by turn 6, so the contextualized query and the retrieved context for turn 6 have no way to reflect that fact, even though it may be genuinely relevant to the answer (e.g. senior citizens getting a different, better interest rate). A fix requires either a much larger truncation window (with its own cost trade-offs), a summarization step that extracts and persistently carries forward key customer-specific facts (like "senior citizen") regardless of how many turns have passed, or a separate, explicit customer-profile/session-memory mechanism (forward reference to Chapter 10) that tracks durable facts about the customer across the whole conversation, independent of the turn-by-turn history truncation window.

**Advanced:**

**Q: Design the full multi-turn RAG flow for this project, including how it interacts with the existing single-turn agent architecture from Chapter 3.**
A: On each new customer message, check whether conversation history exists for this interaction (session-based, keyed by customer/conversation ID). If it's the first turn, proceed exactly as the existing Chapter 3 `run_agent()` pattern does. If history exists, first call the contextualization step (a fast `claude-haiku-4-5-20251001` call) using the accumulated history to rewrite the raw message into a self-contained query. Pass the contextualized query through Chapter 7's unchanged retrieval pipeline. Compute the token budget accounting for both the system prompt and the (possibly truncated or summarized) conversation history, then build the context block from retrieved chunks within whatever budget remains. Generate the answer via the existing structured tool-call pattern (Topics 2 and 4's citation and claim schema), append both the customer's turn and the assistant's turn to the conversation history for the next turn, and persist that updated history for the session. Verification (Topics 2 and 4) runs identically regardless of turn number — multi-turn RAG changes what goes *into* the pipeline, not the verification logic itself.

**Q: A teammate proposes skipping contextualization and instead always sending the FULL raw conversation history as part of the retrieval query (concatenating all turns together). Evaluate this approach.**
A: This avoids the cost and latency of a separate contextualization call, and avoids the risk of a rewriter guessing wrong about what a follow-up means. But it has real downsides: concatenating full history into a retrieval query dilutes the query's signal — early, now-irrelevant turns can pull retrieval toward stale topics, and BM25/dense retrieval (Chapter 7) aren't designed to weight recency or relevance within a long, multi-topic query string the way a purpose-built contextualization step can. It also doesn't solve the token-budget problem (Topic 1) at the retrieval-query-formation stage, only defers it. A better middle ground, if avoiding a dedicated contextualization call is a genuine cost concern, is a lighter-weight approach — e.g. only including the immediately preceding turn rather than full history, or using cheap heuristics (has the topic likely changed, based on simple keyword overlap) to decide whether contextualization is even needed for a given follow-up, reserving the LLM-based rewrite for cases where it's genuinely ambiguous.

**Scenario-based:**

**Q: In production, you notice multi-turn conversations have measurably worse Recall@K (per Chapter 7 Topic 9's evaluation methodology, now extended to contextualized queries) than single-turn queries. Diagnose.**
A: This points to the contextualization step itself as the likely culprit rather than retrieval — if retrieval performs well on well-formed single-turn queries but poorly specifically on multi-turn follow-ups, the contextualized queries being fed into retrieval are likely poorly formed, not the retrieval pipeline itself. Build a labeled evaluation set specifically for multi-turn contextualization (pairs of conversation-history-plus-follow-up and their ideal, human-written self-contained query rewrite), and directly evaluate the contextualization step's output quality independent of retrieval — comparing the rewriter's actual output against the ideal rewrite reveals whether the rewriter is systematically dropping context, hallucinating context that wasn't there, or something else specific enough to fix, following the same evidence-based diagnostic discipline established throughout Chapter 7.

---

### 9. Hidden Concepts and Prerequisites

- **Coreference resolution as the underlying NLP problem**: query contextualization is fundamentally a coreference resolution problem — determining what pronouns, references, and elliptical phrases ("the other one", "what about", "that") refer to within a conversation. This is a long-studied NLP task independent of RAG or LLMs specifically, worth knowing by name since it clarifies why this problem is genuinely hard in general, not just an implementation detail specific to this project.
- **The relationship between multi-turn RAG and agent memory (forward reference to Chapter 10)**: this topic's conversation-history handling is the short-term memory case; Chapter 10 formalizes the distinction between short-term memory (within one conversation, this topic's scope) and long-term memory (persistent facts about a customer across separate conversations, like the repeat-sender pattern noted in this project's EDA) — this topic is the foundation the more general memory architecture builds on.
- **Conversation state as a genuinely stateful system component**: unlike everything covered in Chapters 4-7 (which operate on a fixed knowledge base and stateless queries), multi-turn RAG introduces real per-session state that must be correctly created, persisted, retrieved, and eventually expired — a different category of engineering concern (session management, storage, concurrency) than the largely stateless retrieval and generation logic covered elsewhere in this project.

---

### 10. Revision Summary

> Multi-turn RAG extends single-turn RAG (Topics 1-5) to handle conversations where a follow-up query depends on earlier turns' context. The core new step is query contextualization — rewriting a raw follow-up into a self-contained, retrieval-ready query using conversation history, via a fast LLM call (this project's `claude-haiku-4-5-20251001`) before Chapter 7's unchanged retrieval pipeline runs. Conversation history competes with retrieved context for Topic 1's fixed token budget, growing the budget pressure with every additional turn — requiring a deliberate truncation, summarization, or full-inclusion policy, each with real trade-offs between cost, context-loss risk, and implementation complexity. Chapter 3's prompt-injection defense must extend across the full accumulated history, not just the current turn. This topic's short-term, within-conversation memory handling is the foundation for Chapter 10's more general agent memory architecture, which adds long-term, cross-conversation memory on top.

---
