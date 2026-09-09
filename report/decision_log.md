# Decision Log — Hiver AI Support Agent

15 non-obvious decisions made during this project, with rationale.

---

1. **Chose AppleSupport as the target brand**
   Largest tweet volume in the dataset (~300K conversations), widest variety of intents, and closest to a software/SaaS support pattern relevant to HiverHQ context.

2. **Used Groq (Llama 3.1 70B) instead of GPT-4o**
   Groq's free tier offers 14,400 req/day with Llama 3.1 70B — sufficient quality for classification and drafting. GPT-4o costs money. Mistral was set as a fallback.

3. **Chose 10 intents, not fewer or more**
   8 is too coarse (billing + bug too different). 15+ leads to confusion between near-duplicate intents (e.g., `slow_app` vs `app_crash`). 10 gives good coverage with enough separation. Validated by keyword clustering.

4. **Stratified sampling for golden eval set**
   ~20 examples per intent rather than random sampling ensures all intents are evaluatable. Random sampling would produce ~0 `account_suspension` examples since it's rare.

5. **Used SetFit instead of full fine-tuning**
   SetFit (few-shot fine-tuning of sentence transformers) achieves ~85% of full fine-tune quality with only 8–64 examples per class. We only have 200 labelled examples — full fine-tuning would overfit badly.

6. **ChromaDB locally, Qdrant in production**
   ChromaDB has zero setup for local dev but isn't suitable for cloud deployment (no managed hosting). Qdrant Cloud's free tier (1GB) is production-ready with persistent storage.

7. **Stored only (customer_message, brand_reply) pairs in vector DB, not full threads**
   Full threads add noise and context that doesn't help retrieval. The (message, reply) pair is the minimal unit that captures resolution style.

8. **Escalation as a rule + LLM hybrid, not pure ML**
   Billing and account suspension always escalate (deterministic rules). High anger + low confidence uses LLM scoring. Pure ML escalation classifier would require ground truth escalation labels we don't have at scale.

9. **SQLite instead of PostgreSQL**
   Zero external dependency, ships with Python, sufficient for logging 10K+ conversations. Postgres adds deployment complexity for no benefit at this scale. Easy to migrate later.

10. **Sentence-transformers running locally (not API)**
    `all-MiniLM-L6-v2` runs in ~100ms on CPU. No API cost, no rate limits, no latency overhead. The model is 90MB and fits in Render's free tier memory.

11. **LLM-as-judge uses a rubric with 4 dimensions** (relevance, groundedness, tone, resolution)
    Single score LLM judges are unreliable. 4-dimension rubric catches different failure modes independently and gives interpretable feedback.

12. **Used Render (not Railway or Fly.io) for backend**
    Render free tier has 750 compute hours/month vs Railway's $5 credit cap. Downside: cold starts after 15 min idle. Acceptable for demo/assignment.

13. **Vite for frontend instead of Create React App**
    CRA is deprecated. Vite is Hiver's actual stack choice, has faster HMR, and smaller production bundles.

14. **Embedded reply templates per intent (fallback)**
    If RAG retrieval confidence is low (<0.5), fall back to intent-specific template + LLM polish rather than hallucinating. Prevents confident-but-wrong replies.

15. **Sampled 200 examples not 250**
    250 is the upper bound. 200 gives enough statistical power (±7% margin of error at 95% confidence) while being labellable in ~3 hours vs ~4 hours.
