# clmemory-multimodal

An independent, source-preserving baseline for the **AML Multimodal Memory** open-source track. Version: `multimodal-caption-hybrid-v1.0.1`.

The service writes ordered text/image memories and retrieves original evidence. It does not generate final answers. This is an engineering baseline; no official Smoke, Full, leaderboard score, or competitive advantage is claimed.

## Method

1. Validate the official ordered `ContentPart[]` format and retain the complete original parts, role, timestamp, session, and source order in SQLite.
2. Describe each image with **GPT-4o-mini**, temperature 0, high image detail, at most 768 output tokens. Captions describe visible evidence and are used only for retrieval. Incomplete/refused captions fail the operation.
3. Index source text plus captions using **text-embedding-v4**, 1024 dimensions, document/query modes, L2 normalization. Texts are split into consecutive 6,000-character embedding spans without discarding source text; batches contain at most 10 spans. Dense similarity is the maximum cosine similarity across document/query spans.
4. Rank the same user's messages with BM25 (`k1=1.2`, `b=0.75`) and dense similarity, then combine ranks with equal-weight reciprocal rank fusion (`k=60`). Zero lexical matches do not add a lexical score. Ties preserve insertion order.
5. Return up to `top_k` original messages, each with a clearly marked source-metadata text header followed by the unchanged original parts. Generated captions are not returned. Skip complete messages that would exceed the response's 30 MiB decoded-image budget. No final answer, model reranker, benchmark-specific rule, or learned filter is used.

Pure image queries use the same captioning path. Text-only Add/query strings are also accepted, as permitted by the full documentation; a returned source string is wrapped in a text part without rewriting it. Top-level multiple-choice `options` are accepted and validated but not added to the retrieval query. There is no gold-answer input. Version 1.0.1 adds this input compatibility; provider prompts, embeddings, ranking, and caches remain compatible with 1.0.0.

## Attribution and scope

Implementation by Chang-Heng Yi, Nanjing University / Cllab. This standalone implementation uses the established BM25 family (Robertson and colleagues) and reciprocal rank fusion (Cormack, Clarke, and Buettcher), together with standard caption-based retrieval and dense cosine search. These algorithms are not claimed as new inventions. Runtime dependencies retain their own licenses; no upstream memory-system source is vendored.

The project belongs to the same team as `clmemory` (Textual) and `clmemory-coding`. It is a separate Multimodal baseline with a separate database, cache, service credential, source repository, and deployment path. It does not import their benchmark data, experimental trajectories, method results, or official evaluation results, and does not claim to reproduce another system's leaderboard deployment.

Contracts checked on 2026-10-29:

- [AML API Guide](https://agentmemoryleaderboard.ai/api-guide)
- [AML participation requirements](https://agentmemoryleaderboard.ai/rules)
- [AML full documentation](https://agentmemoryleaderboard.ai/docs)
- [OpenAI image input documentation](https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat)

## Run

Python 3.13 was used for validation. Install in a dedicated environment:

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements-lock.txt
python -m unittest -v test_memory
python mm_memory.py --config /absolute/private/config.json --port 18124
```

Copy `config.example.json` to a private location and provide absolute paths to separate service and provider key files. The vision endpoint must support GPT-4o-mini image inputs through Chat Completions. The embedding endpoint must support the DashScope native `text-embedding-v4` API. Provider credentials must remain outside this repository.

The process binds only to loopback. Put a TLS reverse proxy in front of it. Use one service process; additional processes or Uvicorn workers are not supported by this release. The deployed registration instance uses the team's existing provider accounts through `api.openai-proxy.org` for GPT-4o-mini and an Alibaba Cloud Model Studio workspace endpoint for v4. Provider identity is disclosed separately from the requested model; the observed vision response was `gpt-4o-mini-2024-07-18`.

## API

`GET /health` returns version and implementation SHA-256. `POST /add` and `POST /search` require `Authorization: Token <dedicated service key>`.

```json
{
  "request_id": "request-1",
  "user_id": "user-1",
  "session_id": "session-1",
  "messages": [{
    "role": "user",
    "timestamp": 1704067200000,
    "content": [
      {"type": "text", "text": "This item is in drawer 37."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

```json
{
  "user_id": "user-1",
  "query": [{"type": "text", "text": "Where is the red square?"}],
  "top_k": 100
}
```

Add echoes `success`, `request_id`, `user_id`, and `session_id`. Search returns `{"data":[{"id":"...","content":[...],"score":0.0}]}` in retrieval order. `id` is deterministic for user, request, and message position. Timestamps are not fabricated. Reusing the same request ID with an identical payload is a no-op; a changed payload produces HTTP 409. A success response is sent only after all messages and their retrieval representations commit atomically and are searchable.

## Operational contract

- JPEG, PNG, WebP; at most 10 MiB decoded per image, 30 MiB per Add/query and 30 MiB across a Search response. Remote image URLs are rejected. A 40-million-pixel decoded image safety limit and 48 MiB HTTP-body limit also apply.
- `top_k` must be an integer in 1–100. Fewer than 100 results are valid; the service never fabricates evidence to fill the list.
- Up to 16 concurrent integration requests are supported through asynchronous admission and synchronous workers, with two provider calls in flight. Same-user operations serialize. Default Uvicorn admission limit is 40; saturation may return 503. This is not a Full-scale throughput certification.
- A provider send times out after 120 seconds. There is no automatic provider retry or silent model substitution. Exact successful inputs are cached per user and reused; failures, pending states, and uncertain outcomes are held for inspection. Creating a new run label does not clear them.
- Provider waits occur outside SQLite write transactions. A failed Add does not expose partially written memories. The service does not promise exactly-once execution across an uncertain external transport.
- Private source and derived state is stored in one dedicated database. Request bodies are not logged. Call metadata records hashes, status, model, and usage; credentials are not logged. `STOP` or `PAUSE` in the data directory prevents new provider calls while permitting cached operations.
- Sources and derived cache/call records expire after 29 days from ingestion; cleanup runs on startup and every 30 minutes while the process is running, with SQLite secure deletion and WAL checkpointing. Operators must include any separately created backups in their retention procedures. Official evaluation data must not be used for training, case analysis, or sharing.
- The current hosted instance requires its Windows host, internet connection, and Tailscale Funnel to remain online. No reboot persistence or high-availability guarantee is claimed.

## Validation and limitations

The included 16 offline tests cover authentication, image format checks, exact evidence preservation, official text-only string compatibility, stable reload, cross-user isolation, idempotence, conflicting writes, 16 concurrent duplicate Adds, atomic failure, top-k, media budgets, retention, and uncertain-call suppression.

Eleven finite synthetic live checks passed using actual GPT-4o-mini and v4 calls: text-to-image retrieval, pure-image retrieval, mixed queries, original evidence preservation, authenticated Add/Search, isolation, idempotence, and related protocol checks. The live fixture used two simple colored shapes with visible text. This does not establish accuracy on natural images, charts, fine-grained objects, long histories, or official benchmarks. Captions can omit or hallucinate visual details; text embeddings do not constitute a learned joint image-text embedding model. The baseline has no semantic correction/deletion policy or comprehensive safety filter.

Registration, approval, official Smoke, and Full are distinct stages. Repository publication alone does not imply any of the latter three.
