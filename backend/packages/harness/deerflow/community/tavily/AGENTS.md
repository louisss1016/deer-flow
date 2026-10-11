# Tavily tools

The client helper selects credentials by tool name: search defaults to
`web_search`, fetch passes `web_fetch`, and an omitted key uses the SDK's
`TAVILY_API_KEY` fallback. Keep credential regressions in `backend/tests/test_tavily_tools.py`
on the real helper and SDK constructor, mocking only search/extract calls.

Each tool owns its per-call async client and awaits `close()` in `finally`,
including request failures and cancellation. Lifecycle regressions exercise
the real SDK connection pool without network requests.

Search forwards `include_domains` and `exclude_domains` only when present in
the `web_search` tool config's `model_extra`, alongside `max_results` and optional
`time_range`. Preserve explicit empty lists (no restriction of that kind) and
omit absent keys. These are deployment-only search-source settings; keep the
model-visible signature limited to `query` and `time_range`. Offline regressions
in `backend/tests/test_tavily_tools.py` pin SDK arguments and the tool schema.
For non-empty `include_domains`, explicitly send `include_domains_mode="filter"`
through the SDK's keyword arguments; inclusion must restrict sources rather than
boost them. Omit the mode when the include list is absent or empty.

Malformed 200 payloads must never escape the tool boundary. Both tools
normalize the SDK's decoded body through shape validation and return the
provider's own structured error instead of raising: `web_search` returns
`{"error": "Tavily returned an unexpected response format", "query": ...}`
and `web_fetch` returns `Error: Tavily returned an unexpected response
format`, matching brave (#6550), serper, serply, sofya, tencent_wsa,
unbrowse and groundroute (#6603). Keep the boundary between "no results"
and "malformed":

- A missing, null or empty `results` container keeps returning today's
  no-results output (`[]` / `Error: No results found`); some APIs signal
  "no results" with a null field, so null is not malformed.
- Validate the container type before the empty check (the #6550 rule): a
  wrong-but-falsy container (e.g. `results: {}`) is a format error, not
  "nothing found".
- `web_search` requires every entry to be an object with string `title`,
  `url` and `content` leaves (normalized to `title`/`url`/`snippet`).
- `web_fetch` requires the first `results` entry to be an object with a
  string `raw_content` leaf; `title` stays optional. A valid
  `failed_results` entry object without a usable `error` message reports
  `Error: Extraction failed` instead of raising `KeyError`; a non-object
  entry is malformed.

Regressions live in `backend/tests/test_tavily_tools.py`; keep the SDK's
own missing-key mapping (a missing `results` key arrives as `[]`) in mind
when pinning offline payloads.
