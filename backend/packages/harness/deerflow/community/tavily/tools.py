import json
import logging

from langchain.tools import tool
from tavily import AsyncTavilyClient

from deerflow.community.search_time_range import SearchTimeRange
from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

# (payload field, normalized key) pairs every web_search result entry must
# carry as strings. A 200 body whose entries lack one (or carry a non-string)
# is malformed, not a result.
_SEARCH_LEAF_FIELDS = (("title", "title"), ("url", "url"), ("content", "snippet"))


def _get_tavily_client(tool_name: str = "web_search") -> AsyncTavilyClient:
    config = get_app_config().get_tool_config(tool_name)
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return AsyncTavilyClient(api_key=api_key)


def _unexpected_format_error(query: str) -> str:
    """Structured error for a search body that is not the documented shape."""
    return json.dumps(
        {"error": "Tavily returned an unexpected response format", "query": query},
        ensure_ascii=False,
    )


def _unexpected_fetch_format_error() -> str:
    """Plain-text error for an extract body that is not the documented shape."""
    return "Error: Tavily returned an unexpected response format"


def _normalize_search_results(res: object, query: str) -> tuple[list[dict[str, str]] | None, str | None]:
    """Normalize a decoded search body; return ``(entries, format_error)``.

    A missing, null or empty ``results`` means "no results" (the Tavily SDK
    already maps a missing key to ``[]``, and the sibling providers treat a
    null container the same way), so it keeps returning ``[]``. Anything else
    that is not a list of objects with string ``title``/``url``/``content``
    leaves is malformed and yields the structured format error instead of an
    unhandled ``TypeError``/``KeyError`` escaping the tool boundary.
    """
    if not isinstance(res, dict):
        logger.error("Tavily search returned an unexpected payload type: %s", type(res).__name__)
        return None, _unexpected_format_error(query)
    results = res.get("results")
    if results is None:
        return [], None
    if not isinstance(results, list):
        logger.error("Tavily search returned unexpected 'results' payload type: %s", type(results).__name__)
        return None, _unexpected_format_error(query)
    normalized: list[dict[str, str]] = []
    for result in results:
        if not isinstance(result, dict):
            logger.error("Tavily search returned a non-object result entry: %s", type(result).__name__)
            return None, _unexpected_format_error(query)
        entry: dict[str, str] = {}
        for field, key in _SEARCH_LEAF_FIELDS:
            value = result.get(field)
            if not isinstance(value, str):
                logger.error("Tavily search result entry has a missing or non-string %r: %r", field, value)
                return None, _unexpected_format_error(query)
            entry[key] = value
        normalized.append(entry)
    return normalized, None


def _failed_results_error(res: dict) -> tuple[str | None, str | None]:
    """Inspect extract ``failed_results``; return ``(failure_text, format_error)``.

    Missing or null means no failures and an empty list falls through to the
    results branch, matching the historical ``len(...) > 0`` gate. The container
    type is validated before the empty check (the #6550 rule), so a
    wrong-but-falsy container is a format error rather than "nothing found". A
    valid entry object without a usable ``error`` message still reports a
    failure with a generic message instead of raising ``KeyError``; a non-object
    entry is malformed.
    """
    failed = res.get("failed_results")
    if failed is None:
        return None, None
    if not isinstance(failed, list):
        logger.error("Tavily extract returned unexpected 'failed_results' payload type: %s", type(failed).__name__)
        return None, _unexpected_fetch_format_error()
    for entry in failed:
        if not isinstance(entry, dict):
            logger.error("Tavily extract returned a non-object failed_results entry: %s", type(entry).__name__)
            return None, _unexpected_fetch_format_error()
    if not failed:
        return None, None
    error = failed[0].get("error")
    if not isinstance(error, str):
        error = "Extraction failed"
    return f"Error: {error}", None


@tool("web_search", parse_docstring=True)
async def web_search_tool(query: str, time_range: SearchTimeRange | None = None) -> str:
    """Search the web.

    Args:
        query: The query to search for.
        time_range: Optional relative publication/update window. Use only when the request requires recent results.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 5
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    search_kwargs: dict[str, object] = {"max_results": max_results}
    if config is not None:
        for key in ("include_domains", "exclude_domains"):
            if key in config.model_extra:
                search_kwargs[key] = config.model_extra[key]
    if search_kwargs.get("include_domains"):
        search_kwargs["include_domains_mode"] = "filter"
    if time_range is not None:
        search_kwargs["time_range"] = time_range
    client = _get_tavily_client()
    try:
        res = await client.search(query, **search_kwargs)
    finally:
        await client.close()
    normalized_results, format_error = _normalize_search_results(res, query)
    if format_error is not None:
        return format_error
    json_results = json.dumps(normalized_results, indent=2, ensure_ascii=False)
    return json_results


@tool("web_fetch", parse_docstring=True)
async def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is not a valid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    client = _get_tavily_client("web_fetch")
    try:
        res = await client.extract([url])
    finally:
        await client.close()
    if not isinstance(res, dict):
        logger.error("Tavily extract returned an unexpected payload type: %s", type(res).__name__)
        return _unexpected_fetch_format_error()
    failure_text, format_error = _failed_results_error(res)
    if format_error is not None:
        return format_error
    if failure_text is not None:
        return failure_text
    results = res.get("results")
    if results is None:
        return "Error: No results found"
    # Container type is validated before the empty check (the #6550 rule): a
    # wrong-but-falsy container (e.g. ``{}``) is a format error, not "nothing
    # found".
    if not isinstance(results, list):
        logger.error("Tavily extract returned unexpected 'results' payload type: %s", type(results).__name__)
        return _unexpected_fetch_format_error()
    if not results:
        return "Error: No results found"
    result = results[0]
    # Extract results guarantee a URL and content, but not a page title.
    if not isinstance(result, dict) or not isinstance(result.get("raw_content"), str):
        logger.error("Tavily extract result entry has a missing or non-string 'raw_content'")
        return _unexpected_fetch_format_error()
    title = result.get("title") or result.get("url") or url
    return f"# {title}\n\n{result['raw_content'][:4096]}"
