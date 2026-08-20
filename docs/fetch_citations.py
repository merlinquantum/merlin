"""Refresh citation counts for the reproduced-papers documentation.

Reads the paper registry at ``docs/source/_data/citations/papers.json``,
queries the OpenAlex API for each paper's citation count, and writes the
results to ``docs/source/_data/citations/citations.json``.

The Sphinx build never touches the network: it renders from the committed
``citations.json``. Run this script (from any directory) whenever the counts
should be refreshed, then commit the updated JSON::

    python docs/fetch_citations.py [--mailto you@example.org]

``--mailto`` adds your address to requests, which routes them to OpenAlex's
faster "polite pool". It is optional.

Failure policy
--------------
The script exits non-zero if any registry paper cannot be resolved, unless
that paper is explicitly marked ``"not_indexed": true`` in the registry.
If a paper marked ``not_indexed`` starts resolving, the script also exits
non-zero and asks for the marker to be removed, so the registry cannot
silently drift out of date.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "source" / "_data" / "citations"
REGISTRY_PATH = DATA_DIR / "papers.json"
CITATIONS_PATH = DATA_DIR / "citations.json"
OPENALEX_WORKS_URL = "https://api.openalex.org/works/doi:{doi}"
REQUEST_FIELDS = "id,display_name,cited_by_count,publication_year"
REQUEST_TIMEOUT_S = 30.0
REQUEST_SPACING_S = 0.2
REQUEST_MAX_RETRIES = 3
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "merlin-docs (+https://github.com/merlinquantum/merlin)"


def load_registry() -> list[dict[str, Any]]:
    """Load and minimally validate the paper registry.

    Returns
    -------
    list[dict]
        Registry entries, each with at least ``key`` and ``doi``.

    Raises
    ------
    SystemExit
        If the registry is missing, unparsable, or contains entries
        without the required fields.
    """
    try:
        entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Cannot read registry {REGISTRY_PATH}: {exc}")

    if not isinstance(entries, list) or not entries:
        sys.exit(f"Registry {REGISTRY_PATH} must be a non-empty JSON list.")

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not entry.get("key") or not entry.get("doi"):
            sys.exit(
                f"Registry entry #{index} must be an object with "
                f"non-empty 'key' and 'doi' fields (got: {entry!r})."
            )

    keys = [entry["key"] for entry in entries]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        sys.exit(f"Registry {REGISTRY_PATH} has duplicate keys: {duplicates}.")
    return entries


def fetch_work(doi: str, mailto: str | None) -> dict[str, Any] | None:
    """Fetch one work record from OpenAlex.

    Parameters
    ----------
    doi : str
        DOI of the paper, without a URL prefix.
    mailto : str | None
        Optional contact address for OpenAlex's polite pool.

    Returns
    -------
    dict | None
        The OpenAlex work record, or ``None`` if OpenAlex does not index
        this DOI (HTTP 404).

    Raises
    ------
    RuntimeError
        On any transport or HTTP error other than 404, after retrying
        rate-limit (429) and server (5xx) responses with backoff.
    """
    params = {"select": REQUEST_FIELDS}
    if mailto:
        params["mailto"] = mailto
    url = (
        OPENALEX_WORKS_URL.format(doi=urllib.parse.quote(doi))
        + "?"
        + (urllib.parse.urlencode(params))
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(REQUEST_MAX_RETRIES):
        last = attempt == REQUEST_MAX_RETRIES - 1
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in RETRYABLE_STATUS and not last:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAlex returned HTTP {exc.code} for {doi}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if not last:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAlex request failed for {doi}: {exc}") from exc
    return None


def extract_counts(work: dict[str, Any]) -> dict[str, Any]:
    """Extract the stored citation fields from an OpenAlex work record.

    Raises
    ------
    ValueError
        If the record lacks a usable ``cited_by_count`` or ``id``.
    """
    try:
        return {
            "cited_by_count": int(work["cited_by_count"]),
            "openalex_id": work["id"].rsplit("/", 1)[-1],
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"unexpected OpenAlex payload ({exc}); keys={sorted(work)}"
        ) from exc


def main() -> int:
    """Fetch all citation counts and rewrite citations.json.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on any resolution failure or
        stale ``not_indexed`` marker.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mailto",
        default=None,
        help="Contact email forwarded to OpenAlex (optional, enables the polite pool).",
    )
    args = parser.parse_args()

    registry = load_registry()
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for index, entry in enumerate(registry):
        if index:
            time.sleep(REQUEST_SPACING_S)
        key = entry["key"]
        doi = entry["doi"]
        marked_not_indexed = bool(entry.get("not_indexed"))
        try:
            work = fetch_work(doi, args.mailto)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        if work is None:
            if marked_not_indexed:
                print(f"{key:32s}    --  (marked not_indexed; still unresolved)")
            else:
                errors.append(
                    f"{key}: DOI {doi} is not indexed by OpenAlex. Fix the DOI "
                    'or mark the registry entry with "not_indexed": true.'
                )
            continue

        if marked_not_indexed:
            errors.append(
                f'{key}: marked "not_indexed" but now resolves on OpenAlex '
                f"({work['id']}). Remove the marker from the registry."
            )
            continue

        try:
            results[key] = extract_counts(work)
        except ValueError as exc:
            errors.append(f"{key}: {exc}.")
            continue
        display_name = str(work.get("display_name", ""))[:55]
        print(f"{key:32s} {results[key]['cited_by_count']:5d}  {display_name}")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    payload = {
        "source": "OpenAlex",
        "fetched_at": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "papers": {key: results[key] for key in sorted(results)},
    }
    CITATIONS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(item["cited_by_count"] for item in results.values())
    print(f"\nWrote {CITATIONS_PATH} ({len(results)} papers, {total} citations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
