# Citation tracking for reproduced papers — maintainer guide

How the citation counts on the *Reproduced Papers* docs page are produced,
how to add or update papers, and how to change the system itself.

## How it works

```
papers.json          (paper registry: one entry per reproduced paper)
      │
      ▼
docs/fetch_citations.py      (manual run; queries the OpenAlex API per DOI)
      │
      ▼
citations.json       (generated counts + fetch date; committed, do not edit)
      │
      ▼
Sphinx build         (_ext/merlin_citations.py renders banner, table, badges)
```

Two design rules:

1. **The Sphinx build never touches the network.** Pages render from the
   committed `citations.json`, so builds are reproducible and work offline.
   Freshness is whatever the last `fetch_citations.py` run produced — every
   rendered number carries an explicit "as of *date*" stamp.
2. **Citation counts are never edited by hand.** The only file to maintain is
   the registry (which papers exist and their DOIs); the counts always come
   from OpenAlex via the script.

## The files

| File | Role | Contents |
|---|---|---|
| `papers.json` | maintained | key, title, authors_short, year, venue, doi, doc (reproduction page), optional `"not_indexed": true` |
| `citations.json` | generated — do not edit | per-paper `cited_by_count` + `fetched_at` date |
| `../../_ext/merlin_citations.py` | code | the three Sphinx directives (`merlin-citations-summary`, `-table`, `-badge`) |
| `../../../fetch_citations.py` | code | the OpenAlex client |

## How to add a newly reproduced paper

1. Add an entry to `papers.json`. `key` must equal the reproduction page's
   file stem (e.g. `photonic_kernel` for `reproductions/photonic_kernel.rst`);
   `doc` is the docname path (`reproduced_papers/reproductions/<key>`). Prefer
   the **published** DOI over the arXiv one when the paper has appeared in a
   journal — published DOIs resolve more reliably and aggregate better.
2. Run the fetch script and commit the updated `citations.json`:

   ```bash
   python docs/fetch_citations.py
   ```

3. Add the badge inside the new page's *Paper Information* admonition
   (the reproduction template contains a placeholder comment showing where):

   ```rst
   .. merlin-citations-badge:: <key>
   ```

4. Build the docs; the paper appears in the banner totals and the sorted
   table automatically. Nothing else to update — ordering, totals, and the
   index table are all derived. The fetched count is the paper's all-time
   citation total since publication, so it makes no difference whether the
   paper came out last month or five years ago.

## Maintenance notes

### Unresolvable DOIs and papers OpenAlex has not indexed

Two cases:

- **Wrong or malformed DOI** — verify it resolves at `https://doi.org/<doi>`
  and that OpenAlex knows it: `https://api.openalex.org/works/doi:<doi>`.
  Papers sometimes have both an arXiv DOI (`10.48550/arXiv.XXXX`) and a
  publisher DOI; try the other one.
- **Genuinely fresh preprint** (OpenAlex typically indexes new arXiv papers
  within weeks-to-months) — mark the registry entry with
  `"not_indexed": true`. It renders as "—" / "not yet indexed" instead of a
  count. When OpenAlex eventually indexes it, the next fetch run **fails
  loudly** telling you to remove the marker — the registry cannot silently
  drift out of date.

### Preprint later published in a journal

Update the registry entry's `doi` (and `venue`/`year`) to the published
version and re-run the fetch. OpenAlex merges preprint and published-version
records for most papers, so the count reflects both. Precedent: the
nearest-centroids paper was registered with its npj Quantum Information DOI
even though the reproduction page originally only cited arXiv.

### Counts compared to Google Scholar

Different indexes. Google Scholar counts theses, slides, and anything it can
crawl; OpenAlex counts indexed scholarly works, so its numbers run lower but
are reproducible and API-accessible (Scholar has no API and forbids scraping).
Treat the numbers as a defensible lower bound and don't mix sources.

### Refreshing the counts

The counts don't refresh themselves: run `python docs/fetch_citations.py` and
commit the diff (the git history of `citations.json` doubles as a citation
time-series). A scheduled CI job or an on-page-load live refresh are both
possible extensions — the script exits non-zero on any problem precisely so a
cron run can fail visibly. Neither is wired up yet.

### Switching or extending the data source

All network access lives in `fetch_citations.py` (`fetch_work()` builds the
request; `main()` interprets the response). The renderer only reads
`citations.json`, whose `"source"` field flows into every "as of" stamp. So a
source swap means: rewrite `fetch_work()` for the new API, keep the output
schema, update `"source"`. The Sphinx extension and all pages are untouched.
Candidate alternative: Semantic Scholar (`api.semanticscholar.org`, handles
raw arXiv IDs, tighter rate limits).

### Changing what's displayed

- Columns/sorting/labels: `docs/source/_ext/merlin_citations.py` (the three
  directive/visitor pairs).
- Styling: the `mq-citations-*` block in `docs/source/_static/css/style.css`.
- Placement: the directives are ordinary rst — move them within
  `reproduced_papers.rst` or add them to other pages freely.

### Build-time strictness

The directives raise hard errors (which `-W` turns fatal) when the registry
and `citations.json` disagree — unknown badge key, missing count for an
indexed paper, missing reproduction page. This is deliberate: the alternative
is silently rendering stale or partial numbers. The error message always names
the paper key and the fix (usually: run `fetch_citations.py`, or add/remove a
`not_indexed` marker).
