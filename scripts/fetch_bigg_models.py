#!/usr/bin/env python3
"""Downloads BiGG models (from bigg.bio) that have more than a given number
of reactions.

The site's own sort/filter UI is unreliable, but the underlying table API
(https://bigg.bio/api/v3/models) already returns each model's reaction,
metabolite and gene counts alongside its bigg_id -- so we page through that,
filter and sort locally, and only download the models that match.

At least one of --json or --sbml must be given.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LIST_URL = "https://bigg.bio/api/v3/models"
JSON_URL = "https://bigg.bio/static/models/models/{name}.biggr.json.gz"
SBML_URL = "https://bigg.bio/static/models/models/{name}.biggr.sbml.gz"


def fetch(url, retries=3, backoff=2.0):
    """Fetches a url, retrying on transient failures.

    @return the response body as bytes
    """
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url)
            request.add_header("User-Agent", "bigg-model-fetch/1.0")
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == retries - 1:
                raise
            wait = backoff * (attempt + 1)
            print(f"  {err}, retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def list_models():
    """Lists every model row the table API reports, paging through the table.

    Each row includes model__bigg_id and modelcount__reaction_count (plus
    metabolite/gene counts), so we can filter/sort here instead of relying
    on the website's sort, which the site itself says is broken.
    """
    rows = []
    total = None
    while total is None or len(rows) < total:
        query = urllib.parse.urlencode({"start": len(rows), "length": 500})
        payload = json.loads(fetch(f"{LIST_URL}?{query}"))
        if total is None:
            total = payload.get("recordsTotal", 0)
        batch = payload.get("data", [])
        if not batch:
            break
        rows.extend(batch)
        print(f"  listed {len(rows)}/{total}")
    print(f"{len(rows)} models listed (API reports {total})")
    return rows


def download_file(url, out, pause):
    """Downloads one file, skipping it when it is already present.

    Decompresses gzipped content automatically.
    @return true if a file is present afterwards
    """
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        blob = fetch(url)
    except urllib.error.HTTPError as err:
        print(f"    {out.name}: {err}", file=sys.stderr)
        return False
    except Exception as err:
        print(f"    {out.name}: {err}", file=sys.stderr)
        return False
    try:
        text = gzip.decompress(blob)
    except OSError:
        # Occasionally served uncompressed despite the .gz name.
        text = blob
    out.write_bytes(text)
    time.sleep(pause)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path,
                        help="where to write the models")
    parser.add_argument("--json", action="store_true",
                        help="download the COBRA .json files")
    parser.add_argument("--sbml", action="store_true",
                        help="download the SBML .sbml files")
    parser.add_argument("--min-reactions", type=int, default=2500,
                        help="only download models with more reactions than "
                             "this (default 2500)")
    parser.add_argument("--top", type=int, default=None,
                        help="cap the number of models, largest reaction "
                             "count first (default: no cap)")
    parser.add_argument("--pause", type=float, default=0.2,
                        help="seconds to wait between downloads (default 0.2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the models that would be downloaded and exit")
    args = parser.parse_args()

    if not args.json and not args.sbml:
        parser.error("at least one of --json or --sbml must be given")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list_models()
    rows = [r for r in rows
            if r.get("model__bigg_id")
            and r.get("modelcount__reaction_count") is not None]
    rows = [r for r in rows if r["modelcount__reaction_count"] > args.min_reactions]
    rows.sort(key=lambda r: r["modelcount__reaction_count"], reverse=True)
    if args.top:
        rows = rows[:args.top]

    formats = ", ".join(f for f, on in [("json", args.json), ("sbml", args.sbml)] if on)
    print(f"\n{len(rows)} models have > {args.min_reactions} reactions "
          f"(formats: {formats})")
    for r in rows:
        print(f"  {r['model__bigg_id']:<20} "
              f"reactions={r['modelcount__reaction_count']:<6} "
              f"metabolites={r.get('modelcount__metabolite_count')} "
              f"genes={r.get('modelcount__gene_count')} "
              f"organism={r.get('model__organism')}")

    if args.dry_run:
        return 0

    failed = []
    for n, r in enumerate(rows, 1):
        name = r["model__bigg_id"]
        print(f"[{n}/{len(rows)}] {name} ({r['modelcount__reaction_count']} reactions)")

        ok = True
        if args.json:
            ok &= download_file(JSON_URL.format(name=name),
                                args.out_dir / f"{name}.json",
                                args.pause)
        if args.sbml:
            ok &= download_file(SBML_URL.format(name=name),
                                args.out_dir / f"{name}.sbml",
                                args.pause)
        if not ok:
            failed.append(name)

    total_mb = sum(p.stat().st_size for p in args.out_dir.iterdir()) / 1e6
    print(f"\n{len(rows) - len(failed)} of {len(rows)} downloaded, {total_mb:.0f} MB")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())