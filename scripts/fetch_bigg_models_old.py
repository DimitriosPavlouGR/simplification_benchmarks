#!/usr/bin/env python3
"""Downloads BiGG models with more than a given number of reactions.

The real BiGG API (bigg.bio in the old script doesn't exist) is much
simpler than a paged DataTables endpoint: one call to
http://bigg.ucsd.edu/api/v2/models returns every model's bigg_id plus its
reaction_count, metabolite_count and gene_count in a single JSON payload.
That lets us filter and sort locally instead of relying on the (broken)
website sort.

Model files are served uncompressed JSON, or gzip, at:
  http://bigg.ucsd.edu/static/models/{bigg_id}.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LIST_URL = "http://bigg.ucsd.edu/api/v2/models"
MODEL_URL = "http://bigg.ucsd.edu/static/models/{name}.json.gz"


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
    """Lists every model with its bigg_id and reaction/metabolite/gene counts.

    The whole catalogue (a few hundred models) comes back in one response,
    under the "results" key, so there is no pagination to handle.
    """
    payload = json.loads(fetch(LIST_URL))
    rows = payload.get("results", [])
    print(f"{len(rows)} models listed (API reports {payload.get('results_count')})")
    return rows


def download(name, out_dir, pause):
    """Downloads one model, skipping it when the file is already present.

    @return true if a file is present afterwards
    """
    out = out_dir / f"{name}.json"
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        blob = fetch(MODEL_URL.format(name=name))
    except urllib.error.HTTPError as err:
        print(f"  {name}: {err}", file=sys.stderr)
        return False
    except Exception as err:
        print(f"  {name}: {err}", file=sys.stderr)
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
    parser.add_argument("out_dir", type=Path, help="where to write the models")
    parser.add_argument("--min-reactions", type=int, default=2500,
                         help="only download models with more reactions than this "
                              "(default 2500)")
    parser.add_argument("--top", type=int, default=None,
                         help="cap the number of models, largest reaction count "
                              "first (default: no cap, just the threshold above)")
    parser.add_argument("--pause", type=float, default=0.2,
                         help="seconds to wait between downloads (default 0.2)")
    parser.add_argument("--dry-run", action="store_true",
                         help="list the models that would be downloaded and exit")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list_models()
    rows = [r for r in rows if r.get("bigg_id") and r.get("reaction_count") is not None]
    rows = [r for r in rows if r["reaction_count"] > args.min_reactions]
    rows.sort(key=lambda r: r["reaction_count"], reverse=True)
    if args.top:
        rows = rows[:args.top]

    print(f"{len(rows)} models have > {args.min_reactions} reactions")
    for r in rows:
        print(f"  {r['bigg_id']:<20} reactions={r['reaction_count']:<6} "
              f"metabolites={r.get('metabolite_count')} genes={r.get('gene_count')} "
              f"organism={r.get('organism')}")

    if args.dry_run:
        return 0

    failed = []
    for n, r in enumerate(rows, 1):
        name = r["bigg_id"]
        print(f"[{n}/{len(rows)}] {name} ({r['reaction_count']} reactions)")
        if not download(name, args.out_dir, args.pause):
            failed.append(name)

    total_mb = sum(p.stat().st_size for p in args.out_dir.glob("*.json")) / 1e6
    print(f"\n{len(rows) - len(failed)} of {len(rows)} downloaded, {total_mb:.0f} MB")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())