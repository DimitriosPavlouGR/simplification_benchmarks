from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from PolyRound.api import PolyRoundApi
from PolyRound.settings import PolyRoundSettings

import numpy as np
from PolyRound.mutable_classes.polytope import Polytope

def polytope_to_json(P: Polytope) -> dict:
    S = P.S.to_numpy() if P.S is not None else np.zeros((0, P.A.shape[1]))
    h = P.h.to_numpy() if P.h is not None else np.zeros(0)

    rows, cols = np.nonzero(S)
    triplets = [[int(r), int(c), float(S[r, c])] for r, c in zip(rows, cols)]

    A = P.A.to_numpy()
    b = P.b.to_numpy()
    n = A.shape[1]

    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for i, row in enumerate(A):
        nz = np.nonzero(row)[0]
        if len(nz) == 1:
            j = nz[0]
            if row[j] > 0:
                ub[j] = float(b[i])
            else:
                lb[j] = float(-b[i])

    def to_bounds(v):
        return [None if not np.isfinite(x) else float(x) for x in v]

    finite_bounds = int(np.isfinite(lb).sum()+np.isfinite(ub).sum())
    return {
        "reaction_count": n,
        "metabolite_count": int(S.shape[0]),
        "A_eq": {
            "row_count": int(S.shape[0]),
            "col_count": int(n),
            "triplets": triplets
        },
        "b_eq": h.tolist(),
        "b_l": to_bounds(lb),
        "b_u": to_bounds(ub),
        "finite_bounds": finite_bounds,
    }

def export_run(out_path: Path,
               name: str,
               P: Polytope,
               Ps: Polytope,
               elapsed: float) -> None:

    original = polytope_to_json(P)
    simplified = polytope_to_json(Ps)
    bounds_relaxed = original["finite_bounds"]-simplified["finite_bounds"]
    dims_fixed = simplified["metabolite_count"]-original["metabolite_count"]

    jsn = {
        "name": name,
        "original": original,
        "simplified": simplified,
        "report": {
            "status": "OK",
            "input": {
                "variables": original["reaction_count"],
                "equalities": original["metabolite_count"],
                "finite_bounds": original["finite_bounds"],
            },
            "output": {
                "variables": simplified["reaction_count"],
                "equalities": simplified["metabolite_count"],
                "bounds_relaxed": bounds_relaxed,
            },
            "dimension_fixing": {
                "fixed_by_bounds": dims_fixed
            },
            "total": {"seconds": elapsed},
        },
    }
    out_path.write_text(json.dumps(jsn))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("model_dir", type=Path, help="the folder holding the .xml models")
    args = parser.parse_args()

    settings = PolyRoundSettings(backend="gurobi")

    models = sorted(args.model_dir.glob("*.sbml"))
    if not models:
        print(f"no .sbml models in {args.model_dir}", file=sys.stderr)
        return 1
    
    for path in models:
        name = path.stem
        out = Path("simplified_bigg_polyround")/f"{name}_polyround.json"
        if out.exists() and out.stat().st_size > 0:
            print(f" {name:<24} already done, skipping", flush=True)
            continue

        try:
            P = PolyRoundApi.sbml_to_polytope(str(path))
        except Exception as err:
             print(f" {name:<24} could not be read: {err}", file=sys.stderr)
             continue

        n = P.A.shape[1]
        m = P.S.shape[0]

        status = "FAILED"
        start = time.perf_counter()
        try:
            Ps = PolyRoundApi.simplify_polytope(P, settings=settings)
            status = "OK"
        except Exception as err:
            Ps = None
            print(f" {name}: {err}", file=sys.stderr)

        elapsed = time.perf_counter()-start

        if Ps is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            export_run(out, name, P, Ps, elapsed)

        bounds = P.A.shape[0]-Ps.A.shape[0] if Ps is not None else -1

        print(name, n, m, bounds, status, f"{elapsed:.3f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())