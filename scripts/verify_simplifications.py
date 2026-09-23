"""
Checks that a simplification preserved its polytope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from metabolic_polytope import MetabolicPolytope
from verifier import Verifier

def bounds_from_json(values, infinite):
    return np.array([infinite if v is None else float(v) for v in values], dtype=float)

def network_from_json(model):
    matrix = model["A_eq"]
    rows, cols = matrix["row_count"], matrix["col_count"]

    if matrix["triplets"]:
        i, j, v = zip(*matrix["triplets"])
    else:
        i, j, v = (), (), ()

    A_eq = sp.coo_matrix((v, (i,j)), shape=(rows, cols)).tocsr()

    return MetabolicPolytope(
        matrix["col_count"],
        A_eq,
                np.asarray(model["b_eq"], dtype=float),
        bounds_from_json(model["b_l"], -np.inf),
        bounds_from_json(model["b_u"], np.inf),
    )

def verify(path, tol):
    with open(path) as f:
        run = json.load(f)
        transformed = run["transformed"]
        A0 = np.array(transformed["A"], dtype=float)
        b0 = np.array(transformed["b"], dtype=float)
        N = np.array(transformed["N"], dtype=float)
        shift = np.array(transformed["shift"], dtype=float)
        print(A0.shape, b0.shape, N.shape, shift.shape)

    name = run["name"]
    print(name)
    P = network_from_json(run["original"])
    Ps = network_from_json(run["simplified"])

    print(P)
    print(Ps)
    a, b = Verifier.are_equal(P, Ps, tol)
    ok = bool(a) and bool(b)
    print(f"   verdict: {'SAME SET' if ok else 'DIFFERENT SET'}")
    return ok

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs='+', type=Path, help="the exported .json files to verify")
    parser.add_argument("--tol", type=float, default=1e-6, help="the containment tolerance (default 1e-6)")
    args = parser.parse_args()
    failed = []
    for path in args.paths:
        if not verify(path, args.tol):
            failed.append(path.name)
        print()

    if failed:
        print(f"{len(failed)} of {len(args.paths)} failed: {','.join(failed)}")
        return 1
    
    print(f"all {len(args.paths)} runs preserved their polytope")
    return 0

if __name__ == "__main__":
    sys.exit(main())