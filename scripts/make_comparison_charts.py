from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#3B7EA1"
ORANGE = "#E08A3C"

def polytope_dim(j: dict) -> int:
    import scipy.sparse as sp
    A = j["simplified"]["A_eq"]
    t = np.array(A["triplets"])
    S = sp.csr_matrix((t[:,2], (t[:,0].astype(int), t[:,1].astype(int))),
        shape=(A["row_count"], A["col_count"]))
    return A["col_count"]-np.linalg.matrix_rank(S.toarray())

def load_folder(folder: Path, suffix: str) -> dict[str, dict]:
    results = {}
    for path in sorted(folder.glob("*.json")):
        name = path.stem
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        try:
            with open(path) as f:
                results[name] = json.load(f)
        except Exception as err:
            print(f" could not read {path.name}: {err}", file=sys.stderr)

    return results

def grouped_bar(models, series_a, series_b,
                label_a, label_b, title, ylabel, out_path, ymin=None):
    x = np.arange(len(models))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x-width/2, series_a, width, label=label_a, color=BLUE)
    ax.bar(x+width/2, series_b, width, label=label_b, color=ORANGE)

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=60, ha="right", fontsize=9)

    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")

def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    clarkson_dir = Path(sys.argv[1])
    polyround_dir = Path(sys.argv[2])

    clarkson = load_folder(clarkson_dir, "_clarkson")
    polyround = load_folder(polyround_dir, "_polyround")

    common = sorted(
        name for name in clarkson.keys() & polyround.keys()
        if clarkson[name]["report"].get("status") == "OK"
        and polyround[name]["report"].get("status") == "OK"
    )

    if not common:
        print("no common models found")
        return 1

    print(f"{len(common)} models in common: {', '.join(common)}")

    time_c = [clarkson[n]["report"]["total"]["seconds"] for n in common]
    time_p = [polyround[n]["report"]["total"]["seconds"] for n in common]
    bounds_c = [clarkson[n]["report"]["output"]["bounds_relaxed"] for n in common]
    bounds_p = [polyround[n]["report"]["output"]["bounds_relaxed"] for n in common]

    dims_c = [polytope_dim(clarkson[n]) for n in common]
    dims_p = [polytope_dim(polyround[n]) for n in common]

    out_dir = Path("charts")
    out_dir.mkdir(exist_ok=True)

    grouped_bar(
        common, time_c, time_p,
        "VolEsti (HiGHS)", "Polyround (Gurobi)",
        f"Runtime: VolEsti vs Polyround ({len(common)} models)",
        "seconds",
        out_dir / "runtime_comparison.png"
    )

    grouped_bar(
        common, bounds_c, bounds_p,
        "VolEsti (HiGHS)", "Polyround (Gurobi)",
        f"Bounds relaxed: VolEsti vs Polyround ({len(common)} models)",
        "bounds relaxed",
        out_dir / "bounds_relaxed_comparison.png",
        ymin=min(min(bounds_c), min(bounds_p))*0.98,
    )

    grouped_bar(
        common, dims_c, dims_p,
        "VolEsti (HiGHS)", "Polyround (Gurobi)",
        f"Dimension after simplification: VolEsti vs Polyround ({len(common)} models)",
        "dimension",
        out_dir / "dimension_comparison.png",
        ymin=min(min(dims_c), min(dims_p))*0.98,
    )

if __name__ == "__main__":
    main()