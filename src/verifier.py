from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from metabolic_polytope import MetabolicPolytope

@dataclass
class VerifierVerdict:
    """The outcome of a containment check."""
    ok: bool
    lps_solved: int = 0
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        header = "OK" if self.ok else f"FAILED ({len(self.violations)})"
        info = [f"{header}, {self.lps_solved} LPs solved"]
        info += [f" {v}" for v in self.violations[:10]]

        if (len(self.violations)) > 10:
            info.append(f" .... and {len(self.violations)-10} more")
            
        return "\n".join(info)
    
class Verifier:
    @staticmethod
    def metabolic_to_gurobi(m, P: MetabolicPolytope):
        """Converts P to a gurobi model"""
        A_eq, b_eq = P.getEqualities()
        b_l, b_u = P.getBounds()

        lb = np.where(np.isfinite(b_l), b_l, -GRB.INFINITY)
        ub = np.where(np.isfinite(b_u), b_u, GRB.INFINITY)
        x = m.addMVar(P.getDimension(), lb=lb, ub=ub, name="x")

        if A_eq.shape[0] > 0:
            m.addMConstr(A_eq, x, GRB.EQUAL, b_eq)

        return x

    @staticmethod
    def is_empty(P: MetabolicPolytope) -> bool:
        """Returns true if P has no feasible point."""
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as m:
                Verifier.metabolic_to_gurobi(m, P)
                m.setObjective(0.0)
                m.optimize()
                return m.Status != GRB.OPTIMAL

    @staticmethod
    def is_subset(P: MetabolicPolytope, Q: MetabolicPolytope, tol: float = 1e-6):
        """Checks whether P is contained in Q"""
        if P.getDimension() != Q.getDimension():
            return VerifierVerdict(False, 0, [f"dimension {P.getDimension()} vs {Q.getDimension()}"])

        d = P.getDimension()
        A_q, b_q = Q.getEqualities()
        bl_q, bu_q = Q.getBounds()

        violations = []
        lps_solved = 0

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as m:
                x = Verifier.metabolic_to_gurobi(m, P)

                m.setObjective(0.0)
                m.optimize()
                lps_solved += 1

                if m.Status != GRB.OPTIMAL:
                    return VerifierVerdict(True, lps_solved, [])

                def optimum(expr, sense) -> float | None:
                    nonlocal lps_solved
                    m.setObjective(expr, sense)
                    m.optimize()
                    lps_solved += 1
                    return m.ObjVal if m.Status == GRB.OPTIMAL else None

                # Going over equalities of Q
                for i in range(A_q.shape[0]):
                    first, last = A_q.indptr[i], A_q.indptr[i+1]
                    id, val = A_q.indices[first:last], A_q.data[first:last]

                    if len(id) == 0:
                        continue

                    expr = gp.quicksum(float(v)*x[int(j)] for j, v in zip(id, val))
                    rhs = float(b_q[i])

                    hi = optimum(expr, GRB.MAXIMIZE)
                    if hi is None or hi > rhs+tol:
                        violations.append(f"quality row {i}: max {hi} vs rhs {rhs}")

                    lo = optimum(expr, GRB.MINIMIZE)
                    if lo is None or lo < rhs-tol:
                        violations.append(f"equality row {i}: min {lo} vs rhs {rhs}")

                # Going over finite bounds of Q.
                for k in range(d):
                    if np.isfinite(bu_q[k]):
                        hi = optimum(x[k], GRB.MAXIMIZE)
                        if hi is None or hi > bu_q[k]+tol:
                            violations.append(f"upper bound {k}: max {hi} vs {bu_q[k]}")

                    if np.isfinite(bl_q[k]):
                        lo = optimum(x[k], GRB.MINIMIZE)
                        if lo is None or lo < bl_q[k]-tol:
                            violations.append(f"lower bound {k}: min {lo} vs {bl_q[k]}")

        return VerifierVerdict(not violations, lps_solved, violations)

    @staticmethod
    def are_equal(P: MetabolicPolytope, Q: MetabolicPolytope, tol: float = 1e-6) -> tuple[VerifierVerdict, VerifierVerdict]:
        return Verifier.is_subset(P, Q, tol), Verifier.is_subset(Q, P, tol)
                    