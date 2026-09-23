from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from .metabolic_polytope import MetabolicPolytope
from .verifier import Verifier, VerifierVerdict

gp = pytest.importorskip("gurobipy")

# Check license exists
@pytest.fixture(scope="session", autouse=True)
def _require_gurobi_license():
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as err:
        pytest.skip(f"gurobi unavailable: {err}")

def make_polytope(d, b_l, b_u, A_eq=None, b_eq=None):
    """
    Builds a MetabolicPolytope from dense bounds and an optional equality system.
    """

    if A_eq is None:
        A_eq = sp.csr_matrix((0, d))
        b_eq = np.zeros(0)
    else:
        A_eq = sp.csr_matrix(np.asarray(A_eq, dtype=float))
        b_eq = np.asarray(b_eq, dtype=float)

    return MetabolicPolytope(
        d,
        A_eq,
        b_eq,
        np.asarray(b_l, dtype=float),
        np.asarray(b_u, dtype=float)
    )

def cube(d, lb=0.0, ub=1.0):
    """Creates a d dimensional cube [lb,bu]^d"""
    return make_polytope(d, np.full(d, lb), np.full(d, ub))

# ------- empty -------
def test_is_empty_on_feasible_cube():
    assert Verifier.is_empty(cube(5)) is False

def test_is_empty_on_small_polytope():
    P = make_polytope(2, 
                      [-4.0, 4.0], 
                      [-4.0, 4.0], 
                      A_eq=[[1.0, 2.0]], 
                      b_eq=[0.0])
    
    assert Verifier.is_empty(cube(5)) is False

def test_empty_on_bad_inequality():
    P = make_polytope(2, [1.0, 1.0], [2.0, 2.0], A_eq=[[1.0, 1.0]], b_eq=[0.0])
    assert Verifier.is_empty(P) is True

# ------- subset -------
def test_subset_of_strictly_larger_cube():
    verdict = Verifier.is_subset(cube(3, 0.0, 1.0), cube(3, -1.0, 2.0))
    assert verdict, str(verdict)
    assert verdict.violations == []

def test_larger_cube_not_subset_of_smaller():
    verdict = Verifier.is_subset(cube(3, -1.0, 2.0), cube(3, 0.0, 1.0))
    assert not verdict
    assert len(verdict.violations) == 6

def test_identical_cubes_are_subsets_both_ways():
    lhs, rhs = Verifier.are_equal(cube(5), cube(5))
    assert lhs, str(lhs)
    assert rhs, str(rhs)

def test_one_polytope_with_infinite():
    P = cube(3, 0.0, 1.0)
    Q = make_polytope(3, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    verdict = Verifier.is_subset(P, Q)
    assert verdict
    assert verdict.lps_solved == 1

def test_tolerance_absorbs_overshoot():
    P = cube(1, 0.0, 1.0+1e-9)
    Q = cube(1, 0.0, 1.0)

    assert Verifier.is_subset(P, Q, tol=1e-6)
    assert not Verifier.is_subset(P, Q, tol=1e-12)

def test_relaxing_bound_in_containment():
    P = cube(3, 0.0, 1.0)
    Q = make_polytope(3, [0.0, 0.0, 0.0], [1.0, np.inf, 1.0])

    assert Verifier.is_subset(P, Q, tol=1e-6)
    verdict = Verifier.is_subset(Q, P)
    assert not verdict
    assert any("upper bound 1" in v for v in verdict.violations)

# ------- subset with equalities -------
def test_equality_row_of_q_enforced():
    P = cube(2, 0.0, 1.0)
    Q = make_polytope(2, [0.0, 0.0], [1.0, 1.0], A_eq=[[1.0, 1.0]], b_eq=[1.0])

    verdict = Verifier.is_subset(P, Q)
    assert not verdict
    assert any("row 0" in v for v in verdict.violations)

def test_empty_rows_skipped():
    P = cube(3, 0.0, 1.0)
    Q = make_polytope(3, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], A_eq=[[0.0, 0.0, 0.0]], b_eq=[0.0])

    verdict = Verifier.is_subset(P, Q)
    assert verdict, str(verdict)

def test_satisfying_equalities_and_is_contained():
    P = make_polytope(2, [0.0, 0.0], [1.0, 1.0], A_eq=[[1.0, 1.0]], b_eq=[1.0])
    Q = make_polytope(2, [0.0, 0.0], [1.0, 1.0], A_eq=[[2.0, 2.0]], b_eq=[2.0])

    verdict = Verifier.is_subset(P, Q)
    assert verdict, str(verdict)

# ------- subset with equalities -------
def test_empty_p_is_subset_of_anything():
    empty = make_polytope(2, [0.0, 0.0], [1.0, 1.0], A_eq=[[-3.0, -3.0]], b_eq=[3.0])
    verdict = Verifier.is_subset(empty, cube(2))

    assert verdict
    assert verdict.violations == []
    assert verdict.lps_solved == 1

def test_lp_count_matches():
    P = cube(2, 0.0, 1.0)
    Q = cube(2, 0.0, 1.0)

    verdict = Verifier.is_subset(P, Q)
    assert verdict.lps_solved == 5

def test_pinned_reaction_is_contained_in_its_box():
    P = make_polytope(2, [0.5, 0.0], [0.5, 1.0])
    Q = cube(2, 0.0, 1.0)

    verdict = Verifier.is_subset(P, Q)
    assert verdict, str(verdict)