from __future__ import annotations
import numpy as np
import scipy.sparse as sp

class MetabolicPolytope:
    """
    A MetabolicPolytope

    Attributes
    d : int
        Ambient dimension
    A_eq : scipy.sparse.csr_matrix, shape (m, d)
        Equality constraint matrix
    b_eq : np.ndarray, shape (m,)
        Equality constraint vector
    b_l : np.ndarray, shape (d,)
        Lower bounds for the variables
    b_u : np.ndarray, shape (d,)
        Upper bounds for the variables
    """

    def __init__(self, d , A_eq, b_eq, b_l, b_u):
        self.d = d

        # Safeguard
        if A_eq is None:
            A_eq = sp.csr_matrix((0, self.d))

        self.A_eq = sp.csr_matrix(A_eq, dtype=float)

        self.b_l = np.asarray(b_l, dtype=float).reshape(-1)
        self.b_u = np.asarray(b_u, dtype=float).reshape(-1)
        self.b_eq = np.asarray(b_eq, dtype=float).reshape(-1)

        self._validate()

    def _validate(self) -> None:
        if self.A_eq.shape[1] != self.d:
            raise ValueError(
                f"A_eq has {self.A_eq.shape[1]} columns, but expected d={self.d}"
            )

        if self.A_eq.shape[0] != self.b_eq.shape[0]:
            raise ValueError(
                f"A_eq has {self.A_eq.shape[0]} rows, but b_eq has {self.b_eq.shape[0]} rows"
            )

        if self.b_l.shape[0] != self.d or self.b_u.shape[0] != self.d:
            raise ValueError(
                f"b_l and b_u must have length d={self.d}, but got lengths {self.b_l.shape[0]} and {self.b_u.shape[0]}"
                " instead"
            )
        
        if np.any(self.b_l > self.b_u):
            corrupted = np.where(self.b_l > self.b_u)[0]
            raise ValueError(
                f"Lower bounds exceed upper bounds for indices {corrupted.tolist()}"
            )

    # Getters

    def getDimension(self) -> int:
        return self.d

    def getEqualities(self) -> tuple[sp.csr_matrix, np.ndarray]:
        return self.A_eq, self.b_eq

    def getBounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.b_l, self.b_u

    def getNumEqualities(self) -> int:
        return self.A_eq.shape[0]

    def getSingletonRowCount(self) -> int:
        A = self.A_eq
        if A.shape[0] == 0:
            return 0

        rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        per_row = np.bincount(rows[A.data != 0], minlength=A.shape[0])
        return int((per_row == 1).sum()) 
    
    def copy(self) -> "MetabolicPolytope":
        return MetabolicPolytope(
            self.d,
            self.A_eq.copy(),
            self.b_eq.copy(),
            self.b_l.copy(),
            self.b_u.copy()
        )
    
    def __repr__(self) -> str:
        return (
            f"MetabolicPolytope(d={self.d},"
            f"n={self.getNumEqualities()})"
        )
