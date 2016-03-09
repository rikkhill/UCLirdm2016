import numpy as np


class SparseMatrix:

    def __init__(self, dim, basis=None, dropout=None):

        self.dim = dim

        if basis is None:
            self.basis = generate_basis(dim)
        else:
            self.basis = basis

        assert validate_basis(self.basis), "Basis must be PSD with minimal collinearity"

        if dropout is None:
            self.dropout = lambda x : x
        else:
            self.dropout = dropout

        self.full_matrix = np.zeros((dim, dim))
        self.sparse_matrix = np.zeros((dim, dim))

    # Generate an m x n sparse matrix
    def generate(self, m, n):
        m_matrix = np.random.multivariate_normal(np.zeros(self.dim), self.basis, m)
        n_matrix = np.random.multivariate_normal(np.zeros(self.dim), self.basis, n)
        self.full_matrix = np.dot(m_matrix, n_matrix.transpose())
        self.sparse_matrix = self.dropout(self.full_matrix)


# Returns a symmetric n x n matrix with ones on the diagonal
def generate_basis(n):
    mat = np.random.normal(0, 0.1, (n, n))
    mat = (mat + mat.transpose()) / 2
    np.fill_diagonal(mat, 1)
    return mat


def validate_basis(basis):
    w, v = np.linalg.eig(basis)
    try:
        np.linalg.cholesky(basis)
    except np.linalg.LinAlgError:
        return False

    # No variance
    if np.var(w) > 0.1:
        return False

    return True

