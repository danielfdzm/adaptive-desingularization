"""Verify a real cubic obstruction to a proposed Gaussian closure criterion.

Source: R. Szafarczyk, Communications in Algebra 51 (2023), 2688--2704,
Section 4; https://arxiv.org/html/2206.03732v2#S4.
The source uses divided powers. Q below uses ordinary derivatives.

Requires SymPy and NumPy. No manuscript data or files are modified.
The finite-field computations are exact lower-rank certificates for rational
matrices. Their ranks reach column/row upper bounds, so prove rational ranks.
"""

from itertools import product
import numpy as np
import sympy as s

variables = s.symbols("a1 a2 b1 b2 c1 c2")
a1, a2, b1, b2, c1, c2 = variables
Q = (
    a1*b1*c1 + a2*b2*c2
    + (a1*a2**2 + b1*b2**2 + c1*c2**2)/2
    + (a1**3 + b1**3 + c1**3)/6
)


def monomials(degree):
    return [alpha for alpha in product(range(degree + 1), repeat=6)
            if sum(alpha) == degree]


def derivative(alpha):
    value = Q
    for variable, order in zip(variables, alpha):
        value = s.diff(value, variable, order)
    return s.Poly(value, *variables)


def derivative_matrix(order):
    source = monomials(order)
    target = monomials(3 - order)
    derivatives = [derivative(alpha) for alpha in source]
    return source, s.Matrix([
        [value.coeff_monomial(beta) for value in derivatives]
        for beta in target
    ])


def annihilator_basis(order):
    source, matrix = derivative_matrix(order)
    basis = []
    for vector in matrix.nullspace():
        denominator = s.ilcm(*[value.q for value in vector])
        basis.append({alpha: int(value * denominator)
                      for alpha, value in zip(source, vector) if value})
    return basis


def product_matrix(left, right, degree, symmetric=False):
    monomials_in_degree = monomials(degree)
    indices = {alpha: index for index, alpha in enumerate(monomials_in_degree)}
    columns = []
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            if symmetric and j < i:
                continue
            column = np.zeros(len(monomials_in_degree), dtype=np.int64)
            for alpha, coefficient in first.items():
                for beta, other_coefficient in second.items():
                    exponent = tuple(a + b for a, b in zip(alpha, beta))
                    column[indices[exponent]] += coefficient * other_coefficient
            columns.append(column)
    return np.array(columns, dtype=np.int64).T


def rank_mod_prime(matrix, prime=10007):
    assert s.isprime(prime)
    array = matrix.copy() % prime
    rank = 0
    for column in range(array.shape[1]):
        nonzero = np.flatnonzero(array[rank:, column])
        if not len(nonzero):
            continue
        pivot = rank + nonzero[0]
        array[[rank, pivot]] = array[[pivot, rank]]
        inverse = pow(int(array[rank, column]), -1, prime)
        array[rank] = (array[rank] * inverse) % prime
        if rank + 1 < array.shape[0]:
            array[rank + 1:] = (
                array[rank + 1:]
                - array[rank + 1:, column, None] * array[rank]
            ) % prime
        rank += 1
        if rank == array.shape[0]:
            break
    return rank


def main():
    divided_power_source = s.Poly(
        a1*b1*c1 + a2*b2*c2 + a1*a2**2 + b1*b2**2 + c1*c2**2
        + a1**3 + b1**3 + c1**3,
        *variables,
    )
    ordinary_polynomial = sum(
        coefficient * s.prod(variable**order / s.factorial(order)
                             for variable, order in zip(variables, alpha))
        for alpha, coefficient in divided_power_source.terms()
    )
    assert s.expand(ordinary_polynomial - Q) == 0
    derivative_ranks = [derivative_matrix(order)[1].rank() for order in range(4)]
    assert derivative_ranks == [1, 6, 6, 1]
    quadrics, cubics = annihilator_basis(2), annihilator_basis(3)
    assert (len(quadrics), len(cubics)) == (15, 55)

    degree4 = product_matrix(quadrics, quadrics, 4, symmetric=True)
    degree5 = product_matrix(quadrics, cubics, 5)
    assert degree4.shape == (126, 120)
    assert degree5.shape == (252, 825)
    rank4, rank5 = rank_mod_prime(degree4), rank_mod_prime(degree5)
    assert rank4 == 120  # Reaches the column upper bound, hence exact over Q.
    assert rank5 == 252  # Reaches the row upper bound, hence exact over Q.

    # I is homogeneous, has no linear elements, and contains every quartic.
    # Thus (I^2)_4 = I_2 I_2 and (I^2)_5 = I_2 I_3. The second equality
    # above means every quintic, and hence every higher-degree form, is in I^2.
    quotient_hilbert = [1, 6, 21, 56, 126 - rank4, 252 - rank5]
    algebra_length = sum(derivative_ranks)
    tangent_dimension = sum(quotient_hilbert) - algebra_length
    assert tangent_dimension == 76 < 6 * algebra_length == 84

    laplacian = s.expand(sum(s.diff(Q, variable, 2) for variable in variables))
    assert laplacian == 2 * (a1 + b1 + c1)
    operator_polynomial = -Q - (a1 + b1 + c1)

    # Gaussian derivative identity, for sigma=1, without exponential functions:
    # G^{-1} d_i(f G) = d_i f - x_i f.
    gaussian_quotient = 0
    for alpha, coefficient in s.Poly(operator_polynomial, *variables).terms():
        value = s.Integer(1)
        for variable, order in zip(variables, alpha):
            for _ in range(order):
                value = s.diff(value, variable) - variable * value
        gaussian_quotient += coefficient * value
    assert s.expand(gaussian_quotient - Q) == 0

    print("Q =", Q)
    print("Derivative ranks by order:", derivative_ranks, "; length:", algebra_length)
    print("Annihilator dimensions in degrees 2 and 3:", len(quadrics), len(cubics))
    print("Exact rational ranks of (I^2)_4 and (I^2)_5:", rank4, rank5)
    print("Hilbert function of S/I^2:", quotient_hilbert)
    print("Tangent dimension:", tangent_dimension, "< smoothable dimension:", 84)
    print("Gaussian identity checked: p(partial)G=QG, p=-Q-a1-b1-c1, sigma=1.")
    print("All checks passed.")


if __name__ == "__main__":
    main()
