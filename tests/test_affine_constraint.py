import numpy as np
from ocp_modules.modules import AffineConstraint
from casadi.tools import struct_symSX, entry


def test_affine_constraint():
    var = struct_symSX([
        entry('x', shape=(2, 1))])
    linear_coefficient = np.eye(2) * 2
    affine_coefficient = np.ones((2,))
    lower_bound = np.zeros((2, 1))
    upper_bound = np.ones((2, 1))

    constr = AffineConstraint.gen(
        var, linear_coefficient, affine_coefficient, lower_bound, upper_bound)

    g, lb, ub = constr[0]

    assert (lb == (lower_bound - affine_coefficient)).all()
    assert (ub == (upper_bound - affine_coefficient)).all()
    assert '@1=2, [(@1*x_0), (@1*x_1)]' == str(g)
