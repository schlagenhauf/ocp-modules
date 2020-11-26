from ocp_modules.modules import LinearConstraint


def gen(var, linear_coefficient, affine_coefficient, lower_bound, upper_bound):

    constr = LinearConstraint.gen(var,
                                  linear_coefficient,
                                  lower_bound-affine_coefficient,
                                  upper_bound-affine_coefficient)
    return constr
