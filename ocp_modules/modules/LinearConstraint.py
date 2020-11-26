from casadi import mtimes


def gen(var, A, lba, uba):

    g = mtimes(A, var)
    constrList = [(g, lba, uba)]

    return constrList
