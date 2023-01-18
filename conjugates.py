from .imports import *
from .Matsubarafft import Matsubaraflip



def RtoA(C):
    """
    Transform the retarded Green's function into the advanced.

    Parameters
    ----------
    C : ndarray
        Green's function.

    Returns
    -------
    ndarray
        Green's function.

    """
    return np.swapaxes(C, 0, 1).T.conj()


def ItoJ(C, particle):
    """
    Transform the left mixed Green's function into the right mixed.

    Parameters
    ----------
    C : ndarray
        Green's function.
    particle : int, bool, str
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion.

    Returns
    -------
    ndarray
        Green's function.

    """
    return -Matsubaraflip(np.swapaxes(C, 0, 1).T.conj(), particle)
