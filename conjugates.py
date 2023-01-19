from .imports import *
from .Matsubarafft import Matsubaraflip



def RtoA(C, typeC=None):
    """
    Transform the retarded contour function into the advanced.

    Parameters
    ----------
    C : ndarray
        Contour function.
    typeC : str, optional
        Specify if C is a matrix, "M", or an interaction tensor, "T". Default is None.

    Returns
    -------
    ndarray
        Contour function.

    """
    if len(C.shape)==4 or typeC=="M":
        return np.swapaxes(C, 0, 1).T.conj()
    elif len(C.shape)==6 or typeC=="T":
        return np.einsum("stmnlk->tsklnm", C.conj())


def ItoJ(C, particle, typeC=None):
    """
    Transform the left mixed Green's function into the right mixed.

    Parameters
    ----------
    C : ndarray
        Contour function.
    particle : int, bool, str
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion.
    typeC : str, optional
        Specify if C is a matrix, "M", or an interaction tensor, "T". Default is None.

    Returns
    -------
    ndarray
        Contour function.

    """
    if len(C.shape)==4 or typeC=="M":
        return -Matsubaraflip(np.swapaxes(C, 0, 1).T.conj(), particle)
    elif len(C.shape)==6 or typeC=="T":
        return -Matsubaraflip(np.einsum("stmnlk->tsklnm", C.conj()), particle)