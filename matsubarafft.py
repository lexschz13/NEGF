from .imports import *


def __fermiMatsubaraifft(a, beta=1):
    """
    Computes IFT for a imaginary-time function for fermions.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.

    Returns
    -------
    ndarray
        Transformed function.

    """
    N = a.shape[0]
    aux = np.zeros((2*N,)+a.shape[1:], dtype=np.complex128)
    aux[1::2,...] = np.copy(a) #odd frequencies
    return np.fft.fft(np.fft.ifftshift(aux, axes=0), axis=0)[:N,...]/beta


def __boseMatsubaraifft(a, beta=1):
    """
    Computes IFT for a imaginary-time function for bosons.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.

    Returns
    -------
    ndarray
        Transformed function.

    """
    N = a.shape[0]
    aux = np.zeros((2*N,)+a.shape[1:], dtype=np.complex128)
    aux[::2,...] = np.copy(a) #even frequencies
    return np.fft.fft(np.fft.ifftshift(aux, axes=0), axis=0)[:N,...]/beta


def __fermiMatsubarafft(a, beta=1):
    """
    Computes FT for a imaginary-time function for fermions.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.

    Returns
    -------
    ndarray
        Transformed function.

    """
    N = a.shape[0]
    aux = np.zeros((2*N,)+a.shape[1:], dtype=np.complex128)
    aux[:N] = np.copy(a)
    aux[N:] = -a
    return np.fft.fftshift(np.fft.ifft(aux, axis=0), axes=0)[1::2,...]*beta #odd frequencies


def __boseMatsubarafft(a, beta=1):
    """
    Computes FT for a imaginary-time function for bosons.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.

    Returns
    -------
    ndarray
        Transformed function.

    """
    N = a.shape[0]
    aux = np.zeros((2*N,)+a.shape[1:], dtype=np.complex128)
    aux[:N] = np.copy(a)
    aux[N:] = a
    return np.fft.fftshift(np.fft.ifft(aux, axis=0), axes=0)[::2,...]*beta #even frequencies


def Matsubaraifft(a, beta=1, particle=0):
    """
    Computes IFT for a imaginary-time function.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.
    particle : int, bool, str, optional
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion. The default is 0.

    Returns
    -------
    ndarray
        Transformed function.

    """
    if particle in [0,False,"Boson"]:
        return __boseMatsubaraifft(a, beta)
    elif particle in [1,True,"Fermion"]:
        return __fermiMatsubaraifft(a, beta)


def Matsubarafft(a, beta=1, particle=0):
    """
    Computes IFT for a imaginary-time function.
    Transform is done in axis=0.

    Parameters
    ----------
    a : ndarray
        Function to transform.
    beta : float, optional
        Inverse temperature. The default is 1.
    particle : int, bool, str, optional
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion. The default is 0.

    Returns
    -------
    ndarray
        Transformed function.

    """
    if particle in [0,False,"Boson"]:
        return __boseMatsubarafft(a, beta)
    elif particle in [1,True,"Fermion"]:
        return __fermiMatsubarafft(a, beta)


def Matsubaraflip(a, particle):
    """
    Given a time-imaginary function A(t) computes the negative time-imaginary function A(beta-t).

    Parameters
    ----------
    a : ndarray
        Function to flip.
    particle : int, bool, str
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion.

    Returns
    -------
    ndarray
        Flipped function.

    """
    if particle in [0,False,"Boson"]:
        particle = 0
    elif particle in [1,True,"Fermion"]:
        particle = 1
    return (-1)**particle * np.append(a[0][None,...], a[1:][::-1], axis=0)


def bubbleM(subscripts, a, b, mode=True, beta=1, atype=0, btype=0):
    """
    Computes the bubble product of two functions of Matsubara frequencies.

    Parameters
    ----------
    subscripts : str
        Sum criterion.
    a : ndarray
        Time-imaginary function in Matsubara frequency space.
    b : ndarray
        Time-imaginary function in Matsubara frequency space.
    mode : bool, optional
        Determines the kind of bubble: True for A(t,t')B(t,t') and False for A(t,t')B(t,t). The default is True.
    beta : float, optional
        Inverse temperature. The default is 1.
    atype : int, bool, str, optional
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion. The default is 0.
    btype : int, bool, str, optional
        Sets particle type. If 0,False,"Boson" is a boson and if 1,True,"Fermion" is a fermion. The default is 0.

    Returns
    -------
    ndarray
        Bubble in Matsubara frequency space.

    """
    subsa,subsbc = subscripts.split(",")
    subsb,subsc = subsbc.split("->")
    subscripts = "..."+subsa+",..."+subsb+"->..."+subsc
    
    if atype in [0,False,"Boson"]:
        atype = 0
    elif atype in [1,True,"Fermion"]:
        atype = 1
    
    if btype in [0,False,"Boson"]:
        btype = 0
    elif atype in [1,True,"Fermion"]:
        btype = 1
    
    ctype = atype^btype
    
    ta = Matsubaraifft(a, beta, atype)
    tb = Matsubaraifft(b, beta, btype) if mode else Matsubaraifft(((-1)**btype)*b[::-1], beta, btype)
    tc = np.einsum(subscripts, ta, tb)
    return Matsubarafft(tc, beta, ctype)
