from .imports import *


def tensor_inv(T):
    """
    Performs the tensor inverse under ijkl,jmln->imkn sum criterion.
    
    Parameters
    ----------
    T : ndarray
        Tensor to inverse, with shape (...,N,N,N,N).
    
    Returns
    ----------
    ndarray
        Tensor inverse.
    """
    N = T.shape[-1]
    a = np.array([l+j*N*N for j in range(N) for l in range(N)])
    q,p = np.meshgrid(a,a)
    p_basis = np.array([0*p,p//(N*N),0*p,p%(N*N)]).transpose(1,2,0)
    q_basis = np.array([0*q,q//(N*N),0*q,q%(N*N)]).transpose(1,2,0)
    M = T[...,q_basis[:,:,3],p_basis[:,:,1],q_basis[:,:,1],p_basis[:,:,3]]
    invM = np.linalg.inv(M)
    m,l,j,n = np.meshgrid(*tuple([[k for k in range(N)]]*4))
    pp = j*N*N+l
    qq = m*N*N+n
    return invM[...,np.sum([k*(qq==a[k]) for k in range(a.size)], axis=0),np.sum([k*(pp==a[k]) for k in range(a.size)], axis=0)]