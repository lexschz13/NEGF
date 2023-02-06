from .imports import *


class Interpolator:
    """
    Interpolator class computes and storage the interpolation weights and the corresponding for derivation, backward differentiation, gregory integration and convolution.
    
    Parameters
    ----------
    k : int
        Order of interpolation.
    h : float, optional
        Time step of samples to interpolate. Default is 1.
    
    """
    def __init__(self, k, h=1):
        aux = np.arange(k+1)
        col,row = np.meshgrid(aux,aux)
        self.k = k
        self.h = h
        self.P = np.linalg.inv(row**col)
        self.D = (col[:,1:] * row[:,1:]**(col[:,1:]-1)) @ self.P[1:,:]
        self.a = -self.P[1,:]
        self.s = (row**(col+1)/(col+1)) @ self.P
        self.w = np.copy(self.s)
    
    # def gregory_weights_matrix(self, n):
    #     if n<=self.k:
    #         return self.s[:n+1,:n+1]
        
    #     S = np.zeros((n+1,n+1))
    #     A = np.zeros((n+1,n+1))
    #     S[np.arange(self.k+1,n+1),np.arange(self.k+1,n+1)] += 1
    #     A[np.arange(self.k+1),np.arange(self.k+1)] += self.h
    #     S[:self.k+1,:self.k+1] += self.h*self.s
    #     for m in range(self.k+1,n+1):
    #         A[m,m:m-self.k-1:-1] += self.a
    #     return np.linalg.inv(A) @ S
    
    def gregory_weights(self, n):
        """
        Computes the gregory integration wiegths by an iterative method.

        Parameters
        ----------
        n : int
            Integration order.

        Returns
        -------
        ndarray
            Gregory weights since order n.

        """
        if n+1 <= self.w.shape[0]:
            return self.w[:n+1,:n+1]
        
        for k in range(n+1 - self.w.shape[0]):
            self.__gregory_add()
        return self.w
    
    def __gregory_add(self):
        """
        Adds a new row to gregory weigths matrix.

        Returns
        -------
        None.

        """
        m = self.w.shape[0]
        new_row = np.zeros(m+1)
        new_row[-1] = 1/self.a[0]
        new_row[:-1] = -new_row[-1]*np.sum(self.a[1:][:,None]*self.w[m-1:m-self.k-1:-1,:], axis=0)
        self.w = np.append(self.w, np.zeros(m)[:,None], axis=1)
        self.w = np.append(self.w, new_row[None,:], axis=0)
    
    def conv_weights(self, m):
        """
        Computes the convolution integration weights from t=0 to jh with j=0,...,m.

        Parameters
        ----------
        m : int
            Time steps to convolve.

        Returns
        -------
        ndarray
            Convolution weights.

        """
        a = np.einsum("i,j->ij", np.arange(self.k+1), np.ones(self.k+1))
        b = np.einsum("i,j->ji", np.arange(self.k+1), np.ones(self.k+1))
        factor = (-1)**b * factorial(a) * factorial(b) * m**(a+b+1) / factorial(a+b+1)
        return np.einsum("ar,bs,ab->rs", self.P, self.P, factor)
