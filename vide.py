from .imports import *


def vide_start(interpol, p, q, K, y0, p_sum_criterion, K_sum_criterion, conjugate=False):
    """
    Computes the initial bootstraping of a function described by a Volterra integro-differential equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    p : ndarray
        Weight (one-time-label array).
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (zero-time-label array).
    p_sum_criterion : tuple, str
        Sumation criterion between weight and bootstrapped function.
    K_sum_criterion : tuple, str
        Sumation criterion between kernell and bootstrapped function.
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    p = p[:interpol.k+1]
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.k+1]
    
    time_size = q.shape[0]
    source_orbital_shape = q.shape[1:]
    kernell_orbital_shape = K.shape[2:]
    weight_orbital_shape = p.shape[1:]
    
    #Codifying time indices and orbital indices when functions are vectorized
    idxs = np.indices(q.shape).reshape((len(q.shape),q.size))
    rows = np.einsum("...i,j->...ij", idxs, np.ones(idxs.shape[1], dtype=idxs.dtype))
    cols = np.einsum("...i,j->...ji", idxs, np.ones(idxs.shape[1], dtype=idxs.dtype))
    
    wh_same_orbitals = np.where(np.prod([rows[k]==cols[k] for k in range(1,idxs.shape[0])]), axis=0)
    wh_same_times = np.where(rows[0]==cols[0])
    
    if type(p_sum_criterion) is str:
        p_sum_criterion = p_sum_criterion.replace("->",",").split(",")
    if type(K_sum_criterion) is str:
        K_sum_criterion = K_sum_criterion.replace("->",",").split(",")
    
    if conjugate:
        K = np.swapaxes(K,0,1)
        p_sum_criterion = list(np.array(p_sum_criterion)[np.array([1,0,2])])
        K_sum_criterion = list(np.array(K_sum_criterion)[np.array([1,0,2])])
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    pq_coincidences = np.array([[p_sum_criterion[0].index(e), p_sum_criterion[2].index(f)] for e in p_sum_criterion[0] for f in p_sum_criterion[2] if e==f])
    py_coincidences = np.array([[p_sum_criterion[0].index(e), p_sum_criterion[1].index(f)] for e in p_sum_criterion[0] for f in p_sum_criterion[1] if e==f])
    p_slices = np.zeros((len(weight_orbital_shape)+1,wh_same_times[0].size), dtype=idxs.dtype)
    p_slices[0] += rows[0][wh_same_times] #In VIE product with weight involves a delta on time
    p_slices[pq_coincidences[:,0]+1] += rows[pq_coincidences[:,1]+1][0][wh_same_times]
    p_slices[py_coincidences[:,0]+1] += cols[py_coincidences[:,1]+1][0][wh_same_times]
    p_slices = tuple(p_slices[k] for k in range(p_slices.shape[0])) #Slice works as tuple of arrays
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    Kq_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[2].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[2] if e==f])
    Ky_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[1].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[1] if e==f])
    Ks = K * interpol.s.reshape(interpol.s.shape+(1,)*len(kernell_orbital_shape)) #Multiplication by integration weights
    K_slices = np.zeros((len(kernell_orbital_shape)+2,q.size,q.size), dtype=idxs.dtype)
    K_slices[0] += rows
    K_slices[1] += cols
    K_slices[Kq_coincidences[:,0]+2] += rows[Kq_coincidences[:,1]+1][0]
    K_slices[Ky_coincidences[:,0]+2] += cols[Ky_coincidences[:,1]+1][0]
    K_slices = tuple(K_slices[k] for k in range(K_slices.shape[0])) #Slice works as tuple of arrays
    
    M = np.zeros((q.size,q.size), dtype=np.complex128)
    M[wh_same_orbitals] += interpol.D[rows[0][wh_same_orbitals],cols[0][wh_same_orbitals]] / interpol.h #The integral part involves a delta on indices
    M[wh_same_times] += p[p_slices]
    M += interpol.h * Ks[K_slices]
    
    Mcut = M[np.where((rows[0]>0)*(cols[0]>0))].reshape(((time_size-1)*np.prod(source_orbital_shape),)*2)
    Minit = M[np.where((rows[0]>0)*(cols[0]==0))].reshape(((time_size-1)*np.prod(source_orbital_shape),)+(np.prod(source_orbital_shape),)) @ y0.flatten()
    yboot = np.linalg.inv(Mcut) @ (q.flatten()-Minit)
    return yboot.reshape((time_size-1,)+source_orbital_shape)


def vide_step(interpol, p, q, K, y, p_sum_criterion, K_sum_criterion, conjugate=False):
    """
    Computes the following time-step of a function described by a Volterra integro-differential equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    p : ndarray
        Weight (one-time-label array).
        Its length indicates the current time-steps computed.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y : ndarray
        Function to solve (one-time-label array).
    p_sum_criterion : tuple, str
        Sumation criterion between weight and bootstrapped function.
    K_sum_criterion : tuple, str
        Sumation criterion between kernell and bootstrapped function.
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default is False.

    Returns
    -------
    ndarray
        Next time-step (zero-time-label array).

    """
    n = y.shape[0] #Current time-step
    w = interpol.gregory_weights(n)[n,:] #Only current time-step, one-time-label
    
    if conjugate:
        K = np.swapaxes(K,0,1)
    
    p = p[n] #Only curent time-step, zero-time-label
    K = K[n,:n+1] #Only current time-step, one-time-label
    q = q[n] #Only current time-step, zero-time label
    
    time_size = q.shape[0]
    source_orbital_shape = q.shape
    kernell_orbital_shape = K.shape[1:]
    weight_orbital_shape = p.shape
    
    #Codifying time indices and orbital indices when functions are vectorized
    idxs = np.indices((n+1,)+source_orbital_shape).reshape((len(y.shape),(n+1)*q.size))
    rows = np.einsum("...i,j->...ij", idxs[:,-q.size:], np.ones(idxs.shape[1], dtype=idxs.dtype))
    cols = np.einsum("...i,j->...ji", idxs, np.ones(q.size, dtype=idxs.dtype))
    
    wh_same_orbitals = np.where(np.prod([rows[k]==cols[k] for k in range(1,idxs.shape[0])], axis=0))
    wh_same_times = np.where(rows[0]==cols[0])
    
    if type(p_sum_criterion) is str:
        p_sum_criterion = p_sum_criterion.replace("->",",").split(",")
    if type(K_sum_criterion) is str:
        K_sum_criterion = K_sum_criterion.replace("->",",").split(",")
    
    if conjugate:
        p_sum_criterion = list(np.array(p_sum_criterion)[np.array([1,0,2])])
        K_sum_criterion = list(np.array(K_sum_criterion)[np.array([1,0,2])])
    
    #Back propagation weights reordered
    a = np.flip(np.append(interpol.a, np.zeros(n-interpol.k)))
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    pq_coincidences = np.array([[p_sum_criterion[0].index(e), p_sum_criterion[2].index(f)] for e in p_sum_criterion[0] for f in p_sum_criterion[2] if e==f])
    py_coincidences = np.array([[p_sum_criterion[0].index(e), p_sum_criterion[1].index(f)] for e in p_sum_criterion[0] for f in p_sum_criterion[1] if e==f])
    p_slices = np.zeros((len(weight_orbital_shape),wh_same_times[0].size), dtype=idxs.dtype)
    p_slices[0] += rows[0][wh_same_times] #In VIE product with weight involves a delta on time
    p_slices[pq_coincidences[:,0]] += rows[pq_coincidences[:,1]+1][0][wh_same_times]
    p_slices[py_coincidences[:,0]] += cols[py_coincidences[:,1]+1][0][wh_same_times]
    p_slices = tuple(p_slices[k] for k in range(p_slices.shape[0])) #Slice works as tuple of arrays
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    Kq_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[2].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[2] if e==f])
    Ky_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[1].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[1] if e==f])
    Kw = K * w.reshape(w.shape+(1,)*len(kernell_orbital_shape)) #Multiplication by integration weights
    K_slices = np.zeros((len(kernell_orbital_shape)+1,q.size,q.size*(n+1)), dtype=idxs.dtype)
    K_slices[0] += cols
    K_slices[Kq_coincidences[:,0]+1] += rows[Kq_coincidences[:,1]+1][0]
    K_slices[Ky_coincidences[:,0]+1] += cols[Ky_coincidences[:,1]+1][0]
    K_slices = tuple(K_slices[k] for k in range(K_slices.shape[0])) #Slice works as tuple of arrays
    
    M = np.zeros((q.size,q.size*(n+1)), dtype=np.complex128)
    M[wh_same_orbitals] += a[cols[0][wh_same_orbitals]] / interpol.h #The integral part involves a delta on indices
    M[wh_same_times] += p[p_slices]
    M += interpol.h * Kw[K_slices]
    
    Mcut = M[np.where(cols[0]<n)].reshape((q.size,(time_size-1)*q.size)) @ y[:n].flatten()
    Mfinal = M[np.where(cols[0]==n)].reshape(source_orbital_shape)
    ystep = np.linalg.inv(Mfinal) @ (q.flatten()-Mcut)
    return ystep.reshape(source_orbital_shape)


def vie_start(interpol, q, K, y0, K_sum_criterion, conjugate=False):
    """
    Computes the initial bootstraping of a function described by a Volterra integral equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y0 : ndarray
        Initial condition (zero-time-label array).
    K_sum_criterion : tuple, str
        Sumation criterion between kernell and bootstrapped function.
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default id False.

    Returns
    -------
    ndarray
        Bootstrapped function (one-time-label array).

    """
    q = q[:interpol.k+1]
    K = K[:interpol.k+1,:interpol.K+1]
    
    time_size = q.shape[0]
    source_orbital_shape = q.shape[1:]
    kernell_orbital_shape = K.shape[2:]
    
    #Codifying time indices and orbital indices when functions are vectorized
    idxs = np.indices(q.shape).reshape((len(q.shape),q.size))
    rows = np.einsum("...i,j->...ij", idxs, np.ones(idxs.shape[1], dtype=idxs.dtype))
    cols = np.einsum("...i,j->...ji", idxs, np.ones(idxs.shape[1], dtype=idxs.dtype))
    
    if type(K_sum_criterion) is str:
        K_sum_criterion = K_sum_criterion.replace("->",",").split(",")
    
    if conjugate:
        K = np.swapaxes(K,0,1)
        K_sum_criterion = list(np.array(K_sum_criterion)[np.array([1,0,2])])
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    Kq_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[2].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[2] if e==f])
    Ky_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[1].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[1] if e==f])
    Ks = K * interpol.s.reshape(s.shape+(1,)*len(kernell_orbital_shape)) #Multiplication by integration weights
    K_slices = np.zeros((len(kernell_orbital_shape)+2,q.size,q.size), dtype=idxs.dtype)
    K_slices[0] += rows
    K_slices[1] += cols
    K_slices[Kq_coincidences[:,0]+2] += rows[Kq_coincidences[:,1]+1][0]
    K_slices[Ky_coincidences[:,0]+2] += cols[Ky_coincidences[:,1]+1][0]
    K_slices = tuple(K_slices[k] for k in range(K_slices.shape[0])) #Slice works as tuple of arrays
    
    M = np.zeros((q.size,q.size), dtype=np.complex128)
    M += np.eye(M.shape) #In VIE there is a delta in time and orbitals
    M += interpol.h * Ks[K_slices]
    
    Mcut = M[np.where((rows[0]>0)*(cols[0]>0))].reshape(((time_size-1)*np.prod(source_orbital_shape),)*2)
    Minit = M[np.where((rows[0]>0)*(cols[0]==0))].reshape(((time_size-1)*np.prod(source_orbital_shape),)+(np.prod(source_orbital_shape),)) @ y0.flatten()
    yboot = np.linalg.inv(Mcut) @ (q.flatten()-Minit)
    return yboot.reshape((time_size-1,)+source_orbital_shape)


def vie_step(interpol, q, K, y, K_sum_criterion, conjugate=False):
    """
    Computes the following of a function described by a Volterra integral equation.
    
    Parameters
    ----------
    interpol : Interpolator
        Class with interpolation weights.
    q : ndarray
        Source (one-time-label array).
    K : ndarray
        Kernell (two-time-label array).
    y : ndarray
        Function to solve (one-time-label array).
        Its length indicates the current time-steps computed.
    K_sum_criterion : tuple, str
        Sumation criterion between kernell and bootstrapped function.
    conjugate : boolean, optional
        If True conjugate VIDE is computed. Default if False.

    Returns
    -------
    ndarray
        Next time-step (zero-time-label array).

    """
    n = y.shape[0] #Current time-step
    w = interpol.gregory_weights(n)[n,:] #Only current time-step, one-time-label
    
    if conjugate:
        K = np.swapaxes(K,0,1)
    
    K = K[n,:] #Only current time-step, one-time-label
    q = q[n] #Only current time-step, zero-time label
    
    time_size = q.shape[0]
    source_orbital_shape = q.shape
    kernell_orbital_shape = K.shape[1:]
    
    #Codifying time indices and orbital indices when functions are vectorized
    idxs = np.indices((n+1,)+source_orbital_shape).reshape((len(y.shape),(n+1)*q.size))
    rows = np.einsum("...i,j->...ij", idxs[:,-q.size:], np.ones(idxs.shape[1], dtype=idxs.dtype))
    cols = np.einsum("...i,j->...ji", idxs, np.ones(q.size, dtype=idxs.dtype))
    
    if type(K_sum_criterion) is str:
        K_sum_criterion = K_sum_criterion.replace("->",",").split(",")
    
    if conjugate:
        K_sum_criterion = list(np.array(K_sum_criterion)[np.array([1,0,2])])
    
    #Index coincidence with source correspond to a row and coincidence with y corresponds to a column by matrix product rules
    Kq_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[2].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[2] if e==f])
    Ky_coincidences = np.array([[K_sum_criterion[0].index(e), K_sum_criterion[1].index(f)] for e in K_sum_criterion[0] for e in K_sum_criterion[1] if e==f])
    Kw = K * w.reshape(w.shape+(1,)*len(kernell_orbital_shape)) #Multiplication by integration weights
    K_slices = np.zeros((len(kernell_orbital_shape)+1,q.size,q.size*(n+1)), dtype=idxs.dtype)
    K_slices[0] += cols
    K_slices[Kq_coincidences[:,0]+1] += rows[Kq_coincidences[:,1]+1][0]
    K_slices[Ky_coincidences[:,0]+1] += cols[Ky_coincidences[:,1]+1][0]
    K_slices = tuple(K_slices[k] for k in range(K_slices.shape[0])) #Slice works as tuple of arrays
    
    M = np.zeros((q.size,q.size*(n+1)), dtype=np.complex128)
    wh_all_same = np.where(np.prod([rows[k]==cols[k] for k in range(idxs.shape[0])], axis=0))
    M[wh_all_same] += 1 #In VIE there is a delta in time and orbitals
    M += interpol.h * Kw[K_slices]
    
    Mcut = M[np.where(cols[0]<n)].reshape((q.size,(time_size-1)*q.size)) @ y[:n].flatten()
    Mfinal = M[np.where(cols[0]==n)].reshape(source_orbital_shape)
    ystep = np.linalg.inv(Mfinal) @ (q.flatten()-Mcut)
    return ystep.reshape(source_orbital_shape)
