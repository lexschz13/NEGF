from .imports import *
from .vide import *
from .interpolator import *
from .interact_tensor import *
from .matsubarafft import *
from .conjugates import *
from time import time
from joblib import Parallel,delayed # Paralelization



class Solver:
    """
    Class which stores Green's functions and self-energies and solve them given a Hamiltonian.
    
    Attributes
    ----------
    beta : float
        Inverse temperature.
    Nt : int
        Time steps for real branch.
    Nmats : int
        Time steps for imaginary branch.
    ht : float
        Time interval for real branch.
    hmats : float
        Time interval for imaginary branch.
    t : ndarray
        Real times where functions are defined.
    tau : ndarray
        Imaginary times where functions are defined.
    interpol : Interpolator
        Interpolator for real branch.
    interpolM : Interpolator.
        Interpolator for imaginary branch.
    H0 : ndarray
        Time-independent quadratic term of the Hamiltonian.
    Ht : ndarray
        Time-dependent quadratic term of the Hamiltonian.
    V : ndarray
        Interaction term of the Hamiltonian (time-independent).
    particle : int
        Indicates particle type: 0 for boson and 1 for fermion.
    stat_factor : int
        Sign related with Bose-Einstein statistics (+1) if Fermi-Diracstatistics (-1).
    GM : ndarray
        Matsubara Green's function.
    GR : ndarray
        Retarded Green's function.
    GI : ndarray
        Left-mixed Green's function.
    GL : ndarray
        Lesser Green's function.
    SHF : ndarray
        Hartree-Fock self-energy.
    SM : ndarray
        Matsubara self-energy.
    SR : ndarray
        Retarded self-energy.
    SI : ndarray
        Left-mixed self-energy.
    SL : ndarray
        Lesser self-energy.
    PM : ndarray
        Matsubara polarization bubble.
    PR : ndarray
        Retarded polarization bubble.
    PI : ndarray
        Left-mixed polarization bubble.
    PL : ndarray
        Lesser polarization bubble.
    WM : ndarray
        Matsubara screen interaction.
    WR : ndarray
        Retarded screen interaction.
    WI : ndarray
        Left-mixed screen interaction.
    WL : ndarray
        Lesser screen interaction.
    mats_freq : ndarray
        Matsubara frequancies for the given particle (boson or fermion).
    g : ndarray
        Non-interactive equilibrium Green's function of the correspondent H0.
    convergence_limit : float
        Maximum error between old and new functions to consider convergence.
    max_iter : int
        Iterations where convergence achievement is expected.
    
    Methods
    ----------
    solve():
        Solve Green's functions by self-consistend method.
    
    """
    def __init__(self, Nt, Nmats, ht, inter_order, T, H0, Ht, V, particle, convergence_limit=1e-12, max_iter=1e4):
        """
        Initialize solver.

        Parameters
        ----------
        Nt : int
            Time steps for real branch.
        Nmats : int
            Time steps for imaginary branch.
        ht : float
            Time interval for real branch.
        inter_order : int
            Interpolation order.
        T : float
            Temperature.
        H0 : ndarray
            Time-independent quadratic term of the Hamiltonian.
        Ht : function
            Time-dependent quadratic term of the Hamiltonian. Returns ndarray.
        V : ndarray
            Interaction term of the Hamiltonian.
        particle : int
            Indicates particle type: 0 for boson and 1 for fermion.
        convergence_limit : float
            Maximum error between old and new functions to consider convergence.
        max_iter : int
            Iterations where convergence achievement is expected.

        Raises
        ------
        ValueError
            Non-valid value to describe particle type.
            For boson: 0,False,"Boson".
            For fermion: 1,True,"Fermion".

        Returns
        -------
        None.

        """
        kB = 8.617333262e-5
        self.beta = 1/(kB*T)
        
        self.Nt = Nt
        self.Nmats = Nmats
        self.ht = ht
        self.interpol = Interpolator(inter_order, ht)
        
        self.t = np.arange(0,Nt*ht,ht)
        self.tau = np.linspace(0,beta,Nmats)
        self.htau = self.tau[1]-self.tau[0]
        self.interpolM = Interpolator(inter_order, self.htau)
        
        self.H0 = H0
        self.Ht = Ht(self.t) #Time function
        self.V = V
        
        self.__delta_time_orb = (np.eye(Nt)[...,None,None]*np.eye(H0.shape[0])[None,None,...]).astype(np.complex128) #Usefull for vide
        
        if particle in [0,False,"Boson"]:
            self.particle = 0
        elif particle in [1,True,"Fermion"]:
            self.particle = 1
        else:
            raise ValueError("Non valid particle type. See documentation for more.")
        self.stat_factor = (-1)**particle
        
        useHDD = kwargs.get("useHDD", False)
        if useHDD:
            pass
        else:
            #Green's functions
            self.GM = np.zeros((Nmats,)+H.shape, dtype=np.complex128)
            self.GI = np.zeros((Nt,Nmats)+H.shape, dtype=np.complex128)
            self.GR = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            self.GL = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            #Self energies
            self.SHF = np.zeros((Nt,)+H.shape, dtype=np.complex128)
            self.SM = np.zeros((Nmats,)+H.shape, dtype=np.complex128)
            self.SI = np.zeros((Nt,Nmats)+H.shape, dtype=np.complex128)
            self.SR = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            self.SL = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            #Polarizatin bubbles
            self.PM = np.zeros((Nmats,)+H.shape*2, dtype=np.complex128)
            self.PI = np.zeros((Nt,Nmats)+H.shape*2, dtype=np.complex128)
            self.PR = np.zeros((Nt,Nt)+H.shape*2, dtype=np.complex128)
            self.PL = np.zeros((Nt,Nt)+H.shape*2, dtype=np.complex128)
            #Screened interactions
            self.WM = np.zeros((Nmats,)+H.shape*2, dtype=np.complex128)
            self.WI = np.zeros((Nt,Nmats)+H.shape*2, dtype=np.complex128)
            self.WR = np.zeros((Nt,Nt)+H.shape*2, dtype=np.complex128)
            self.WL = np.zeros((Nt,Nt)+H.shape*2, dtype=np.complex128)
            #VIDE sources
            self.__QI = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            self.__QL = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            #VIE sources
            self.__ER = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            self.__EI = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
            self.__EL = np.zeros((Nt,Nt)+H.shape, dtype=np.complex128)
        
        #Non-interactive G
        k = np.arange(2*Nmats) - Nmats
        self.mats_freq = k[[slice(None,None,2),slice(1,None,2)][particle]]*np.pi/self.beta
        self.g = Matsubaraifft(np.linalg.inv(1j*self.mats_freq[:,None,None]*np.eye(H0.shape[0]) - H0[None,:,:]), self.beta, self.particle)
        
        #GR initial condition
        GRidxs = np.indices(GR.shape)
        GR[np.where((GRidxs[0]==GRidxs[1])*(GRidxs[2]==GRidxs[3]))] -= 1j
        
        self.convergence_limit = convergence_limit
        self.max_iter = kwargs = max_iter
    
    
    def __hartree_fock_init(self):
        """
        Solve initial Hartree-Fock self-energy in Matsubara frequency space.
        
        Raises
        ------
        RuntimeError
            If number of iterations reaches which is specified convergence is considered unachivable.
        
        Returns
        -----------
        None.
        
        """
        print("Initializing Hartree-Fock self-energy")
        count = 0
        while True:
            gHF = Matsubaraifft(np.linalg.inv(1j*self.mats_freq[:,None,None]*np.eye(H0.shape[0]) - H0[None,:,:] - self.SHF[0][None,:,:]), self.beta, self.particle)
            #gHF_at_beta = self.stat_factor*gHF[0]
            newSHF = -np.einsum("lk,mkln->mn", gHF[0], self.V-np.swapaxes(self.V,-1,-2))
            distance = np.sum(abs(newSHF-self.SHF[0]))
            self.SHF[0] = np.copy(newSHF)
            count += 1
            if distance<=self.convergence_limit:
                print("Hartree-Fock self-energy initialized after %i iterations" % count)
                break
            if count==self.max_iter:
                raise RuntimeError("Convergence not get at required iterations")
    
    
    def __G_init(self):
        """
        Solve Matsubara branch of Green's function and self-energy in frequency space.

        Raises
        ------
        RuntimeError
            If number of iterations reaches which is specified convergence is considered unachivable.
        
        Returns
        -----------
        None.

        """
        print("Initializing Matsubara branch")
        count = 0
        while True:
            self.GM = Matsubaraifft(np.linalg.inv(1j*self.mats_freq[:,None,None]*np.eye(H0.shape[0]) - H0[None,:,:] - self.SHF[0][None,:,:] - self.SM), self.beta, self.particle)
            self.PM = -np.einsum("...ln,...km->...lkmn", self.GM, Matsubaraflip(self.GM, self.beta, self.particle))
            fP = Matsubarafft(self.PM, self.beta, 0) #Bubble is always a boson
            fZ = np.einsum("...iakb,...bcad->...ickd", self.V, fP)
            I = np.einsum("il,jk->ijkl", np.eye(self.V.shape[0]), np.eye(self.V.shape[0]))
            #Frequency space
            #W_ijkl - Z_ickd W_djcl = Z_ickd v_djcl
            #(I-Z)_ickd W_djcl = Z_ickd v_djcl
            #W_djcl = inv(I-Z)_dkci (Zv)_ijkl
            #fW = np.einsum("...dkci,...ijkl->...djcl", tensor_inv(I-Z), np.einsum("...ickd,...djcl->...ijkl", Z, v))
            fW = np.einsum("...akbi,...ickd,...djcl->ajbl", tensor_inv(I-Z), Z, v)
            self.WM = Matsubaraifft(fW, self.beta, 0) #Interaction is always a boson
            newSM = np.einsum("lk,mkln->mn", self.GM, self.WM)
            distance = np.sum(abs(newSM-self.SM))
            self.SM = np.copy(newSM)
            count += 1
            if distance<=self.convergence_limit:
                print("Matsubara branch computed after %i iterations" % count)
                break
            if count==self.max_iter:
                raise RuntimeError("Convergence not get at required iterations")
        print("Recomputing Matsubara Green's function after convergence")
        self.GM = Matsubaraifft(np.linalg.inv(1j*self.mats_freq[:,None,None]*np.eye(H0.shape[0]) - H0[None,:,:] - self.SHF[0][None,:,:] - self.SM), self.beta, self.particle)
        self.GI[0,:] = 1j*Matsubaraflip(self.GM, self.particle) #Initialize left-mixed
    
    
    def __boot_iter_GR(self):
        """
        Performs the bootstraping for retarded Green's function with a given self-energy.

        Returns
        -------
        None.

        """
        # for n in range(self.Nt-1):
        def pooled(i, n):
            #Over indiced could happen near edges, conditional slice avoids it
            slc = slice(None) if n+self.interpol.k+1<=self.Nt else slice(None,self.Nt-n-interpol.k-1)
            self.GR[n+1:max(n+self.interpol.k+1,self.Nt),n] = vide_start(self.interpol,
                                                            1j*(self.H0[None,...]+self.Ht[n:]+self.SHF[n:]), #p
                                                            -1j*self.__delta_time_orb[n:,n], #q
                                                            1j*self.SR[n:,n:], #K
                                                            self.GR[n,n], #y0
                                                            "ij,jk->ik", "ij,jk->ik", #sum criterions
                                                            conjugate=False)[slc]
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(self.Nt-1)))
    
    
    def __step_iter_GR(self, n):
        """
        Performs the n-th time step for retarded Green's function with a given self-energy.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(self.Nt-n):
        def pooled(i, m):
            self.GR[m+n,m] = vide_step(self.interpol,
                             1j*(H0[None,...]+Ht[m:]+self.SHF[m:]), #p
                             -1j*self.__delta_time_orb[m:], #q
                             1j*self.SR[m:,m:], #K
                             self.GR[m:m+n,m], #y
                             "ij,jk-ik", "ij,jk,ik", #sum criterions
                             conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(self.Nt-n)))
    
    
    def __boot_iter_GI(self):
        """
        Performs the bootstraping for left-mixed Green's function with a given self-energy.

        Returns
        -------
        None.

        """
        self.__QI = np.swapaxes(Matsubaraifft(np.einsum("...ij,...jk->...ik", Matsubarafft(np.swapaxes(self.SI, 0, 1), self.beta, self.particle), Matsubarafft(Matsubaraflip(self.GM,self.particle), self.beta, self.particle)), self.beta, self.particle), 0, 1)
        # for n in range(self.Nmats):
        def pooled(i, n):
            self.GI[1:self.interpol.k+1,n] = vide_start(self.interpol,
                                                        1j*(self.H0[None,...]+self.Ht+self.SHF), #p
                                                        -1j*self.__QI[:,n], #q
                                                        1j*self.SR, #K
                                                        self.GI[0,n], #y0
                                                        "ij,jk->ik", "ij,jk->ik", #sum criterions
                                                         conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(self.Nmats)))
    
    
    def __step_iter_GI(self, n):
        """
        Performs the n-th time step for left-mixed Green's function with a given self-energy.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(self.Nmats):
        def pooled(i, m):
            self.GI[n,m] = vide_step(self.interpol,
                           1j*(self.H0[None,...] + self.Ht + self.SHF), #p
                           -1j*self.__QI[:,m], #q
                           1j*self.SR, #K
                           self.GI[:n,m], #y
                           "ij,jk->ik", "ij,jk->ik", #sum criterions
                           conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(self.Nmats)))
    
    
    def __boot_iter_GL(self):
        """
        Performs the bootstraping for lesser Green's function with a given self-energy.
        Retarded and left-mixed Green's functions are assumed to be already computed.

        Returns
        -------
        None.

        """
        self.GL[0,:] = -np.swapaxes(self.GI[:,0], -1, -2).conj() #Initialize GL
        GA = RtoA(self.GR)
        GJ = ItoJ(self.GI, self.particle)
        self.__QL = (self.ht*np.einsum("ml,slij,lmjk->smik", self.interpol.gregory_weights(self.Nt-1), self.SL, GA) -
              1j*self.htau*np.einsum("l,slij,lmjk->smik", self.interpolM.gregory_weights(self.Nmats-1)[-1,:], self.SI, GJ))
        # for n in range(1,self.Nt):
        def pooled(i, n):
            self.GL[1:min(self.interpol.k+1,n+1),n] = vide_start(self.interpol,
                                                        1j*(self.H0[None,...]+self.Ht+self.SHF), #p
                                                        -1j*self.__QL[:,n], #q
                                                        1j*self.SR, #K
                                                        self.GL[0,n], #y0
                                                        "ij,jk->ik", "ij,jk->ik", #sum criterions
                                                         conjugate=False)[:,min(self.interpol.k+1,n+1)] #Only upper trienangle is computed since it is hermitian
            self.GL[n,1:min(self.interpol.k+1,n+1)] = -self.GL[1:min(self.interpol.k+1,n+1),n].T.conj() #Updating lower triangle
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(1,self.Nt)))
    
    
    def __step_iter_GL(self, n):
        """
        Performs the n-th time step for lesser Green's function with a given self-energy.
        Retarded and left-mixed Green's functions are assumed to be already computed.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(n,self.Nt):
        def pooled(i, m):
            self.GL[n,m] = vide_step(self.interpol,
                           1j*(self.H0[None,...]+self.Ht+self.SHF), #p
                           -1j*self.__QL[:,m], #q
                           1j*self.SR, #K
                           self.GL[:n,m], #y
                           "ij,jk->ik", "ij,jk->ik", #sum criterions
                           conjugate=False)
            self.GL[n,m] = -self.GL[m,n].T.conj() #Updating lower triangle
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(n,self.Nt)))
    
    
    def __boot_iter_WR(self):
        """
        Performs the bootstraping for retarded screen interaction with a given Green's function.

        Returns
        -------
        None.

        """
        self.__ER = np.einsum("male,...exab,bnxk->...mnlk", self.V, PR, self.V)
        self.WR[np.arange(self.Nt),np.arange(self.Nt),...] = np.einsum("ss..->s...", self.__ER) #Initializing screen WR(t,t) = vPR(t,t)v
        # for n in range(self.Nt-1):
        def pooled(i, n):
            #Over indices could happen near edges, conditional slice avoids it
            slc = slice(None) if n+self.interpol.k+1<=self.Nt else slice(None,self.Nt-n-interpol.k-1)
            self.WR[n+1:max(n+self.interpol.k+1,self.Nt),n] = vie_start(self.interpol,
                                                            self.__ER[n:,n], #q
                                                            -np.einsum("male,...exab->mxlb", self.V, self.PR[n:,n:]), #K
                                                            self.WR[n,n], #y0
                                                            "male,exab->mxlb", #sum criterion
                                                            conjugate=False)[slc]
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(self.Nt-1)))
    
    
    def __step_iter_WR(self, n):
        """
        Performs the n-th time step for retarded screen interaction with a given Green's function.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(self.Nt-n):
        def pooled(i, m):
            self.WR[m+n,m] = vie_step(self.interpol,
                             self.__ER[m:,m], #q
                             -np.einsum("male,...exab->mxlb", self.V, self.PR[m:,m:]), #K
                             self.WR[m:m+n,m], #y
                             "male,exab->mxlb", #sum criterion
                             conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(self.Nt-n)))
    
    
    def __boot_iter_WI(self):
        """
        Performs the bootstraping for left-mixed screen interaction with a given Green's function.

        Returns
        -------
        None.

        """
        self.__EI = (np.einsum("male,...exab,bnxk->...mnlk", self.V, PI, self.V) +
              Matsubaraifft(np.einsum("male,swexab,wbnxk->swmnlk", self.V, np.swapaxes(Matsubarafft(np.swapaxes(self.PI,0,1), self.beta, 0),0,1), Matsubarafft(Matsubaraflip(self.WM,0), self.beta, 0)), self.beta, 0))
        self.WI[0,...] = __EI[0,...] #Initializing screen WI(0,t') = EI(0,t')
        # for n in range(self.Nmats):
        def pooled(i, n):
            self.WI[1:self.interpol.k+1,n] = vie_start(self.interpol,
                                                        self.__EI[:,n], #q
                                                        -np.einsum("male,...exab->mxlb", self.V, self.PR), #K
                                                        self.WI[0,n], #y0
                                                        "male,exab->mxlb", #sum criterion
                                                         conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(self.Nmats)))
    
    
    def __step_iter_WI(self, n):
        """
        Performs the n-th time step for left-mixed screen interaction with a given Green's function.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(self.Nmats):
        def pooled(i, m):
            self.WI[n,m] = vie_start(self.interpol,
                                     self.__EI[:,m], #q
                                     -np.einsum("male,...exab->mxlb", self.V, self.PR), #K
                                     self.WI[:n,m], #y
                                     "male,exab->mxlb", #sum criterion
                                     conjugate=False)
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(self.Nmats)))
    
    
    def __boot_iter_WL(self):
        """
        Performs the bootstraping for lesser screen interaction with a given Green's function.
        Retarded and left-mixed screen interactions are assumed to be already computed.

        Returns
        -------
        None.

        """
        self.__EL = (np.einsum("male,...exab,bnxk->...mnlk", self.V, PL, self.V) +
              np.einsum("gf,male,sfexab,fgbnxk->sgmnlk", self.interpol.gregory_wieghts(self.Nt-1), self.V, self.PL, RtoA(self.WR)) -
              1j*np.einsum("f,male,sfexab,fgbnxk->sgmnlk", self.interpolM.gregory_weights(self.Nmats-1)[-1,:], self.V, self.PI, ItoJ(self.WI, 0)))
        self.WL[0,...] = __EL[0,...] #Initializing screen WL(0,t') = vPL(0,t')v - i integ(0 to beta)[vPI(0,tau)WJ(tau,t')dtau]
        # for n in range(1,self.Nt):
        def pooled(i, n):
            self.WL[1:min(self.interpol.k+1,n+1),n] = vie_start(self.interpol,
                                                        self.__EL[:,n], #q
                                                        -np.einsum("male,...exab,->mxlb", self.V, self.PR), #K
                                                        self.WL[0,n], #y0
                                                        "male,exab->mxlb", #sum criterion
                                                         conjugate=False)[:,min(self.interpol.k+1,n+1)] #Only upper trienangle is computed since it is hermitian
            self.WL[n,1:min(self.interpol.k+1,n+1)] = -np.einsum("...klnm->...mnlk", self.WL[1:min(self.interpol.k+1,n+1),n]).conj() #Updating lower triangle
        Parallel(n_jobs=-1)(delayed(pooled)(i,n) for i,n in enumerate(range(1,self.Nt)))
    
    
    def __step_iter_WL(self, n):
        """
        Performs the n-th time step for lesser screen interaction with a given Green's function.
        Retarded and left-mixed screen interactions are assumed to be already computed.

        Parameters
        ----------
        n : int
            Time step.

        Returns
        -------
        None.

        """
        # for m in range(n,self.Nt):
        def pooled(i, m):
            self.WL[n,m] = vie_start(self.interpol,
                                     self.__EL[:,m], #q
                                     -np.einsum("male,...exab,->mxlb", self.V, self.PR), #K
                                     self.WL[:n,m], #y
                                     "male,exab->mxlb", #sum criterion
                                     conjugate=False)
            self.WL[m,n] = -np.einsum("klnm->mnlk", self.WL[n,m]).conj() #Updating lower triangle
        Parallel(n_jobs=-1)(delayed(pooled)(i,m) for i,m in enumerate(range(self.Nt-n)))
    
    
    def __update_polarization_bubbles(self):
        """
        Computes polarization bubble for a given Green's function.

        Returns
        -------
        None.

        """
        self.PR = -1j*np.einsum("abln,bakm->ablkmn", self.GR, self.GL) - 1j*np.einsum("abln,bakm->ablkmn", self.GL, RtoA(self.GR))
        self.PI = -1j*np.einsum("abln,bakm->ablkmn", self.GI, ItoJ(self.GI))
        GG = self.GR - RtoA(self.GR) + self.GL
        self.PL = -1j*np.einsum("abln,bakm->ablkmn", self.GL, GG)
        self.__ER = np.einsum("male,...exab,bnxk->...mnlk", self.V, PR, self.V)
    
    
    def __update_self_energy(self):
        """
        Computes self-energy for a given screen interaction.

        Returns
        -------
        None.

        """
        self.SHF = 1j*np.einsum("sslk,mkln->smn", GL, self.V-np.swapaxes(self.V,-1,-2))
        self.SR  = 1j*np.einsum("...lk,...mkln->...mn", self.GR, self.WL) + 1j*np.einsum("...lk,...mkln->...mn", self.GL, self.WR)
        self.SI  = 1j*np.einsum("...lk,...mkln->...mn", self.GI, self.WI)
        self.SL  = 1j*np.einsum("...lk,...mkln->...mn", self.GL, self.WL)
    
    
    def solve(self):
        """
        Solves the Kadanoff-Baym equations with GW approximation for self-energy using a self-consistend method.

        Raises
        ------
        RuntimeError
            If number of iterations reaches which is specified convergence is considered unachivable.

        Returns
        -------
        None.

        """
        #Initializing HF
        self.__hartree_fock_init()
        #Initializing Matsubara branch
        self.__G_init()
        
        #Self-consisten solution of Green's function
        i = 0
        print("Initializing self-consisten solution of Green's functions")
        while True:
            time_init = time()
            #Green bootstrap
            self.__boot_iter_GR()
            self.__boot_iter_GI()
            #Green propagation
            for n in range(self.interpol.k+1,self.Nt):
                self.__step_iter_GR(n)
                self.__step_iter_GI(n)
            
            #Lesser bootsrap
            self.__boot_iter_GL()
            #Lesser propagation
            for n in range(self.interpol.k+1,self.Nt):
                self.__step_iter_GL(n)
            
            self.__update_polarization()
            
            #Screen bootstrap
            self.__boot_iter_WR()
            self.__boot_iter_WI()
            #Screen propagation
            for n in range(self.interpol.k+1,self.Nt):
                self.__step_iter_WR(n)
                self.__step_iter_WI(n)
            
            #Screen-lesser bootstrap
            self.__boot_iter_WL()
            #Screen-lesser propagation
            for n in range(self.interpol.k+1,self.Nt):
                self.__step_iter_WL(n)
            
            #Checking for convergence
            oldSHF, oldSR, oldSI, oldSL = np.copy(self.SHF), np.copy(self.SR), np.copy(self.SI), np.copy(self.SR)
            self.__update_self_energy()
            dHF = np.sum(abs(self.SHF-oldSHF))
            dR  = np.sum(abs(self.SR-oldSR))
            dI  = np.sum(abs(self.SI-oldSI))
            dL  = np.sum(abs(self.SL-oldSL))
            
            i += 1
            time_final = time()
            time_iter = time_final - time_init()
            print("Iteration %i completed in %imin %.2fs" % (i, time_iter//60, time_iter%60))
            if np.all([dHF,dR,dI,dL])<=self.convergence_limit:
                break
            if count==self.max_iter:
                raise RuntimeError("Convergence not get at required iterations")
        
        print("Recomputing Green's functions after convergence")
        #Green bootstrap
        self.__boot_iter_GR()
        self.__boot_iter_GI()
        #Green propagation
        for n in range(self.interpol.k+1,self.Nt):
            self.__step_iter_GR(n)
            self.__step_iter_GI(n)
        #Screen-lesser bootstrap
        self.__boot_iter_WL()
        #Screen-lesser propagation
        for n in range(self.interpol.k+1,self.Nt):
            self.__step_iter_WL(n)
        
        print("Green's function computed succefully")