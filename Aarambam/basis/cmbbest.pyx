import numpy as np
import os, sys
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from scipy.special import legendre, eval_legendre
from scipy.interpolate import RegularGridInterpolator
from numpy.random import default_rng
from scipy.sparse.linalg import cg as conjugate_gradient
from datetime import datetime as dt
from tqdm import tqdm
import itertools
import joblib, threadpoolctl

CMB_T0 = 2.72548
PLANCK_F_SKY_T = 0.77941
BASE_A_S = 2.100549E-9
BASE_N_SCALAR = 0.9649
BASE_K_PIVOT = 0.05
BASE_K_MIN = 2.08759e-4
BASE_K_MAX = 2.08759e-1

# For numpy PyArray_* API in Cython
np.import_array()

# A trick to find out which integer dtype to use
def GET_SIGNED_NUMPY_INT_TYPE():
    cdef int tmp
    return np.asarray(<int[:1]>(&tmp)).dtype
# If the below fails, there might be problems using int arrays in Cython
assert GET_SIGNED_NUMPY_INT_TYPE() == np.dtype("i")


cdef extern from "tetrapyd.h":
    void monte_carlo_tetrapyd_weights(double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    double *k_grid, double *k_weights, int k_npts, int n_samples)

    void compute_mode_bispectra_covariance(double *bispectra_covariance, 
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts)

    void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts)
    

def get_cond(i, OMP_msk, QQ_tilde):
    m = OMP_msk.copy()
    m[i] = True
    return np.linalg.cond(QQ_tilde[np.ix_(m, m)])

class Basis:
    ''' Class for CMBBEST's basis set'''

    def __init__(self, k_min, k_max, parameter_n_scalar, mode_p_max, k_grid_size):
        
        self.mode_p_max = mode_p_max
        self.parameter_n_scalar = parameter_n_scalar
        
        self.mode_k_min = k_min
        self.mode_k_max = k_max
        
        self.mode_functions = self.legendre_basis()

        self.k_grid_size = k_grid_size
        self.k_grid, self.k_weights = self.create_k_grid(grid_type = "uniform")

        start = dt.now()
        self.tetrapyd_indices, self.tetrapyd_grid = self.create_tetrapyd_grid()
        self.tetrapyd_grid_size = self.tetrapyd_grid.shape[1]
        self.tetrapyd_grid_weights = np.prod(self.k_weights[self.tetrapyd_indices], axis = 0)
        
        sym_factors = np.zeros_like(self.tetrapyd_grid_weights)
        p1, p2, p3  = self.tetrapyd_indices
        sym_factors[(p1 != p2) & (p2 != p3)] = 6
        sym_factors[(p1 == p2) & (p2 != p3)] = 3
        sym_factors[(p1 != p2) & (p2 == p3)] = 3
        sym_factors[(p1 == p2) & (p2 == p3)] = 1
        
        self.tetrapyd_grid_weights = self.tetrapyd_grid_weights / np.sum(self.tetrapyd_grid_weights)
        
        print("Tetrapyd weights computed")
        print(f"(Finished in {(dt.now() - start).total_seconds()}s)\n")
        
        # Evaluate mode functions on 1D k grid
        self.mode_function_evaluations = self.mode_functions(self.k_grid)
        print("1D mode functions evaluated")

        # 3D mode indices
        self.mode_indices, self.mode_symmetry_factor = self.create_mode_indices()
        self.mode_bispectra_evaluations = None

        start = dt.now()
        self.mode_bispectra_covariance = self.compute_mode_bispectra_covariance_C()
        self.mode_bispectra_norms = np.sqrt(np.diag(self.mode_bispectra_covariance))

        print("Basis is now ready")
        print(f"(Finished {(dt.now() - start).total_seconds()}s)\n")

    def legendre_basis(self):
        # Legendre polynomials + one 1/k mode which is orthogonalised
        # Note that the coefficients of orthogonalisation is fixed
        # for each basis set, regardless of the grid specifications

        n_s = self.parameter_n_scalar
        k_max = self.mode_k_max
        k_min = self.mode_k_min
        p_max = self.mode_p_max
        
        scale = (4 - n_s)/3
        
        # print("DEBUG: NORMALIZING SO P(k_mid) = 1 IN ALL CASES.")

        def basis_function(k):
            # Rescale k to lie in [-1, 1]
            fact = 2 / ( np.power(k_max, scale) -  np.power(k_min, scale) )
            k_bar = -1 + fact * ( np.power(k, scale) -  np.power(k_min, scale) )
            k_mid = np.sqrt(k_max * k_min)
            k_mid_b  = -1 + fact * ( np.power(k_mid, scale) -  np.power(k_min, scale) )
            mode_evals = np.zeros((p_max, len(k)))


            # Modes number 0-3 are monomials, and 3 to p_max are Legendre polynomials
            for p in range(0, p_max):
            
                if p <= 3:
                    mode_evals[p,:]  = np.power(k, 2 + (4 - n_s)/3 * (p - 3))

                else:
                    poly = legendre(p-1)

                    #This is kbar with k = 0, so it's a "zeropoint" (zp) value
                    kbar_zp  = -1 - 2*np.power(k_min, scale)/(np.power(k_max, scale) -  np.power(k_min, scale))
                    coef     = eval_legendre(p-1, kbar_zp)                    

                    mode_evals[p,:]  = poly(k_bar) - coef
                    
            return mode_evals

        return basis_function
    

    def create_k_grid(self, grid_type="uniform"):
        # Creates a 1D grid of k's from mode_k_min to mode_k_max

        if grid_type == "uniform":
            # Create a uniform one-dimensional k grid 
            k_grid    = np.linspace(self.mode_k_min, self.mode_k_max, self.k_grid_size)
            k_weights = np.ones_like(k_grid) * (self.mode_k_max - self.mode_k_min) / (self.k_grid_size - 1)
            
            #Downweight larger k a bit to allow more minor errors to occur there
            #in favor of smaller errors are larger scales. Similarly downweight 
            #k < 2*k_F to allow for more error at largest scales if needed.
            #In practice, neither of these really matter and we just keep
            #the below for reproducability's sake.
            k_median  = np.sqrt(self.mode_k_min * self.mode_k_max)
            k_weights = np.where(k_grid > k_median * 2, 1/np.sqrt(k_grid), 1/k_grid)
            k_weights[k_grid < 2 * self.mode_k_min] = 1/np.sqrt(k_grid[k_grid < 2 * self.mode_k_min])
                    
        else:
            print("Grid type {} is currently unsupported".format(grid_type))

        return k_grid, k_weights
    

    def create_tetrapyd_grid(self, include_borderline=False):
        # Creates a 3D grid of k's confined in a 'tetrapyd', satisfying
        # k_max >= k_1 >= k_2 >= k_3 >= k_min  and  k_2 + k_3 >= k_1 
        # if include_borderline is True, keep points that are outside the tetrapyd but
        # its voxel intersects the tetrapyd.

        Nk = self.k_grid_size
        k_grid = self.k_grid
        k_weights = self.k_weights

        tuples = [[i1, i2, i3] for i1 in range(Nk)
                    for i2 in range(i1+1)
                        for i3 in range(i2+1)]
        i1, i2, i3 = np.array(tuples).T
        
        in_tetrapyd = (k_grid[i2] + k_grid[i3] >= k_grid[i1])

        if include_borderline:
            # 1D bounds for k grid points
            interval_bounds = self.mode_k_min
            interval_bounds[0] =  k_grid[0]
            interval_bounds[1:-1] = (k_grid[:-1] + k_grid[1:]) / 2
            interval_bounds[-1] = self.mode_k_max

            # Corners specifying the grid volume (voxel)
            lb1, ub1 = interval_bounds[i1], interval_bounds[i1+1]   # upper and lower bounds of k1
            lb2, ub2 = interval_bounds[i2], interval_bounds[i2+1]   # upper and lower bounds of k2
            lb3, ub3 = interval_bounds[i3], interval_bounds[i3+1]   # upper and lower bounds of k3
            borderline = (((lb1 + lb2 < ub3) & (ub1 + ub2 > lb3))      # The plane k1+k2-k3=0 intersects
                        | ((lb2 + lb3 < ub1) & (ub2 + ub3 > lb1))   # The plane -k1+k2+k3=0 intersects
                        | ((lb3 + lb1 < ub2) & (ub3 + ub1 > lb2)))  # The plane k1-k2+k3=0 intersects
            keep = (in_tetrapyd | borderline)

        else:
            keep = in_tetrapyd

        i1, i2, i3 = i1[keep], i2[keep], i3[keep]

        tetrapyd_indices = np.stack([i1, i2, i3], axis=0)
        k1 = k_grid[i1]
        k2 = k_grid[i2]
        k3 = k_grid[i3]
        tetrapyd_grid = np.stack([k1, k2, k3], axis=0)

        return tetrapyd_indices, tetrapyd_grid


    def create_mode_indices(self, p_max=None):
        # Creates 3D indices for mode bispectra
        # If p_max=None, use self.mode_p_max

        # Vectorise the 1D mode indices to 3D
        if p_max is None:
            p_max = self.mode_p_max
        p_inds = np.arange(p_max, dtype=np.dtype("i"))
        p1, p2, p3 = np.meshgrid(p_inds, p_inds, p_inds, indexing="ij")
        ordered = (p1 >= p2) & (p2 >= p3)
        p1, p2, p3 = p1[ordered], p2[ordered], p3[ordered]
        mode_indices = np.stack([p1, p2, p3], axis=0)

        # Compute the symmetry factor: 6 if distinct, 3 if two are identical, 1 otherwise
        mode_symmetry_factor = np.ones(mode_indices.shape[1])
        mode_symmetry_factor[(p1 != p2) & (p2 != p3)] = 6
        mode_symmetry_factor[(p1 == p2) & (p2 != p3)] = 3
        mode_symmetry_factor[(p1 != p2) & (p2 == p3)] = 3

        return mode_indices, mode_symmetry_factor

    def evaluate_mode_bispectra(self):
        # Evaluate mode bispectra ('Q') on a 3D tetrapyd grid

        p1, p2, p3 = self.mode_indices
        
        # Compute the 3D mode function
        func_evals  = self.mode_function_evaluations
        t1, t2, t3 = self.tetrapyd_indices
        func_evals_1 = func_evals[p1,:]
        func_evals_2 = func_evals[p2,:]
        func_evals_3 = func_evals[p3,:]
        # Symmetrising over mode indices is equivalent to symetrising over tetrapyd indices
        bisp_sum = (func_evals_1[:,t1] * func_evals_2[:,t2] * func_evals_3[:,t3]
                    + func_evals_1[:,t1] * func_evals_2[:,t3] * func_evals_3[:,t2]
                    + func_evals_1[:,t2] * func_evals_2[:,t1] * func_evals_3[:,t3]
                    + func_evals_1[:,t2] * func_evals_2[:,t3] * func_evals_3[:,t1]
                    + func_evals_1[:,t3] * func_evals_2[:,t1] * func_evals_3[:,t2]
                    + func_evals_1[:,t3] * func_evals_2[:,t2] * func_evals_3[:,t1])

        mode_bispectra_evaluations = bisp_sum / 6.0

        return mode_bispectra_evaluations
    
    def evaluate_modal_basis_on_grid(self, k1, k2, k3):
        # Returns a (n_modes) X (shape of k1) sized array contianing
        # 3D mode functions evaluations on a given k1, k2, k3 grid

        p1, p2, p3 = self.mode_indices
        evals_k1 = self.mode_functions(k1)
        evals_k2 = self.mode_functions(k2)
        evals_k3 = self.mode_functions(k3)

        evals = np.zeros((len(p1), *k1.shape))
        for pp1, pp2, pp3 in itertools.permutations([p1, p2, p3]):
            evals = evals + (evals_k1[pp1,:] * evals_k2[pp2,:] * evals_k3[pp3,:] / 6)
        
        return evals


    def evaluate_modal_bispectra_on_grid(self, coeffs, k1, k2, k3):
        # Returns a (n_coeffs) X (shape of k1) sized array contianing
        # 3D modal bispectrum evaluations on a given k1, k2, k3 grid

        basis_evals = self.evaluate_modal_basis_on_grid(k1, k2, k3)
        bisp_evals = np.matmul(coeffs, basis_evals)

        return bisp_evals


    def compute_mode_bispectra_covariance_C(self):
        # Evaluate the covariance matrix ('QQ') between mode bispectra
        # (QQ)_{mn} := <Q_m, Q_n>

        i1, i2, i3 = self.tetrapyd_indices
        p1, p2, p3 = self.mode_indices

        cdef double [::1] tetra_weights = self.tetrapyd_grid_weights
        cdef int [::1] tetra_i1 = i1.astype("i")
        cdef int [::1] tetra_i2 = i2.astype("i")
        cdef int [::1] tetra_i3 = i3.astype("i")
        cdef int tetra_npts = self.tetrapyd_grid_size

        cdef int [::1] mode_p1 = p1.astype("i")
        cdef int [::1] mode_p2 = p2.astype("i")
        cdef int [::1] mode_p3 = p3.astype("i")
        cdef int n_modes = self.mode_indices.shape[1]

        cdef double [:,::1] mode_evals = self.mode_function_evaluations
        cdef int mode_p_max = self.mode_p_max
        cdef int k_npts = self.k_grid_size

        mode_bispectra_covariance = np.zeros((n_modes, n_modes), dtype=np.dtype("d"))
        cdef double [:,::1] mode_bispectra_covariance_view = mode_bispectra_covariance

        # Call the wrapped C function
        compute_mode_bispectra_covariance(&mode_bispectra_covariance_view[0,0],
                        &tetra_weights[0], <int *> &tetra_i1[0], <int *> &tetra_i2[0], <int *> &tetra_i3[0], tetra_npts,
                        <int *> &mode_p1[0], <int *> &mode_p2[0], <int *> &mode_p3[0], n_modes,
                        &mode_evals[0,0], mode_p_max, k_npts)

        return mode_bispectra_covariance 
    

    def basis_expansion(self, model_list, regularization = 0, Niter = 1e4, 
                        rtol = 1e-12, bad_modes = False, RR_tol = 1e-2, cond_tol = 1e9,
                        max_modes = None):
        # Expand given model shape functions with respect to separable basis

        N_models = len(model_list)
        N_modes = self.mode_indices.shape[1]
        Q = self.evaluate_mode_bispectra()      # (N_modes, N_tetrapyd_points)
        QQ = self.mode_bispectra_covariance     # (N_modes, N_modes)
        norms = self.mode_bispectra_norms       # (N_modes)
        sym_fact = self.mode_symmetry_factor    # (N_modes)
        p1, p2, p3 = self.mode_indices          # (N_modes)
        w = self.tetrapyd_grid_weights          # (N_tetrapyd_points)
        k1, k2, k3 = self.tetrapyd_grid         # (N_tetrapyd_points)
        p_max = self.mode_p_max

        if max_modes is None: 
            print(f"NO max_modes SET. DEFAULTING TO {p1.size}. LOWER THIS VALUE TO SPEED UP DECOMPOSITION.")
            max_modes = p1.size

        # Evaluate given shape functions and their covariance on a tetrapyd
        # 'S' is a matrix of size (N_models, N_tetrapyd_points)
        # S is assumed to be symmetric under permutations of k1, k2, k3
        S = np.stack([model.shape_function(k1, k2 ,k3) for model in model_list])
        shape_covariance = np.matmul(S * w[np.newaxis,:], S.T)

        # Find the inner product between shapes and mode bispectra
        # 'QS' is a matrix of size (N_models, N_modes)

        if Q is None:
            QS = self.compute_QS_C(S)      # Doesn't require having computed Q
        else:
            QS = np.matmul(S * w[np.newaxis,:], Q.T)       # Requires having computed Q
            
            
        ##########################################################################
        #DHAYAA: Hack QS and QQ to get rid of correlations and zero out some modes
        QQ_temp = QQ * 1
        QS_temp = QS * 1
        Q_temp  = Q  * 1
        
        
        #A few terms cause unavoidable divergences in P(k) so drop it from basis        
        #First  is k1^-3 k2^-3 k3^-3
        #Second is k1^-3 k2^-3 k3^-2
        #Third  is k1^-3 k2^-3 k3^-1
        #Fourth is k1^-3 k2^-2 k3^-3

        badmask = ( ((p1 <= 2) & (p2 == 0) & (p3 == 0)) |
                    ((p1 == 1) & (p2 == 1) & (p3 == 0)) 
                )
        
        
        badmask |= bad_modes        
        badinds = np.where(badmask)[0]
        print("Zeroing out mode inds", badinds, "\n")

        #Try to get a SLURM env variable, else default to os.cpu_count()
        cpus_allocated = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
        os.environ["MKL_NUM_THREADS"] = str(cpus_allocated)
        os.environ["OMP_NUM_THREADS"] = str(cpus_allocated)

        for bi in badinds:
            QQ_temp[bi, :] = 0 #Lie to algorithm and say this mode is uncorrelated with everything else
            QQ_temp[:, bi] = 0
            QQ_temp[bi,bi] = QQ[bi,bi] #Otherwise matrix is not invertible

            QS_temp[:, bi] = 0 #Say there is no correlation between mode and the shape function

            Q_temp[bi] = 0 #Also do the same of the basis function array (not matrix) to be consistent
        
        ##########################################################################

        QQ_tilde = QQ_temp / norms[:,np.newaxis] / norms[np.newaxis,:]     # Normalise modes
        Q_tilde  = Q_temp / norms[:, np.newaxis]
        alpha    = np.zeros((N_models, N_modes))

        #Loop over models
        for model_no in range(N_models):
            QS_tilde = QS_temp[model_no,:] / norms   # Normalise mode
            
            OMP_msk = np.zeros_like(p1, dtype = bool)
            OMP_msk[(p1 <= 3) & (p2 <= 3) & (p3 <= 3)] = True

            #If max_modes < 0 then we just use all modes
            if max_modes == -1:
                OMP_msk = np.ones_like(OMP_msk, dtype = bool)

            delta_RR = np.inf
            iterator = 0
            pbar     = tqdm(desc = 'Building targeted Mode basis')

            while True:
                
                #If we pass in fewer than 4 modes, those are already selected above
                #And so this Orthogonal Matching Pursuit step has no point to it
                if len(norms) <= np.sum(OMP_msk): break

                alpha_tmp, exit_code = conjugate_gradient(QQ_tilde[np.ix_(OMP_msk, OMP_msk)] + np.eye(np.sum(OMP_msk)) * regularization, 
                                                          QS_tilde[OMP_msk], rtol = rtol, atol=0, maxiter = int(Niter))
                
                #Compute the residual, and the correlation of all modes with residuals
                R  = S - np.sum(Q_tilde[OMP_msk] * alpha_tmp[:, None], axis = 0)
                RR = np.matmul(R * w[np.newaxis,:], R.T)[0, 0]
                delta_RR = RR/shape_covariance

                #An estimate of how correlated a given function is with all other functions in the basis set
                #Don't have to account for off-diagonal because it is implicitly removed here.
                #Divide by number of modes you compared again. So if this 1 it is exactly correlated, and if its
                #0 it is not correlated at all.
                prho = np.sqrt(np.sum(QQ_tilde[:, OMP_msk]**2, axis = 1)[np.invert(OMP_msk)]) / np.sum(OMP_msk)

                if np.abs(delta_RR[0, 0]) <= RR_tol: 
                    print(f"\n HIT LIMIT OF VAR_RES/VAR_TOT = {RR_tol:0.4e}. ENDING SEARCH", flush = True)
                    break

                if np.sum(OMP_msk) == max_modes - 1: 
                    print("\n USED ALL AVAILABLE MODES", flush = True)
                    break

                cond_here = get_cond(np.where(OMP_msk)[0][0], OMP_msk, QQ_tilde) #Use existing True ind just so we can get cond of full matrix
                with threadpoolctl.threadpool_limits(limits = 1):
                    cond  = joblib.Parallel(n_jobs = os.cpu_count())(joblib.delayed(get_cond)(i, OMP_msk, QQ_tilde) 
                                                                     for i in np.where(np.invert(OMP_msk))[0])    
                fcond     = cond/cond_here - 1

                if cond_here > cond_tol: 
                    print(f"\n HIT LIMIT OF COND NUM {cond_here:0.4e} > {cond_tol:0.4e}. ENDING SEARCH", flush = True)
                    break

                QR = self.compute_QS_C(R)[0] / norms / np.sqrt(RR) #Renormalize like QQ_temp
                QR[badinds] = 0 #Null out inds we asked to remove
                QR = QR[np.invert(OMP_msk)] #Keep only modes we haven't used yet
                
                m  = np.argsort(np.abs(QR))[-10:] #Grab the ten most important indices
                m  = m[np.argmin(fcond[m])] #Find the one wih the smallest condition num increase
                m  = np.where(np.invert(OMP_msk))[0][m]

                OMP_msk[m] = True #Add mode to the basis set

                iterator += 1

                pbar.update(1)
                pbar.set_postfix({'loss' : f"{delta_RR[0, 0]:.5f}", 
                                  'norm' : f"{np.sum(alpha_tmp**2):0.6e}", 
                                  'cond' : f"{cond_here:0.6e}"})

            pbar.close()

            print("\n USING MODES", np.where(OMP_msk)[0])

            alpha_tmp, exit_code = conjugate_gradient(QQ_tilde[np.ix_(OMP_msk, OMP_msk)] + np.eye(np.sum(OMP_msk)) * regularization, 
                                                      QS_tilde[OMP_msk], rtol = 1e-16, atol=0, maxiter = int(1e6))
            
            alpha_tilde = np.zeros_like(norms)
            alpha_tilde[OMP_msk] = alpha_tmp
            print("\n Shape expanded using CG")
            alpha[model_no,:] = alpha_tilde / norms          # Reintroduce normalisation factor

        expansion_coefficients = alpha
        
        # Optional check on convergence of the mode expansion
        sum_SS = np.diag(shape_covariance)                      # <S, S>
        sum_SR = np.sum(alpha * QS, axis=1)                     # <S, S_rec>    
        sum_RR = np.sum(alpha * np.matmul(alpha, QQ), axis=1)   # <S_rec, S_rec>    

        convergence_correlation = sum_SR / np.sqrt(sum_SS * sum_RR)
        convergence_correlation = 1.0 - np.abs(1 - convergence_correlation)     # For when corr > 1 due to numerical errors
        convergence_MSE = np.abs(sum_SS + sum_RR - 2 * sum_SR) / sum_SS

        return expansion_coefficients, shape_covariance, convergence_correlation, convergence_MSE


    def compute_QS_C(self, S):
        # Evaluate the inner product('QS') between mode bispectra and shape functions
        # (QS)_{in} := <S_i, Q_n}

        i1, i2, i3 = self.tetrapyd_indices
        p1, p2, p3 = self.mode_indices

        cdef int mode_p_max = self.mode_p_max
        cdef int k_npts = self.k_grid_size

        cdef double [::1] tetra_weights = self.tetrapyd_grid_weights
        cdef int [::1] tetra_i1 = i1.astype("i")
        cdef int [::1] tetra_i2 = i2.astype("i")
        cdef int [::1] tetra_i3 = i3.astype("i")

        cdef int tetra_npts = self.tetrapyd_grid_size

        cdef int [::1] mode_p1 = p1.astype("i")
        cdef int [::1] mode_p2 = p2.astype("i")
        cdef int [::1] mode_p3 = p3.astype("i")
        cdef int n_modes = self.mode_indices.shape[1]

        cdef double [::1] S_view = S.flatten()
        cdef int n_shapes = S.shape[0]

        cdef double [:,::1] mode_evals = self.mode_function_evaluations

        QS = np.zeros((n_shapes, n_modes), dtype=np.dtype("d"))
        QS = QS.flatten()
        cdef double [::1] QS_view = QS
        S = S.flatten()

        # Call the wrapped C function

        compute_QS(&QS_view[0], &S_view[0], n_shapes,
                    &tetra_weights[0], <int *> &tetra_i1[0], <int *> &tetra_i2[0], <int *> &tetra_i3[0], tetra_npts,
                    <int *> &mode_p1[0], <int *> &mode_p2[0], <int *> &mode_p3[0], n_modes,
                    &mode_evals[0,0], mode_p_max, k_npts)

        QS = QS.reshape((n_shapes, n_modes))

        return QS


class Model:
    ''' Class for the bispectrum template or model of interest '''

    def __init__(self, shape_type, **kwargs):
        self.shape_type = shape_type.lower()

        if shape_type == "custom_shape_evals":
            # Custom shape function specified by the k grid and evalutations
            self.grid_k_1 = kwargs["grid_k_1"]
            self.grid_k_2 = kwargs["grid_k_2"]
            self.grid_k_3 = kwargs["grid_k_3"]
            self.shape_name = kwargs.get("shape_name", "custom")
            self.shape_function_values = kwargs["shape_function_values"]
            self.shape_function = self.custom_shape_function_from_evals()
        
        elif shape_type == "custom":
            # Custom shape function specified by the given function
            # Takes 'shape_function' defined as
            # S(k_1, k_2, k_3) := (k_1 k_2 k_3)^2 B_\Phi (k_1, k_2, k_3),
            # where <\Phi \Phi \Phi> = B_\Phi(k_1, k_2, k_3) (2\pi)^3 \delta^{(3)}(\mathbf{K})
            self.shape_name = kwargs.get("shape_name", "custom")
            self.shape_function = kwargs["shape_function"]
        
        else:
            # Preset shapes
            preset_shapes_list = ["local", "equilateral", "orthogonal"]

            if shape_type == "local":
                self.shape_name = kwargs.get("shape_name", "local")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)            # k value where P(k) = A_s

                self.shape_function = self.local_shape_function()

            elif shape_type == "equilateral" or shape_type == "equil":
                self.shape_name = kwargs.get("shape_name", "equilateral")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)    # k value where P(k) = A_s

                self.shape_function = self.equilateral_shape_function()
                
            elif shape_type == "orthogonal" or shape_type == "ortho":
                self.shape_name = kwargs.get("shape_name", "orthogonal")
                self.parameter_A_scalar = kwargs.get("parameter_A_scalar", BASE_A_S)   # Scalar power spectrum amplitude
                self.parameter_n_scalar = kwargs.get("parameter_n_scalar", BASE_N_SCALAR)    # Scalar spectral index
                self.parameter_k_pivot = kwargs.get("parameter_k_pivot", 0.05)    # k value where P(k) = A_s

                self.shape_function = self.orthogonal_shape_function()

            else:
                print("Shape type preset '{}' is currently not supported".format(shape_type)) 
                print("Supported shapes:", str(preset_shapes_list))
                return
        

    def custom_shape_function_from_evals(self):
        # Performs a 3D linear interporlation to evaluate the shape function

        interp = RegularGridInterpolator((self.grid_k_1, self.grid_k_2, self.grid_k_3), self.shape_function_values)

        def shape_function(k_1, k_2, k_3):
            ks = np.column_stack([k_1, k_2, k_3])
            return interp(ks)
        
        return shape_function


    def local_shape_function(self):
        # Local template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 2 * (delta_phi ** 2)

            S_1 = k_1 * k_1 * np.power(k_2 * k_3, n_s - 2)
            S_2 = k_2 * k_2 * np.power(k_3 * k_1, n_s - 2)
            S_3 = k_3 * k_3 * np.power(k_1 * k_2, n_s - 2)

            S = pref * (S_1 + S_2 + S_3)
            return S
        
        return shape_function


    def equilateral_shape_function(self):
        # Equilateral template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 6 * (delta_phi ** 2)

            # Precompute different powers of k's: square, linear, constant, inverse
            k_1_sq = k_1 * k_1   
            k_1_li = np.power(k_1, (n_s + 2) / 3)
            k_1_co = np.power(k_1, 2 * (n_s - 1) / 3)
            k_1_in = np.power(k_1, n_s - 2)

            k_2_sq = k_2 * k_2
            k_2_li = np.power(k_2, (n_s + 2) / 3)
            k_2_co = np.power(k_2, 2 * (n_s - 1) / 3)
            k_2_in = np.power(k_2, n_s - 2)

            k_3_sq = k_3 * k_3
            k_3_li = np.power(k_3, (n_s + 2) / 3)
            k_3_co = np.power(k_3, 2 * (n_s - 1) / 3)
            k_3_in = np.power(k_3, n_s - 2)

            S_1 = (k_1_sq * k_2_in * k_3_in
                    + k_2_sq * k_3_in * k_1_in
                    + k_3_sq * k_1_in * k_2_in)

            S_2 = k_1_co * k_2_co * k_3_co

            S_3 = (k_1_in * k_2_co * k_3_li 
                    + k_1_in * k_3_co * k_2_li
                    + k_2_in * k_1_co * k_3_li
                    + k_2_in * k_3_co * k_1_li
                    + k_3_in * k_1_co * k_2_li
                    + k_3_in * k_2_co * k_1_li)

            S = pref * ((-1) * S_1 + (-2) * S_2 + 1 * S_3)
            return S
        
        return shape_function


    def orthogonal_shape_function(self):
        # Orthogonal template with scale dependence given by n_scalar

        A_s = self.parameter_A_scalar
        n_s = self.parameter_n_scalar
        k_pivot = self.parameter_k_pivot
        delta_phi = 2 * (np.pi ** 2) * ((3 / 5) ** 2) * (k_pivot ** (1 - n_s)) * A_s

        def shape_function(k_1, k_2, k_3):

            pref = 6 * (delta_phi ** 2)

            # Precompute different powers of k's: square, linear, constant, inverse
            k_1_sq = k_1 * k_1   
            k_1_li = np.power(k_1, (n_s + 2) / 3)
            k_1_co = np.power(k_1, 2 * (n_s - 1) / 3)
            k_1_in = np.power(k_1, n_s - 2)

            k_2_sq = k_2 * k_2
            k_2_li = np.power(k_2, (n_s + 2) / 3)
            k_2_co = np.power(k_2, 2 * (n_s - 1) / 3)
            k_2_in = np.power(k_2, n_s - 2)

            k_3_sq = k_3 * k_3
            k_3_li = np.power(k_3, (n_s + 2) / 3)
            k_3_co = np.power(k_3, 2 * (n_s - 1) / 3)
            k_3_in = np.power(k_3, n_s - 2)

            S_1 = (k_1_sq * k_2_in * k_3_in
                    + k_2_sq * k_3_in * k_1_in
                    + k_3_sq * k_1_in * k_2_in)

            S_2 = k_1_co * k_2_co * k_3_co

            S_3 = (k_1_in * k_2_co * k_3_li 
                    + k_1_in * k_3_co * k_2_li
                    + k_2_in * k_1_co * k_3_li
                    + k_2_in * k_3_co * k_1_li
                    + k_3_in * k_1_co * k_2_li
                    + k_3_in * k_2_co * k_1_li)

            S = pref * ((-3) * S_1 + (-8) * S_2 + 3 * S_3)
            return S
        
        return shape_function