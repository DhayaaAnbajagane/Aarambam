import os, sys, numpy as np, itertools
from scipy import special
from . import cmbbest as best

class BasicBasisDecompose:
    """
    Perform a modal decomposition of a theoretical bispectrum into a finite
    basis of separable templates.

    This class wraps the functionality of our modified CMB-BEST code
    to expand a given bispectrum shape function into
    a modal basis. It provides convenience methods to compute the modal
    coefficients (`alpha`), auxiliary mode-normalization tables (`Ai`),
    and to save these results to disk.

    The actual bispectrum definition must be supplied by overriding the 
    :math:`raw_bispectrum` method. Please use this class with
    ``utils.Decomposer`` for actual analyses.

    We use an orthogonal matching pursuit (OMP) approach, where the
    basis is iteratively expanded. We recompute the decomposition at each
    iteration.

    Parameters
    ----------
    N_modes : int
        The highest order of the basis modes to use in the decomposition.
    n_s : float
        Spectral tilt parameter of the primordial power spectrum.
    Lbox : float
        Size of the simulation box (used to set fundamental wavenumber).
    Nmax : int
        Maximum k-grid index to consider (sets k_max = k_F * Nmax).
        Important as it defines the wavenumber normalization for input
        into the Legendre polynomials
    ModeTol : float
        (relative) accuracy up to which the code continues doing the mode expansion.
        It is fine if this condition is not met. Almost always, `MaxModeCount` is met
        prior to this condition being met.
    MaxModeCount : int
        Maximum number of modes to keep in the expansion.
    """

    def __init__(self, N_modes, n_s, Lbox, Nmax, ModeTol, MaxModeCount, **kwargs):
        
        args = dict(N_modes = N_modes, n_s = n_s, Lbox = Lbox, Nmax = Nmax, 
                    ModeTol = ModeTol, MaxModeCount = MaxModeCount)
        args.update(**kwargs)

        self.args = args

    def go(self):
        """
        Run the full decomposition workflow.

        Executes both :math:`generate_alpha` and :math:`generate_Ai` and
        combines their outputs.

        Returns
        -------
        dict
            A dictionary containing:
            - `S_original` : ndarray
              Original bispectrum evaluated on sample configurations.
            - `S_apprx` : ndarray
              Approximate bispectrum reconstructed from the modal expansion.
            - `Templates` : ndarray
              Modal basis templates evaluated on sample configurations.
            - `kbin` : ndarray
              k-bin array used for evaluation.
            - `AlphaTable` : ndarray
              Table of mode indices and corresponding alpha coefficients.
            - `S_apprx_individual` : ndarray
              Contribution of each mode separately to the approximation.
            - `AiTable` : ndarray
              Normalization table for the basis modes.
        """
        out = {}
        out.update(self.generate_alpha())
        out.update(self.generate_Ai())

        return out    

    def generate_alpha(self, verbose = True):
        """
        Compute the modal coefficients (alpha) for the bispectrum expansion.

        Builds the basis, performs the expansion against the user-supplied
        bispectrum shape, and reports convergence information. Also
        evaluates the approximation against the original bispectrum on
        specific limits of the bispectrum.

        Parameters
        ----------
        verbose : bool, optional
            If True, print convergence diagnostics to stdout.

        Returns
        -------
        dict
            Dictionary containing:
            - ``S_original`` : ndarray
              Original bispectrum shape values.
            - ``S_apprx`` : ndarray
              Approximate bispectrum from modal expansion.
            - ``Templates`` : ndarray
              Modal basis templates on sample k configurations.
            - ``kbin`` : ndarray
              Geometric k-bin array used for evaluation.
            - ``AlphaTable`` : ndarray
              Mode index table and alpha coefficients.
            - ``S_apprx_individual`` : ndarray
              Individual mode contributions to the bispectrum.
        """
        basis = best.Basis(mode_p_max = self.args['N_modes'],
                           parameter_n_scalar = self.args['n_s'], 
                           k_grid_size = 150,
                           k_min = 2 * np.pi / self.args['Lbox'],
                           k_max = 2 * np.pi / self.args['Lbox'] * self.args['Nmax'])
        
        p1, p2, p3 = basis.mode_indices
        
        #For the lower orders, just keep the main necessary ones and skip the others. 
        #This works in all cases, so we just do this for simplicity.
        msk   = ((p1 < 4) & 
                  np.invert(((p1 == 1) & (p2 == 1) & (p3 == 1)) | 
                            ((p1 == 2) & (p2 == 1) & (p3 == 0)) | 
                            ((p1 == 3) & (p2 == 0) & (p3 == 0)))
            )
        
        model = [best.Model(shape_type = "custom", shape_name = "TMP", shape_function = self.shape_function)]
        
        # alpha, shape_cov, converg, converg_MSE = basis.basis_expansion(model, rtol = 1e-16, Niter = 1e5, bad_modes = msk)
        res   = basis.basis_expansion(model, rtol = 1e-10, Niter = 1e3, bad_modes = msk, RR_tol = self.args['ModeTol'],
                                      max_modes = self.args['MaxModeCount'])
        alpha, shape_cov, converg, converg_MSE = res
        eps   = np.sqrt(2 * (1 - converg[0]**2))

        if verbose:
            print("\n-------------- BASIS DECOMPOSITION --------------")
            print("correlation : %0.10f (std = %0.8e, eps = %0.8e)" % (converg[0], np.sqrt(converg_MSE[0]), eps) ) 
            print("-------------------------------------------------\n")

        #Evaluate the original shape function
        kF   = 2 * np.pi / self.args['Lbox']
        kbin = np.geomspace(kF, kF * self.args['Nmax'] / 2, 100)

        Orig = np.vstack([self.shape_function(kbin, kbin, kF * np.ones_like(kbin)),
                          self.shape_function(kbin, kbin, kbin),
                          self.shape_function(kbin, kbin, 2*kbin)]).T
        Aprx = np.vstack([basis.evaluate_modal_bispectra_on_grid(alpha, kbin, kbin, kF * np.ones_like(kbin)),
                          basis.evaluate_modal_bispectra_on_grid(alpha, kbin, kbin, kbin),
                          basis.evaluate_modal_bispectra_on_grid(alpha, kbin, kbin, 2*kbin)]).T

        Aprx_split = np.array([alpha[0][:, None] * basis.evaluate_modal_basis_on_grid(kbin, kbin, kF * np.ones_like(kbin)),
                               alpha[0][:, None] * basis.evaluate_modal_basis_on_grid(kbin, kbin, kbin),
                               alpha[0][:, None] * basis.evaluate_modal_basis_on_grid(kbin, kbin, 2*kbin)])
        
        Templates  = np.array([basis.evaluate_modal_basis_on_grid(kbin, kbin, kF * np.ones_like(kbin)),
                               basis.evaluate_modal_basis_on_grid(kbin, kbin, kbin),
                               basis.evaluate_modal_basis_on_grid(kbin, kbin, 2*kbin)])

        AlphaTable = np.vstack([basis.mode_indices, alpha[0]]).T

        output = {}
        output['S_original'] = Orig
        output['S_apprx']    = Aprx
        output['Templates']  = Templates 
        output['kbin']       = kbin
        output['AlphaTable'] = AlphaTable
        output['S_apprx_individual'] = Aprx_split

        return output


    def generate_Ai(self):
        """
        Compute the Ai normalization table for the modal basis.

        The Ai table provides mode-dependent normalization coefficients
        based on the Legendre polynomial construction used in CMB-BEST.

        Returns
        -------
        dict
            A dictionary with key ``AiTable`` containing an array of shape
            (N_modes, 3). Each row contains:
            - mode index
            - coefficient for Ai
            - coefficient for Aj (always zero here and never used in the actual analysis/papers)
        """
        scale = (4 - self.args['n_s'])/3
        k_min = 2 * np.pi / self.args['Lbox']
        k_max = 2 * np.pi / self.args['Lbox'] * self.args['Nmax']
        
        k_min_ns = np.power(k_min, scale)
        k_max_ns = np.power(k_max, scale)
        
        x0 = np.zeros([self.args['N_modes'], 2])
        
        # Modes number 0-3 are monomials, and 3 to p_max are Legendre polynomials
        for p in range(self.args['N_modes']):

            #Just repeats exactly what is done within CMB-BEST here
            kbar_zp  = -1 - 2*k_min_ns/(k_max_ns - k_min_ns)
            coef     = special.eval_legendre(p, kbar_zp)
            x0[p, 0] = coef

        #This is an extra column we kept in case we wanted to null
        #the x1 (and not just x0) term. We do not use it anywhere.
        #So just set it to 0.
        x0[:, 1] = 0 
            
        output = {'AiTable' : np.vstack([range(self.args['N_modes']), x0.T]).T}
        
        return output
    

    def shape_function(self, k1, k2, k3):
        """
        Define the bispectrum shape function.

        The shape function is the bispectrum multiplied by the momentum
        prefactor (k1 * k2 * k3)^2.

        Parameters
        ----------
        k1, k2, k3 : array_like
            Wavenumber magnitudes of the triangle configuration.

        Returns
        -------
        ndarray
            Shape function values for the given triangle(s).
        """
        return self.bispectrum(k1, k2, k3) * (k1 * k2 * k3)**2


    def bispectrum(self, k1, k2, k3):
        """
        Evaluate the symmetrized bispectrum.

        Calls the user-supplied :math:`raw_bispectrum` and symmetrizes it
        over all permutations of (k1, k2, k3). You can also override this
        class if you want to supply this symmetrized bispectrum yourself.

        Parameters
        ----------
        k1, k2, k3 : array_like
            Wavenumber magnitudes of the triangle configuration.

        Returns
        -------
        ndarray
            Symmetrized bispectrum values.

        Raises
        ------
        NotImplementedError
            If no ``raw_bispectrum`` is defined in the subclass.
        """
        if hasattr(self, 'raw_bispectrum'):
            
            k = np.stack([k1, k2, k3])
            B = 0

            #Permute over the possible indices
            for a, b, c in itertools.permutations(range(3)):
                B += self.raw_bispectrum(k[a], k[b], k[c])
                
            return B
        
        else:
            raise NotImplementedError("Bispectrum not implemented. Must define `raw_bispectrum` method.")
        

    def raw_bispectrum(self, k1, k2, k3):
        """
        Define the raw (unsymmetrized) bispectrum.

        This method **must** be implemented by subclasses to provide the
        underlying theoretical bispectrum model.

        Parameters
        ----------
        k1, k2, k3 : array_like
            Wavenumber magnitudes of the triangle configuration.

        Returns
        -------
        ndarray
            Bispectrum values for the given triangle(s).
        """
        raise NotImplementedError("Implement a raw_bispectrum class")
    

    def finalize(self, out, directory):
        """
        Write decomposition outputs to disk.

        Saves the alpha coefficients and Ai table as ASCII files in the
        specified directory.

        Parameters
        ----------
        out : dict
            Dictionary of outputs as returned by :meth:`go`.
        directory : str
            Path to the directory where output files should be written.

        Notes
        -----
        - Writes ``AlphaTable.dat`` with columns (p1, p2, p3, alpha).
        - Writes ``AiTable.dat`` with columns (mode_index, x0, x1).
        """
        
        #Write it all out into disk
        np.savetxt(directory + '/AlphaTable.dat', out['AlphaTable'], fmt='%d %d %d %.17g')
        np.savetxt(directory + '/AiTable.dat',    out['AiTable'],    fmt='%d %.17g %.17g')

        print(f"Tables written out to {directory}")

        