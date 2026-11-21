"""Driver that combines a bispectrum model with a decomposition basis."""
import os

__all__ = ['Decomposer']


class Decomposer:
    """Run a bispectrum template through the Legendre or the binned decomposition.

    Give exactly one of ``N_modes`` (Legendre) or ``Binned_Nk`` (binned).

    Parameters
    ----------
    n_s : float
        Scalar spectral index.
    Lbox : float
        Box side in Mpc/h; sets the smallest wavenumber ``2 pi / Lbox``.
    Nmax : int
        Largest wavenumber, in units of ``2 pi / Lbox`` (use the ``Nmax`` of the 2LPT config).
    N_modes : int, optional
        Number of 1D Legendre modes. Selects the Legendre decomposition.
    ModeTol : float, optional
        Legendre only (required): stop once the residual variance is below this fraction of the template's.
    MaxModeCount : int, optional
        Legendre only (required): largest number of 3D modes kept; usually reached first.
    Binned_Nk : int, optional
        Number of bins per k axis. Selects the binned decomposition.
    Binned_filter : str, optional
        Binned interpolation: ``'hybrid_cubic_k3cubic'`` (default), ``'hybrid_linear_k3linear'``,
        ``'hybrid_cubic_k3linear'``, ``'hybrid_linear_k3cubic'`` or ``'heaviside'``.
    **kwargs
        Template parameters and other options of the basis class, passed on to it.

    Examples
    --------
    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 256, Binned_Nk = 20)
    >>> out = dec.go(Am.models.Local, Am.binned.BasicBinnedDecompose, outdir = './tables')
    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 256, N_modes = 15, ModeTol = 1e-6, MaxModeCount = 100)
    >>> out = dec.go(Am.models.ScalarI, Am.legendre.BasicLegendreDecompose, mass = 1, outdir = './tables')
    """

    def __init__(self, n_s, Lbox, Nmax, N_modes = None, ModeTol = None, MaxModeCount = None, Binned_Nk = None, Binned_filter = 'hybrid_cubic_k3cubic', **kwargs):
        use_legendre = N_modes is not None
        use_binned   = Binned_Nk is not None

        if use_legendre == use_binned:
            raise ValueError('Supply exactly one of N_modes or Binned_Nk.')

        args = dict(n_s = n_s, Lbox = Lbox, Nmax = Nmax)

        if use_legendre:
            if ModeTol is None or MaxModeCount is None:
                raise ValueError('ModeTol and MaxModeCount are required when N_modes is supplied.')

            args.update(N_modes = N_modes, ModeTol = ModeTol, MaxModeCount = MaxModeCount)
            self.kind = 'legendre'
        else:
            if ModeTol is not None or MaxModeCount is not None:
                raise ValueError('ModeTol and MaxModeCount apply only to the Legendre decomposition.')

            args.update(Binned_Nk = Binned_Nk, Binned_filter = Binned_filter)
            self.kind = 'binned'

        args.update(**kwargs)
        self.args = args

    @staticmethod
    def _basis_kind(Basis):
        defaults = getattr(Basis, 'DEFAULTS', {})
        names    = [base.__name__.lower() for base in Basis.__mro__]

        if 'Binned_Nk' in defaults or any('binned' in name for name in names):
            return 'binned'
        return 'legendre'

    def go(self, Model, Basis, outdir = None, **kwargs):
        """Decompose one template.

        Parameters
        ----------
        Model : class
            Template class, e.g. ``Am.models.Local``.
        Basis : class
            ``Am.legendre.BasicLegendreDecompose`` or ``Am.binned.BasicBinnedDecompose``, matching the
            constructor's choice.
        outdir : str, optional
            Existing directory to write the 2LPT tables into (checked before the decomposition starts).
        **kwargs
            Template parameters (e.g. ``mass``); they override the constructor's keywords.

        Returns
        -------
        dict
            The decomposition (``Basis.go()``), including the template and its approximation in the
            squeezed, equilateral and folded limits (``S_original``, ``S_apprx``, ``kbin``).
        """
        basis_kind = self._basis_kind(Basis)
        if basis_kind != self.kind:
            selector = 'Binned_Nk' if basis_kind == 'binned' else 'N_modes'
            raise ValueError(f'{Basis.__name__} is a {basis_kind} basis; initialize Decomposer with {selector}.')

        if outdir is not None and not os.path.isdir(outdir):     #before the (possibly long) decomposition
            raise FileNotFoundError(f'Output directory {outdir} does not exist')

        inpars = self.args.copy()
        inpars.update(**kwargs)

        decomp = type('C', (Model, Basis), {})
        instan = decomp(**inpars)
        result = instan.go()

        if outdir is not None:
            instan.finalize(result, directory = outdir)

        return result
