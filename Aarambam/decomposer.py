"""Driver that combines a bispectrum model with a decomposition basis."""
import os

__all__ = ['Decomposer']


class Decomposer:
    """Unified driver for Legendre-modal and binned bispectrum decompositions.

    Exactly one decomposition selector must be supplied:

    * ``N_modes`` selects the Legendre/modal decomposition and additionally
      requires ``ModeTol`` and ``MaxModeCount``.
    * ``Binned_Nk`` selects the binned decomposition. ``Binned_filter`` chooses
      the C-level interpolation scheme and defaults to
      ``'hybrid_cubic_k3cubic'``.

    Parameters
    ----------
    n_s : float
        Scalar spectral index used by either decomposition.
    Lbox : float
        Side length of the simulation box in comoving units.
    Nmax : int
        Maximum dimensionless wavenumber index.
    N_modes : int, optional
        Number of Legendre/modal basis modes. Supplying this selects the modal
        decomposition.
    ModeTol : float, optional
        Modal truncation tolerance. Required when ``N_modes`` is supplied.
    MaxModeCount : int, optional
        Maximum number of retained modal terms. Required when ``N_modes`` is
        supplied.
    Binned_Nk : int, optional
        Number of bins on each binned-kernel axis. Supplying this selects the
        binned decomposition.
    Binned_filter : str, optional
        Binned interpolation/filter type. The supported values are those
        accepted by ``BasicBinnedDecompose``, including ``'heaviside'``,
        ``'hybrid_linear_k3linear'``, ``'hybrid_cubic_k3linear'``,
        ``'hybrid_linear_k3cubic'`` and ``'hybrid_cubic_k3cubic'``.
    **kwargs : dict, optional
        Model parameters and backend-specific settings forwarded to the
        combined model/basis class.

    Examples
    --------
    Legendre/modal decomposition:

    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 256,
    ...                  N_modes = 50, ModeTol = 1e-6, MaxModeCount = 500)
    >>> out = dec.go(Model = SomeBispectrumModel, Basis = BasicLegendreDecompose)

    Binned decomposition:

    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 256,
    ...                  Binned_Nk = 25)
    >>> out = dec.go(Model = SomeBispectrumModel, Basis = BasicBinnedDecompose,
    ...              outdir = './results')
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
        """Run the selected decomposition for a chosen model and basis."""
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
