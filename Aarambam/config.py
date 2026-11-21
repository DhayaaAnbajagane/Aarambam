"""Config-file templates and writers for the Aarambam 2LPT executables."""
import os, tempfile, textwrap

__all__ = ['AUTO_FAST_MAX_NMESH', 'DEFAULT_BASIS_CONFIG', 'DEFAULT_BASIS_INPUTS', 'DEFAULT_RESBASIS_CONFIG', 'DEFAULT_RESBASIS_INPUTS', 'DEFAULT_BINNED_CONFIG', 'DEFAULT_BINNED_INPUTS', 'make_config_basis', 'make_config_resbasis', 'make_config_binned', 'binned_kernel_inputs', 'make_example_config_basis', 'make_example_config_resbasis', 'make_example_config_binned']


#The three configs share one template. The modal (Basis/ResBasis) and binned variants differ
#only in the mode/table blocks, and ResBasis additionally sets WhichSpectrum and the P(k) oscillation.
_TEMPLATE = """
Nmesh                               %(Nmesh)d
Nsample                             %(Nsample)d
Nmax                                %(Nmax)d
Box                                 %(Lbox).17g
FileBase                            %(FileBase)s
OutputDir                           %(OutputDir)s
GlassFile                           %(GlassFile)s 
GlassTileFac                        %(GlassTileFac)d

Seed                                %(seed)d


Omega                               %(Om).17g
OmegaLambda                         %(Ode).17g
OmegaBaryon                         0.00000 #Doesn't really matter since we use the transfer function from CAMB
OmegaDM_2ndSpecies                  0.00
HubbleParam                         %(h).17g
Sigma8                              %(sigma8).17g
PrimordialIndex                     %(n_s).17g
Redshift                            %(z_ini).17g
Fnl                                 %(Fnl).17g
@MODES@
FixedAmplitude                      0
PhaseFlip                           0
SphereMode                          0
SavePotentialField                  %(SavePotentialField)d
ModeTables                          %(ModeTables)d #1: per-mode tables and cached fields (faster, more memory); 0: least memory
FFTWWisdomDir                       %(FFTWWisdomDir)s #FFT plans: auto (measured once, kept in ~/.cache/aarambam), a directory, or none
FFTBackend                          %(FFTBackend)s #FFTs of the non-Gaussian term: fftw (FFTW3-MPI) or band (band-limited, faster)
NGPrecision                         %(NGPrecision)s #non-Gaussian term's fields and FFTs: double, or float (faster, ~1e-6 relative)

WhichSpectrum                       @SPECTRUM@
WhichTransfer                       2 #Fixed to 2 because we always want a tabulated Transfer function (from CAMB)

FileWithInputSpectrum               ThisIsAFakeStringThatIsNeverUsed
FileWithInputTransfer               %(TransferPath)s
@TABLES@

NumFilesWrittenInParallel           %(NWriteoutProcesses)d

InputSpectrum_UnitLength_in_cm      3.085678e24 #One Mpc in cm. This is fixed
UnitLength_in_cm                    3.085678e24 #One Mpc in cm, units of  Pkdgrav (gadget uses kpc)
UnitMass_in_g                       1.989e43
UnitVelocity_in_cm_per_s            1e5

ShapeGamma                          0.0    #Dont use this but it complains if I dont include it.
WDM_On                              0
WDM_Vtherm_On                       0
WDM_PartMass_in_kev                 10.0
"""

_NO_OSCILLATION = "0 #Fixed to 0 because we always want to use only the Transfer Function"

_MODAL_MODES  = "N_modes                             %(N_modes)d\n\n"
_MODAL_TABLES = ("FileWithAlphaCoeff                  %(AlphaPath)s\n"
                 "FileWithX0Coeff                     %(AiPath)s")

_BINNED_TABLES = ("FileWithBinnedK1Edges               %(BinnedK1EdgesPath)s     #k1 and k2 (the internal legs) share these\n"
                  "FileWithBinnedK3Edges               %(BinnedK3EdgesPath)s\n"
                  "FileWithBinnedKernelValues          %(BinnedKernelValuesPath)s\n"
                  "BinnedPNGFilter                     %(BinnedPNGFilterCode)d         #1,2,3 for top-hat/linear/cubic\n"
                  "BinnedPNGOutputFilter               %(BinnedPNGOutputFilterCode)d   #1,2,3 for constant/linear/cubic\n"
                  "BinnedPNGCoordinate                 2                               #1,2   for k/log(k). WE ONLY IMPLEMENT LOG(k) FOR NOW\n"
                  "BinnedPNGCacheBlockSize             %(BinnedPNGCacheBlockSize)d #Filtered fields kept in memory per block: fewer FFTs, ~that many extra grids\n"
                  "BinnedPNGMultiGrid                  %(BinnedPNGMultiGrid)d         #1: each bin pair on its smallest alias-free grid (same result, faster)")

_OSCILLATION = ("\nA_res               %(Pk_A_res).17g\n"
                "w_res               %(Pk_w_res).17g\n"
                "phi_res             %(Pk_phi_res).17g\n")


def _fill(modes, spectrum, tables, extra = ''):
    return (_TEMPLATE.replace('@MODES@', modes).replace('@SPECTRUM@', spectrum)
                     .replace('@TABLES@', tables) + extra)


DEFAULT_BASIS_CONFIG    = _fill(_MODAL_MODES, _NO_OSCILLATION, _MODAL_TABLES)
DEFAULT_RESBASIS_CONFIG = _fill(_MODAL_MODES, '%(Pk_mode)d', _MODAL_TABLES, _OSCILLATION)
DEFAULT_BINNED_CONFIG   = _fill('', _NO_OSCILLATION, _BINNED_TABLES)

DEFAULT_BASIS_INPUTS = dict(Nmesh = 256, Nsample = 256, Nmax = 256, Lbox = 512, FileBase = 'Aarambam_ics',
                            OutputDir = os.environ.get('TMPDIR', tempfile.gettempdir()), 
                            GlassFile = os.path.dirname(__file__) + "/defaults/dummy_glass_dmonly_64.dat",
                            GlassTileFac = 4,
                            seed = 42, Om = 0.3, Ode = 0.7, h = 0.7, sigma8 = 0.8, n_s = 0.96, z_ini = 49,
                            Fnl = 100, N_modes = 15, 
                            TransferPath = os.path.dirname(__file__) + "/defaults/TransferFunctionConverted.dat",
                            AlphaPath    = os.path.dirname(__file__) + "/defaults/AlphaTable.dat",
                            AiPath       = os.path.dirname(__file__) + "/defaults/AiTable.dat",
                            SavePotentialField = True, NWriteoutProcesses = 1,
                            ModeTables = 'auto', FFTWWisdomDir = 'auto', FFTBackend = 'fftw',
                            NGPrecision = 'double'
                            )


DEFAULT_RESBASIS_INPUTS = DEFAULT_BASIS_INPUTS.copy()
DEFAULT_RESBASIS_INPUTS.update(dict(Pk_A_res = 0.9, Pk_w_res = 2, Pk_phi_res = 0.0, Pk_mode = 1))

DEFAULT_BINNED_INPUTS = dict(Nmesh = 256, Nsample = 256, Nmax = 256, Lbox = 512, FileBase = 'Aarambam_ics',
                             OutputDir = os.environ.get('TMPDIR', tempfile.gettempdir()),
                             GlassFile = os.path.dirname(__file__) + "/defaults/dummy_glass_dmonly_64.dat",
                             GlassTileFac = 4,
                             seed = 42, Om = 0.3, Ode = 0.7, h = 0.7, sigma8 = 0.8, n_s = 0.96, z_ini = 49,
                             Fnl = 100,
                             TransferPath = os.path.dirname(__file__) + "/defaults/TransferFunctionConverted.dat",
                             BinnedK1EdgesPath      = os.path.dirname(__file__) + "/defaults/BinnedKernel_k1_edges.txt",
                             BinnedK3EdgesPath      = os.path.dirname(__file__) + "/defaults/BinnedKernel_k3_edges.txt",
                             BinnedKernelValuesPath = os.path.dirname(__file__) + "/defaults/BinnedKernel_hybrid_values.txt",
                             BinnedPNGFilterCode       = 3,
                             BinnedPNGOutputFilterCode = 3,
                             BinnedPNGCacheBlockSize   = 'auto',
                             BinnedPNGMultiGrid        = 1,
                             SavePotentialField = True, NWriteoutProcesses = 1,
                             ModeTables = 'auto', FFTWWisdomDir = 'auto', FFTBackend = 'fftw',
                             NGPrecision = 'double'
                            )


#Speed/memory options set to 'auto' take their fast (memory-hungry) value for Nmesh up to this, and their
#low-memory value above it. See "Speed and memory" in docs/source/conventions.rst.
AUTO_FAST_MAX_NMESH = 512


def _resolve_auto(value, Nmesh, fast, lean, name):
    """An integer option, or ``'auto'``: ``fast`` if ``Nmesh <= AUTO_FAST_MAX_NMESH``, else ``lean``."""
    if isinstance(value, str):
        if value != 'auto':
            raise ValueError(f"{name} must be an integer or 'auto', got {value!r}")
        return fast if Nmesh <= AUTO_FAST_MAX_NMESH else lean
    return int(value)


def _resolve_mode_tables(ModeTables, Nmesh, lean = 0):
    """ModeTables 0/1, or 'auto' (1 up to AUTO_FAST_MAX_NMESH, ``lean`` above)."""
    ModeTables = _resolve_auto(ModeTables, Nmesh, 1, lean, 'ModeTables')
    if ModeTables not in (0, 1):
        raise ValueError(f"ModeTables must be 0, 1 or 'auto', got {ModeTables!r}")
    return ModeTables


def make_config_basis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                      seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                      AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses, ModeTables = 'auto',
                      FFTWWisdomDir = 'auto', FFTBackend = 'fftw', NGPrecision = 'double'):
    """
    Generate a basis configuration file string for initial condition generation
    with the standard basis method.

    This function fills a default configuration template with the provided
    simulation and cosmology parameters, returning the formatted string.
    The configuration is intended for use in non-Gaussian initial condition
    generation codes (e.g. N-GenIC variants with modal decomposition support).

    Parameters
    ----------
    Nmesh : int
        Number of mesh cells per dimension used for the density and potential grids.
    Nsample : int
        Sets the maximum k that the code uses, i.e. this effectively determines the Nyquist frequency 
        that the code assumes, k_Nyquist = 2*PI/Box * Nsample/2 Normally, one chooses Nsample such that 
        Ntot =  Nsample^3, where Ntot is the total number of particles
    Nmax : int
        Maximum wavenumber index or mode cutoff used in the modal expansion. In units of
        the fundamental frequency of the box. Must be >= Nsample (the C code asserts this);
        normally set it equal to Nsample.
    Lbox : float
        Side length of the simulation box in comoving units (e.g. Mpc/h).
    FileBase : str
        Base name for all output files
    OutputDir : str
        Directory where output files will be written. The defaults use
        ``$TMPDIR``, or ``tempfile.gettempdir()`` if it is unset.
    GlassFile : str
        Path to the glass file used to generate the initial particle load. Aarambam provides a default.
    GlassTileFac : int
        Tiling factor applied to the glass file to match the desired number of particles.
    seed : int
        Random seed for generating the random phases of the simulation.
    Om : float
        Matter density parameter :math:`\\Omega_m`.
    Ode : float
        Dark energy density parameter :math:`\\Omega_\\Lambda`.
    h : float
        Dimensionless Hubble parameter.
    sigma8 : float
        Normalization of the linear power spectrum at 8 Mpc/h.
    n_s : float
        Scalar spectral index of primordial fluctuations.
    z_ini : float
        Initial redshift for the simulation. Good choice is z = 49.
    Fnl : float
        Non-Gaussian amplitude parameter.
    N_modes : int
        Number of modes in the modal expansion. The expansion happens
        outside of LPT being run. Make sure you ran it first.
    TransferPath : str
        Path to the transfer function file from CAMB/CLASS.
    AlphaPath : str
        Path to the alpha-coefficients file for the modal basis.
    AiPath : str
        Path to the A_i-coefficients file for the modal basis.
    SavePotentialField : int
        Flag (0 or 1) indicating whether to save the potential field to disk.
    NWriteoutProcesses : int
        Written as ``NumFilesWrittenInParallel``. Kept for config compatibility: the C code
        writes all files from task 0 and does not use it. Default 1.
    ModeTables : {0, 1, 'auto'}, optional
        Speed/memory switch of the 2LPT code. ``1`` keeps all ``N_modes`` mode fields in
        real space and tabulates the per-mode kernel values (about ``N_modes`` extra grids
        of ``8 Nmesh^3`` bytes, half the FFTs): the fast setting. ``0`` rebuilds the fields
        for every mode pair and uses the least memory. Both give the same ICs to rounding.
        ``'auto'`` (default) picks 1 for ``Nmesh <= AUTO_FAST_MAX_NMESH`` (512) and 0 above.
    FFTWWisdomDir : str, optional
        Directory in which the 2LPT code keeps its FFT plans: they are measured (``FFTW_MEASURE``)
        the first time a grid size and rank count are run, and later runs read them back at no cost.
        A path: that directory (created if needed; the run stops with an error if it cannot be
        created or written). ``'auto'`` (default): ``$XDG_CACHE_HOME/aarambam`` or
        ``~/.cache/aarambam`` (the ``AARAMBAM_FFTW_WISDOM_DIR`` environment variable, if set, takes
        its place). ``'none'``: plan with ``FFTW_ESTIMATE``, no measuring and no file (the results
        are then bit-for-bit reproducible across rank counts).
    FFTBackend : {'fftw', 'band'}, optional
        FFTs of the non-Gaussian term. ``'fftw'`` (default): FFTW3's MPI transforms of the full
        grid. ``'band'``: the 2LPT code's own slab FFT (``src/lpt/common/band_fft.c``, built on
        FFTW3's serial transforms), which transforms and communicates only the band of k-space
        that is sampled (``|k| <= Nsample/2 + 1`` per axis) and spreads it evenly over the ranks:
        faster when ``Nmesh > Nsample``, the same ICs to rounding. It keeps one communication
        buffer of about ``(Nsample / Nmesh)^2`` of a grid per rank (0.44 for ``Nmesh = 3/2 Nsample``).
    NGPrecision : {'double', 'float'}, optional
        Precision of the non-Gaussian term's real-space fields, their products and their FFTs.
        ``'double'`` (default). ``'float'``: single precision, always with the band FFT; about 40%
        faster for that stage and half its memory. The Gaussian potential, the accumulated
        non-Gaussian term and the 2LPT stage stay double; the non-Gaussian term then agrees with
        ``'double'`` to ~4e-7 (binned) and ~1e-6 (basis) relative. Needs FFTW3 also built in single
        precision (``--enable-float``, see the installation page).

    Returns
    -------
    str
        A configuration file string with all placeholders filled by the given parameters.
        The string is formatted according to ``DEFAULT_BASIS_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_basis(
    ...     Nmesh=384, Nsample=256, Nmax=256, Lbox=1000.0,
    ...     FileBase="ICs", OutputDir="./output/",
    ...     GlassFile="glass256.dat", GlassTileFac=2,
    ...     seed=12345, Om=0.3, Ode=0.7, h=0.7,
    ...     sigma8=0.8, n_s=0.965, z_ini=99.0,
    ...     Fnl=10.0, N_modes=50,
    ...     TransferPath="transfer.dat", AlphaPath="alpha.dat", AiPath="ai.dat",
    ...     SavePotentialField=1, NWriteoutProcesses=1
    ... )
    >>> print(cfg)  # prints the beginning of the config file
    """
    ModeTables = _resolve_mode_tables(ModeTables, Nmesh)
    return textwrap.dedent(DEFAULT_BASIS_CONFIG % locals())


def make_config_resbasis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                         seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                         AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses,
                         Pk_A_res, Pk_w_res, Pk_phi_res, Pk_mode, ModeTables = 'auto',
                         FFTWWisdomDir = 'auto', FFTBackend = 'fftw', NGPrecision = 'double'):

    """
    Generate a basis configuration file string for initial condition generation
    with the standard basis method supplemented with oscillations in the
    power spectra as well.

    This function fills a default configuration template with the provided
    simulation and cosmology parameters, returning the formatted string.
    The configuration is intended for use in non-Gaussian initial condition
    generation codes (e.g. N-GenIC variants with modal decomposition support).

    Parameters
    ----------
    Nmesh : int
        Number of mesh cells per dimension used for the density and potential grids.
    Nsample : int
        Sets the maximum k that the code uses, i.e. this effectively determines the Nyquist frequency 
        that the code assumes, k_Nyquist = 2*PI/Box * Nsample/2 Normally, one chooses Nsample such that 
        Ntot =  Nsample^3, where Ntot is the total number of particles
    Nmax : int
        Maximum wavenumber index or mode cutoff used in the modal expansion. In units of
        the fundamental frequency of the box. Must be >= Nsample (the C code asserts this);
        normally set it equal to Nsample.
    Lbox : float
        Side length of the simulation box in comoving units (e.g. Mpc/h).
    FileBase : str
        Base name for all output files
    OutputDir : str
        Directory where output files will be written. The defaults use
        ``$TMPDIR``, or ``tempfile.gettempdir()`` if it is unset.
    GlassFile : str
        Path to the glass file used to generate the initial particle load. Aarambam provides a default.
    GlassTileFac : int
        Tiling factor applied to the glass file to match the desired number of particles.
    seed : int
        Random seed for generating the random phases of the simulation.
    Om : float
        Matter density parameter :math:`\\Omega_m`.
    Ode : float
        Dark energy density parameter :math:`\\Omega_\\Lambda`.
    h : float
        Dimensionless Hubble parameter.
    sigma8 : float
        Normalization of the linear power spectrum at 8 Mpc/h.
    n_s : float
        Scalar spectral index of primordial fluctuations.
    z_ini : float
        Initial redshift for the simulation. Good choice is z = 49.
    Fnl : float
        Non-Gaussian amplitude parameter.
    N_modes : int
        Number of modes in the modal expansion. The expansion happens
        outside of LPT being run. Make sure you ran it first.
    TransferPath : str
        Path to the transfer function file from CAMB/CLASS.
    AlphaPath : str
        Path to the alpha-coefficients file for the modal basis.
    AiPath : str
        Path to the A_i-coefficients file for the modal basis.
    SavePotentialField : int
        Flag (0 or 1) indicating whether to save the potential field to disk.
    NWriteoutProcesses : int
        Written as ``NumFilesWrittenInParallel``. Kept for config compatibility: the C code
        writes all files from task 0 and does not use it. Default 1.
    Pk_A_res : float
        Amplitude of the power spectrum oscillations
    Pk_w_res : float
        Frequency (in either linear or log-k, set by ``Pk_mode`` below) of the oscillations
    Pk_phi_res : float
        Phase of the oscillations in P(k)
    Pk_mode : int
        Control flag for the different P(k) oscillation options:
            0 - No oscillations
            1 - Oscillations in :math:`k`.
            2 - Oscillations in :math:`\\log k`.
    ModeTables : {0, 1, 'auto'}, optional
        Speed/memory switch of the 2LPT code; see ``make_config_basis``.
    FFTWWisdomDir : str, optional
        FFT plan directory of the 2LPT code; see ``make_config_basis``.
    FFTBackend : {'fftw', 'band'}, optional
        FFTs of the non-Gaussian term; see ``make_config_basis``.
    NGPrecision : {'double', 'float'}, optional
        Precision of the non-Gaussian stage; see ``make_config_basis``.

    Returns
    -------
    str
        A configuration file string with all placeholders filled by the given parameters.
        The string is formatted according to ``DEFAULT_RESBASIS_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_resbasis(
    ...     Nmesh=384, Nsample=256, Nmax=256, Lbox=1000.0,
    ...     FileBase="ICs", OutputDir="./output/",
    ...     GlassFile="glass256.dat", GlassTileFac=2,
    ...     seed=12345, Om=0.3, Ode=0.7, h=0.7,
    ...     sigma8=0.8, n_s=0.965, z_ini=99.0,
    ...     Fnl=10.0, N_modes=50,
    ...     TransferPath="transfer.dat", AlphaPath="alpha.dat", AiPath="ai.dat",
    ...     SavePotentialField=1, NWriteoutProcesses=1,
    ...     Pk_A_res = 0.8, Pk_w_res = 3, Pk_phi_res = 0, Pk_mode = 1
    ... )
    >>> print(cfg)  # prints the beginning of the config file
    """

    ModeTables = _resolve_mode_tables(ModeTables, Nmesh)
    return textwrap.dedent(DEFAULT_RESBASIS_CONFIG % locals())


def make_config_binned(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                       seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, TransferPath,
                       BinnedK1EdgesPath, BinnedK3EdgesPath, BinnedKernelValuesPath,
                       BinnedPNGFilterCode, BinnedPNGOutputFilterCode, BinnedPNGCacheBlockSize,
                       SavePotentialField, NWriteoutProcesses, ModeTables = 'auto',
                       FFTWWisdomDir = 'auto', FFTBackend = 'fftw', NGPrecision = 'double',
                       BinnedPNGMultiGrid = 1):
    """
    Generate a configuration string for the binned-PNG initial-condition generator.

    The returned configuration uses precomputed one-dimensional bin-edge tables
    and a C-readable binned-kernel table produced by
    ``BasicBinnedDecompose.finalize``. The interpolation codes supplied here must
    agree with the filter used when that kernel table was generated.

    Parameters
    ----------
    Nmesh : int
        Number of mesh cells per dimension used for the density and potential grids.
    Nsample : int
        Fourier sampling size used by the initial-condition generator. The assumed
        Nyquist wavenumber is ``2*pi/Lbox * Nsample/2``.
    Nmax : int
        Maximum dimensionless wavenumber index used by the generator. Must be
        >= ``Nsample`` (the C code asserts this); normally set equal to ``Nsample``.
    Lbox : float
        Side length of the simulation box in comoving Mpc/h.
    FileBase : str
        Base name for the generated initial-condition files.
    OutputDir : str
        Directory in which the generated files are written. The defaults use
        ``$TMPDIR``, or ``tempfile.gettempdir()`` if it is unset.
    GlassFile : str
        Path to the glass particle load.
    GlassTileFac : int
        Number of times the supplied glass is tiled along each dimension.
    seed : int
        Random seed used to generate the Gaussian phases.
    Om : float
        Matter density parameter :math:`\\Omega_m`.
    Ode : float
        Dark-energy density parameter :math:`\\Omega_\\Lambda`.
    h : float
        Dimensionless Hubble parameter.
    sigma8 : float
        Linear-theory normalization at 8 Mpc/h.
    n_s : float
        Scalar spectral index.
    z_ini : float
        Initial redshift.
    Fnl : float
        Primordial non-Gaussian amplitude.
    TransferPath : str
        Path to the tabulated CAMB/CLASS transfer function in Aarambam format.
    BinnedK1EdgesPath : str
        Path to ``BinnedKernel_k1_edges.txt``: the bins of both internal legs, ``k1`` and ``k2``.
    BinnedK3EdgesPath : str
        Path to ``BinnedKernel_k3_edges.txt``.
    BinnedKernelValuesPath : str
        Path to the C-readable kernel table. Should have format
        ``BinnedKernel_values.txt`` for the top-hat filter and
        ``BinnedKernel_hybrid_values.txt`` for a hybrid filter.
    BinnedPNGFilterCode : int
        Interpolation code on the internal ``k1`` and ``k2`` legs:
        ``1`` for top-hat, ``2`` for linear, or ``3`` for cubic.
    BinnedPNGOutputFilterCode : int
        Cell-local interpolation code on the output ``k3`` leg:
        ``1`` for constant, ``2`` for linear, or ``3`` for cubic.
    BinnedPNGCacheBlockSize : int or 'auto'
        Number of filtered k1 (and k2) real-space basis fields kept in memory per block;
        0 disables the blocked path. Larger values
        cut repeated FFTs at the cost of roughly that many extra grids of memory (the C
        code caps it at the number of k1 bins); the result does not depend on it.
        ``'auto'`` (default) is 64, i.e. all bins, for ``Nmesh <= AUTO_FAST_MAX_NMESH``
        and 4 above.
    SavePotentialField : int or bool
        Whether to save the Gaussian and non-Gaussian potential fields.
    NWriteoutProcesses : int
        Written as ``NumFilesWrittenInParallel``. Kept for config compatibility: the C code
        writes all files from task 0 and does not use it. Default 1.
    ModeTables : {0, 1, 'auto'}, optional
        Speed/memory switch of the 2LPT code. ``1`` tabulates |k|, the output bin and
        the in-bin coordinate of every sampled mode once (56 bytes per sampled mode, about
        one extra grid at ``Nmesh = 3/2 Nsample``): the fast setting. ``0`` recomputes them
        for every bin pair. The result is bit-identical. ``'auto'`` (default) is 1 at every
        ``Nmesh``; the binned memory/speed trade-off is ``BinnedPNGCacheBlockSize``.
    FFTWWisdomDir : str, optional
        FFT plan directory of the 2LPT code; see ``make_config_basis``.
    FFTBackend : {'fftw', 'band'}, optional
        FFTs of the non-Gaussian term; see ``make_config_basis``.
    NGPrecision : {'double', 'float'}, optional
        Precision of the non-Gaussian stage; see ``make_config_basis``.

    Returns
    -------
    str
        Configuration text formatted using ``DEFAULT_BINNED_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_binned(
    ...     Nmesh = 384, Nsample = 256, Nmax = 256, Lbox = 1000.0,
    ...     FileBase = 'ICs', OutputDir = './output/',
    ...     GlassFile = 'glass256.dat', GlassTileFac = 2,
    ...     seed = 12345, Om = 0.3, Ode = 0.7, h = 0.7,
    ...     sigma8 = 0.8, n_s = 0.965, z_ini = 49.0, Fnl = 100.0,
    ...     TransferPath = 'TransferFunctionConverted.dat',
    ...     BinnedK1EdgesPath = 'BinnedKernel_k1_edges.txt',
    ...     BinnedK3EdgesPath = 'BinnedKernel_k3_edges.txt',
    ...     BinnedKernelValuesPath = 'BinnedKernel_hybrid_values.txt',
    ...     BinnedPNGFilterCode = 3, BinnedPNGOutputFilterCode = 3,
    ...     BinnedPNGCacheBlockSize = 'auto', ModeTables = 'auto',
    ...     SavePotentialField = True, NWriteoutProcesses = 1,
    ... )
    """
    _check_binned_codes(BinnedKernelValuesPath, BinnedPNGFilterCode, BinnedPNGOutputFilterCode)
    BinnedPNGMultiGrid = int(bool(BinnedPNGMultiGrid))
    ModeTables = _resolve_mode_tables(ModeTables, Nmesh, lean = 1)      #the binned tables cost ~1 grid
    BinnedPNGCacheBlockSize = _resolve_auto(BinnedPNGCacheBlockSize, Nmesh, 64, 4, 'BinnedPNGCacheBlockSize')
    return textwrap.dedent(DEFAULT_BINNED_CONFIG % locals())


def make_example_config_basis():
    """Return an example configuration for the standard modal IC generator."""
    return make_config_basis(**DEFAULT_BASIS_INPUTS)


def make_example_config_resbasis():
    """Return an example configuration for the modal P(k) + B(k) IC generator."""
    return make_config_resbasis(**DEFAULT_RESBASIS_INPUTS)


def make_example_config_binned():
    """Return an example configuration for the binned-PNG IC generator."""
    return make_config_binned(**DEFAULT_BINNED_INPUTS)


#C filter codes of the binned 2LPT code (src/lpt/binned/binned_vars.h): 1 heaviside, 2 linear, 3 cubic
_BINNED_CODES = {'heaviside': 1, 'top_hat': 1, 'linear': 2, 'cubic': 3}


def binned_kernel_inputs(kernel_dir):
    """
    Read the ``BinnedKernel_metadata.txt`` written by a binned decomposition
    (``Decomposer.go(..., outdir = kernel_dir)``) and return the
    ``make_config_binned`` inputs that describe its tables: the three table
    paths and the two C filter codes.

    Parameters
    ----------
    kernel_dir : str
        Directory the binned decomposition wrote its tables into.

    Returns
    -------
    dict
        ``BinnedK1EdgesPath``, ``BinnedK3EdgesPath``,
        ``BinnedKernelValuesPath``, ``BinnedPNGFilterCode`` and
        ``BinnedPNGOutputFilterCode``.

    Examples
    --------
    >>> inputs = dict(DEFAULT_BINNED_INPUTS)
    >>> inputs.update(binned_kernel_inputs(outdir))
    >>> cfg = make_config_binned(**inputs)
    """
    from .basis.binned.io import _metadata_path, _parse_metadata

    path = _metadata_path(kernel_dir)
    meta = _parse_metadata(path)
    if not meta:
        raise FileNotFoundError(f"No binned kernel metadata at {path}")

    if meta['normalized_filter_mode'] == 'heaviside':
        codes  = (1, 1)
        values = meta['values_file']
    else:
        codes  = (_BINNED_CODES[meta['hybrid_input_interpolator']], _BINNED_CODES[meta['hybrid_output_filter']])
        values = meta['hybrid_values_file']

    return dict(BinnedK1EdgesPath         = os.path.join(kernel_dir, meta['k1_edges_file']),
                BinnedK3EdgesPath         = os.path.join(kernel_dir, meta['k3_edges_file']),
                BinnedKernelValuesPath    = os.path.join(kernel_dir, values),
                BinnedPNGFilterCode       = codes[0],
                BinnedPNGOutputFilterCode = codes[1])


def _check_binned_codes(values_path, filter_code, output_code):
    """Raise if the kernel table's own metadata disagrees with the requested C filter codes."""
    kernel_dir = os.path.dirname(os.path.abspath(values_path))
    try:
        expected = binned_kernel_inputs(kernel_dir)
    except (FileNotFoundError, KeyError):
        return #No (usable) metadata next to the table, e.g. the shipped defaults

    if os.path.abspath(expected['BinnedKernelValuesPath']) != os.path.abspath(values_path):
        return #A different table in the same directory; nothing to compare against
    if (filter_code, output_code) != (expected['BinnedPNGFilterCode'], expected['BinnedPNGOutputFilterCode']):
        raise ValueError(f"{values_path} was written for BinnedPNGFilterCode = {expected['BinnedPNGFilterCode']}, "
                         f"BinnedPNGOutputFilterCode = {expected['BinnedPNGOutputFilterCode']}, but "
                         f"{filter_code}, {output_code} were passed. Use binned_kernel_inputs() to get them.")
