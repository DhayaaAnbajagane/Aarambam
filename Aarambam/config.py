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
    Config file text for ``2LPTBasis`` (Legendre/modal decomposition).

    Parameters
    ----------
    Nmesh : int
        Grid points per side of the potential and displacement grids.
    Nsample : int
        Largest sampled mode: the code keeps modes with ``|k| <= Nsample/2`` (in units of
        ``2 pi / Lbox``) per axis. Usually the particle count per side; use ``Nmesh >= 3/2 Nsample``
        for an alias-free non-Gaussian term.
    Nmax : int
        Largest wavenumber of the decomposition, in units of ``2 pi / Lbox``. Must be ``>= Nsample``;
        normally equal to it.
    Lbox : float
        Box side in Mpc/h.
    FileBase : str
        Base name of the snapshot files.
    OutputDir : str
        Directory for the snapshots and potentials.
    GlassFile : str
        Glass file of the unperturbed particle load.
    GlassTileFac : int
        Times the glass is tiled per side (particles per side = glass side x ``GlassTileFac``).
    seed : int
        Seed of the Gaussian random field.
    Om, Ode, h, sigma8, n_s : float
        Omega_m, Omega_Lambda, H0 / (100 km/s/Mpc), sigma_8 and the scalar spectral index.
    z_ini : float
        Starting redshift (e.g. 49).
    Fnl : float
        Amplitude of the template: :math:`B_\\Phi = 6 f_{\\rm NL} B_{\\rm template}`.
    N_modes : int
        Number of 1D Legendre modes of the decomposition (``N_modes`` of ``Decomposer``).
    TransferPath : str
        Two-column (k, T(k)) transfer function (see ``camb2input``).
    AlphaPath, AiPath : str
        ``AlphaTable.dat`` and ``AiTable.dat`` written by the Legendre decomposition.
    SavePotentialField : bool or int
        Also write the Gaussian and non-Gaussian potentials (read them with ``collate_potential``).
    NWriteoutProcesses : int
        Written as ``NumFilesWrittenInParallel``; not used by the C code.
    ModeTables : {0, 1, 'auto'}, optional
        ``1``: keep all mode fields in real space (about ``N_modes`` extra grids of ``8 Nmesh^3``
        bytes, half the FFTs). ``0``: least memory. Same ICs to rounding. ``'auto'``: 1 for
        ``Nmesh <= AUTO_FAST_MAX_NMESH`` (512), else 0.
    FFTWWisdomDir : str, optional
        Where measured FFT plans are kept: a directory, ``'auto'`` (``~/.cache/aarambam``, or
        ``$AARAMBAM_FFTW_WISDOM_DIR``) or ``'none'`` (unmeasured plans, no file).
    FFTBackend : {'fftw', 'band'}, optional
        FFTs of the non-Gaussian term: FFTW3-MPI on the whole grid, or the code's own FFT of the
        sampled band only (faster when ``Nmesh > Nsample``; same ICs to rounding).
    NGPrecision : {'double', 'float'}, optional
        Precision of the non-Gaussian stage. ``'float'`` is ~40% faster for that stage, agrees to
        ~1e-6 relative and needs FFTW built with ``--enable-float``.

    Returns
    -------
    str
        The config text; write it to a file and pass that to ``aarambam 2lpt basis``.

    See the Conventions and Running faster pages of the docs for details on the options.
    """
    ModeTables = _resolve_mode_tables(ModeTables, Nmesh)
    return textwrap.dedent(DEFAULT_BASIS_CONFIG % locals())


def make_config_resbasis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                         seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                         AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses,
                         Pk_A_res, Pk_w_res, Pk_phi_res, Pk_mode, ModeTables = 'auto',
                         FFTWWisdomDir = 'auto', FFTBackend = 'fftw', NGPrecision = 'double'):

    """
    Config file text for ``2LPTBasis`` with an oscillating primordial power spectrum (ResBasis).

    Parameters
    ----------
    (all of make_config_basis)
        See ``make_config_basis``.
    Pk_A_res, Pk_w_res, Pk_phi_res : float
        Amplitude, frequency and phase of the oscillation in P(k).
    Pk_mode : {0, 1, 2}
        ``0``: no oscillation (same ICs as ``make_config_basis``); ``1``: oscillation in k;
        ``2``: oscillation in log k.

    Returns
    -------
    str
        The config text, for ``aarambam 2lpt resbasis``.
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
    Config file text for ``2LPTBinned`` (binned decomposition).

    ``binned_kernel_inputs(outdir)`` returns the five table and filter arguments for tables
    written by ``Decomposer.go(..., outdir = outdir)``.

    Parameters
    ----------
    (all of make_config_basis except N_modes, AlphaPath, AiPath)
        See ``make_config_basis``. ``ModeTables`` here tabulates the per-mode bin lookups (about one
        extra grid, bit-identical result); ``'auto'`` is 1 at every ``Nmesh``.
    BinnedK1EdgesPath, BinnedK3EdgesPath : str
        Bin edges of the internal legs (k1 and k2 share them) and of the output leg k3.
    BinnedKernelValuesPath : str
        Kernel table (``BinnedKernel_hybrid_values.txt``, or ``BinnedKernel_values.txt`` for
        top-hat filters).
    BinnedPNGFilterCode : {1, 2, 3}
        Interpolation on k1, k2: top-hat, linear or cubic. Must match the table.
    BinnedPNGOutputFilterCode : {1, 2, 3}
        Interpolation within a k3 bin: constant, linear or cubic. Must match the table.
    BinnedPNGCacheBlockSize : int or 'auto'
        Filtered k1 fields kept in memory at once (about that many extra grids; fewer FFTs).
        ``'auto'``: 64 (all bins) for ``Nmesh <= 512``, else 4.
    BinnedPNGMultiGrid : bool, optional
        Compute each bin pair on the smallest alias-free grid (default; same result to ~1e-14,
        faster). ``False``: every pair on the full grid.

    Returns
    -------
    str
        The config text, for ``aarambam 2lpt binned``.

    Raises
    ------
    ValueError
        If the filter codes disagree with the metadata written next to the kernel table.
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
