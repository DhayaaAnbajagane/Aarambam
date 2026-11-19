import numpy as np, os, sys, gc
import textwrap, glob, tqdm

__all__ = ['DEFAULT_BASIS_CONFIG', 'DEFAULT_BASIS_INPUTS', 'DEFAULT_RESBASIS_CONFIG', 'DEFAULT_RESBASIS_INPUTS',
           'DEFAULT_BINNED_CONFIG', 'DEFAULT_BINNED_INPUTS',
           'make_config_basis', 'make_config_resbasis', 'make_config_binned',
           'make_example_config_basis', 'make_example_config_resbasis', 'make_example_config_binned',
           'collate_potential', 'camb2input', 'Decomposer']


DEFAULT_BASIS_CONFIG = """
Nmesh                               %(Nmesh)d
Nsample                             %(Nsample)d
Nmax                                %(Nmax)d
Box                                 %(Lbox)f
FileBase                            %(FileBase)s
OutputDir                           %(OutputDir)s
GlassFile                           %(GlassFile)s 
GlassTileFac                        %(GlassTileFac)d

Seed                                %(seed)d


Omega                               %(Om)f
OmegaLambda                         %(Ode)f
OmegaBaryon                         0.00000 #Doesn't really matter since we use the transfer function from CAMB
OmegaDM_2ndSpecies                  0.00
HubbleParam                         %(h)f
Sigma8                              %(sigma8)f
PrimordialIndex                     %(n_s)f
Redshift                            %(z_ini)f
Fnl                                 %(Fnl)f
N_modes                             %(N_modes)d


FixedAmplitude                      0
PhaseFlip                           0
SphereMode                          0
SavePotentialField                  %(SavePotentialField)d

WhichSpectrum                       0 #Fixed to 0 because we always want to use only the Transfer Function
WhichTransfer                       2 #Fixed to 2 because we always want a tabulated Transfer function (from CAMB)

FileWithInputSpectrum               ThisIsAFakeStringThatIsNeverUsed
FileWithInputTransfer               %(TransferPath)s
FileWithAlphaCoeff                  %(AlphaPath)s
FileWithX0Coeff                     %(AiPath)s

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

DEFAULT_BASIS_INPUTS = dict(Nmesh = 256, Nsample = 256, Nmax = 256, Lbox = 512, FileBase = 'Aarambam_ics',
                            OutputDir = os.environ['TMPDIR'], 
                            GlassFile = os.path.dirname(__file__) + "/defaults/dummy_glass_dmonly_64.dat",
                            GlassTileFac = 4,
                            seed = 42, Om = 0.3, Ode = 0.7, h = 0.7, sigma8 = 0.8, n_s = 0.96, z_ini = 49,
                            Fnl = 100, N_modes = 15, 
                            TransferPath = os.path.dirname(__file__) + "/defaults/TransferFunctionConverted.dat",
                            AlphaPath    = os.path.dirname(__file__) + "/defaults/AlphaTable.dat",
                            AiPath       = os.path.dirname(__file__) + "/defaults/AiTable.dat",
                            SavePotentialField = True, NWriteoutProcesses = 1
                            )


DEFAULT_RESBASIS_CONFIG = (
DEFAULT_BASIS_CONFIG + 
"""
A_res               %(Pk_A_res)f
w_res               %(Pk_w_res)f
phi_res             %(Pk_phi_res)f
"""
)
DEFAULT_RESBASIS_CONFIG = DEFAULT_RESBASIS_CONFIG.replace("0 #Fixed to 0 because we always want to use only the Transfer Function", r"%(Pk_mode)d")

DEFAULT_RESBASIS_INPUTS = DEFAULT_BASIS_INPUTS.copy()
DEFAULT_RESBASIS_INPUTS.update(dict(Pk_A_res = 0.9, Pk_w_res = 2, Pk_phi_res = 0.0))



DEFAULT_BINNED_CONFIG = """
Nmesh                               %(Nmesh)d
Nsample                             %(Nsample)d
Nmax                                %(Nmax)d
Box                                 %(Lbox)f
FileBase                            %(FileBase)s
OutputDir                           %(OutputDir)s
GlassFile                           %(GlassFile)s 
GlassTileFac                        %(GlassTileFac)d

Seed                                %(seed)d


Omega                               %(Om)f
OmegaLambda                         %(Ode)f
OmegaBaryon                         0.00000 #Doesn't really matter since we use the transfer function from CAMB
OmegaDM_2ndSpecies                  0.00
HubbleParam                         %(h)f
Sigma8                              %(sigma8)f
PrimordialIndex                     %(n_s)f
Redshift                            %(z_ini)f
Fnl                                 %(Fnl)f

FixedAmplitude                      0
PhaseFlip                           0
SphereMode                          0
SavePotentialField                  %(SavePotentialField)d

WhichSpectrum                       0 #Fixed to 0 because we always want to use only the Transfer Function
WhichTransfer                       2 #Fixed to 2 because we always want a tabulated Transfer function (from CAMB)

FileWithInputSpectrum               ThisIsAFakeStringThatIsNeverUsed
FileWithInputTransfer               %(TransferPath)s
FileWithBinnedK1Edges               %(BinnedK1EdgesPath)s
FileWithBinnedK2Edges               %(BinnedK2EdgesPath)s
FileWithBinnedK3Edges               %(BinnedK3EdgesPath)s
FileWithBinnedKernelValues          %(BinnedKernelValuesPath)s
BinnedPNGFilter                     %(BinnedPNGFilterCode)d         #1,2,3 for top-hat/linear/cubic
BinnedPNGOutputFilter               %(BinnedPNGOutputFilterCode)d   #1,2,3 for constant/linear/cubic
BinnedPNGCoordinate                 2                               #1,2   for k/log(k). WE ONLY IMPLEMENT LOG(k) FOR NOW
BinnedPNGUseInternalSymmetry        1                               #No reason to turn-off. Uses symmetry between k1, k2
BinnedPNGCacheBlockSize             %(BinnedPNGCacheBlockSize)d #Cache to speed-up code, but increases memory usage by a factor of the block size

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

DEFAULT_BINNED_INPUTS = dict(Nmesh = 256, Nsample = 256, Nmax = 256, Lbox = 512, FileBase = 'Aarambam_ics',
                             OutputDir = os.environ['TMPDIR'],
                             GlassFile = os.path.dirname(__file__) + "/defaults/dummy_glass_dmonly_64.dat",
                             GlassTileFac = 4,
                             seed = 42, Om = 0.3, Ode = 0.7, h = 0.7, sigma8 = 0.8, n_s = 0.96, z_ini = 49,
                             Fnl = 100,
                             TransferPath = os.path.dirname(__file__) + "/defaults/TransferFunctionConverted.dat",
                             BinnedK1EdgesPath      = os.path.dirname(__file__) + "/defaults/BinnedKernel_k1_edges.txt",
                             BinnedK2EdgesPath      = os.path.dirname(__file__) + "/defaults/BinnedKernel_k2_edges.txt",
                             BinnedK3EdgesPath      = os.path.dirname(__file__) + "/defaults/BinnedKernel_k3_edges.txt",
                             BinnedKernelValuesPath = os.path.dirname(__file__) + "/defaults/BinnedKernel_hybrid_values.txt",
                             BinnedPNGFilterCode       = 3,
                             BinnedPNGOutputFilterCode = 3,
                             BinnedPNGCacheBlockSize   = 4,
                             SavePotentialField = True, NWriteoutProcesses = 1
                            )


def make_config_basis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                      seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                      AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses):
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
        the fundamental frequency of the box. Normally set it to Nmesh/2 (the nyquist scale)
    Lbox : float
        Side length of the simulation box in comoving units (e.g. Mpc/h).
    FileBase : str
        Base name for all output files
    OutputDir : str
        Directory where output files will be written.
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
        Number of processes used for writing output files in parallel.

    Returns
    -------
    str
        A configuration file string with all placeholders filled by the given parameters.
        The string is formatted according to ``DEFAULT_BASIS_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_basis(
    ...     Nmesh=256, Nsample=256, Nmax=30, Lbox=1000.0,
    ...     FileBase="ICs", OutputDir="./output/",
    ...     GlassFile="glass256.dat", GlassTileFac=2,
    ...     seed=12345, Om=0.3, Ode=0.7, h=0.7,
    ...     sigma8=0.8, n_s=0.965, z_ini=99.0,
    ...     Fnl=10.0, N_modes=50,
    ...     TransferPath="transfer.dat", AlphaPath="alpha.dat", AiPath="ai.dat",
    ...     SavePotentialField=1, NWriteoutProcesses=4
    ... )
    >>> print(cfg)  # prints the beginning of the config file
    """
    return textwrap.dedent(DEFAULT_BASIS_CONFIG % locals())


def make_config_resbasis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                         seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                         AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses,
                         Pk_A_res, Pk_w_res, Pk_phi_res, Pk_mode):

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
        the fundamental frequency of the box. Normally set it to Nmesh/2 (the nyquist scale)
    Lbox : float
        Side length of the simulation box in comoving units (e.g. Mpc/h).
    FileBase : str
        Base name for all output files
    OutputDir : str
        Directory where output files will be written.
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
        Number of processes used for writing output files in parallel.
    Pk_A_res : float
        Amplitude of the power spectrum oscillations
    Pk_w_res : float
        Frequency (in either linear or log-k, set by ``Pk_mode`` below) of the oscillations
    Pk_phi_res : float
        Phase of the oscillations in P(k)
    Pk_mode : int
        Control flag for the different P(k) oscillation options:
            0 - No oscillations
            1 - Oscillations in :math:``k``.
            2 - Oscillations in :math:``\\log k``.

    Returns
    -------
    str
        A configuration file string with all placeholders filled by the given parameters.
        The string is formatted according to ``DEFAULT_BASIS_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_basis(
    ...     Nmesh=256, Nsample=256, Nmax=30, Lbox=1000.0,
    ...     FileBase="ICs", OutputDir="./output/",
    ...     GlassFile="glass256.dat", GlassTileFac=2,
    ...     seed=12345, Om=0.3, Ode=0.7, h=0.7,
    ...     sigma8=0.8, n_s=0.965, z_ini=99.0,
    ...     Fnl=10.0, N_modes=50,
    ...     TransferPath="transfer.dat", AlphaPath="alpha.dat", AiPath="ai.dat",
    ...     SavePotentialField=1, NWriteoutProcesses=4,
    ...     Pk_A_res = 0.8, Pk_w_res = 3, Pk_phi_res = 0, Pk_mode = 1
    ... )
    >>> print(cfg)  # prints the beginning of the config file
    """

    return textwrap.dedent(DEFAULT_RESBASIS_CONFIG % locals())


def make_config_binned(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                       seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, TransferPath,
                       BinnedK1EdgesPath, BinnedK2EdgesPath, BinnedK3EdgesPath, BinnedKernelValuesPath,
                       BinnedPNGFilterCode, BinnedPNGOutputFilterCode, BinnedPNGCacheBlockSize,
                       SavePotentialField, NWriteoutProcesses):
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
        Maximum dimensionless wavenumber index used by the generator. This is
        normally set to ``Nmesh/2``.
    Lbox : float
        Side length of the simulation box in comoving Mpc/h.
    FileBase : str
        Base name for the generated initial-condition files.
    OutputDir : str
        Directory in which the generated files are written.
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
        Path to ``BinnedKernel_k1_edges.txt``.
    BinnedK2EdgesPath : str
        Path to ``BinnedKernel_k2_edges.txt``.
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
    BinnedPNGCacheBlockSize : int
        Number of output modes cached. Larger
        values improve speed at the cost of proportionally greater memory use.
    SavePotentialField : int or bool
        Whether to save the Gaussian and non-Gaussian potential fields.
    NWriteoutProcesses : int
        Number of processes used for parallel file output.

    Returns
    -------
    str
        Configuration text formatted using ``DEFAULT_BINNED_CONFIG``.

    Examples
    --------
    >>> cfg = make_config_binned(
    ...     Nmesh = 256, Nsample = 256, Nmax = 128, Lbox = 1000.0,
    ...     FileBase = 'ICs', OutputDir = './output/',
    ...     GlassFile = 'glass256.dat', GlassTileFac = 2,
    ...     seed = 12345, Om = 0.3, Ode = 0.7, h = 0.7,
    ...     sigma8 = 0.8, n_s = 0.965, z_ini = 49.0, Fnl = 100.0,
    ...     TransferPath = 'TransferFunctionConverted.dat',
    ...     BinnedK1EdgesPath = 'BinnedKernel_k1_edges.txt',
    ...     BinnedK2EdgesPath = 'BinnedKernel_k2_edges.txt',
    ...     BinnedK3EdgesPath = 'BinnedKernel_k3_edges.txt',
    ...     BinnedKernelValuesPath = 'BinnedKernel_hybrid_values.txt',
    ...     BinnedPNGFilterCode = 3, BinnedPNGOutputFilterCode = 3,
    ...     BinnedPNGCacheBlockSize = 4,
    ...     SavePotentialField = True, NWriteoutProcesses = 8,
    ... )
    """
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


def camb2input(path):
    """
    Simple util to convert a camb-formatted T(k) file into
    the format expected by 2LPTIc (and therefore by Aarambam).

    Parameters
    ----------
    path : str
        Path to the T(k) file from CAMB

    Returns
    -------
    numpy array: (N, 2)
        2D array where first column is k-values and
        2nd column is the matter transfer function.

    """
    return np.loadtxt(path, usecols = [0, 6])
    

def collate_potential(OutputDir):
    """
    Utility to convert the list of potential grid outputs from
    Aarambam into a single numpy file. The numpy arrays are
    saved into the same directory. The old files are also
    cleaned up.

    Parameters
    ----------
    OutputDir : str
        The directory that these potential files were
        written into
    """

    #Now save the initial conditions that you have generated
    dtype = np.dtype([('ind', 'i4'), ('pot', 'f8')])
    out   = {}
    for t in ['Gauss_potential', 'Nongauss_potential']:
        
        files = sorted(glob.glob(OutputDir + f'/{t}_*'))
        Nmesh = sum([int(f.split('_')[-1]) for f in files])
        arr   = np.zeros(Nmesh**3, dtype = np.float64)
        i     = 0
        #Loop over files and concat
        for filename in tqdm.tqdm(files, desc = f"Collating files for {t}"):
            
            #Don't concate an existing npy file
            if '.npy' in filename: 
                continue 
            
            tmp = np.memmap(filename, offset = 16, dtype = dtype, mode = 'r')
            arr[i:i+tmp.size] = tmp['pot']
            
            i += tmp.size
            del tmp; gc.collect()
            os.remove(filename)

        out[t] = arr.reshape([Nmesh]*3)

    return out



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

    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 128,
    ...                  N_modes = 50, ModeTol = 1e-6, MaxModeCount = 500)
    >>> out = dec.go(Model = SomeBispectrumModel, Basis = BasicBasisDecompose)

    Binned decomposition:

    >>> dec = Decomposer(n_s = 0.965, Lbox = 1000.0, Nmax = 128,
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

        inpars = self.args.copy()
        inpars.update(**kwargs)

        decomp = type('C', (Model, Basis), {})
        instan = decomp(**inpars)
        result = instan.go()

        if outdir is not None:
            assert os.path.exists(outdir), f'Path {outdir} does not exist'
            instan.finalize(result, directory = outdir)

        return result