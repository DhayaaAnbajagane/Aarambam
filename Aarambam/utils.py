import numpy as np, os, sys, gc
import textwrap, glob, tqdm

__all__ = ['DEFAULT_BASIS_CONFIG', 'DEFAULT_BASIS_INPUTS', 'DEFAULT_RESBASIS_CONFIG', 'DEFAULT_RESBASIS_INPUTS',
           'make_config_basis', 'make_example_config_basis', 'make_example_config_resbasis', 
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
                            SavePotentialField = True, NWriteoutProcesses = os.cpu_count()
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


def make_example_config_basis():
    """
    Generate example config for the standard IC generator 
    """
    return make_config_basis(**DEFAULT_BASIS_INPUTS)

def make_example_config_resbasis():
    """
    Generate example config for the P(k) + B(k) IC generator 
    """
    return make_config_resbasis(**DEFAULT_RESBASIS_INPUTS)


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
    """
    Bispectrum modal decomposition driver.

    This class manages the setup and execution of a decomposition by combining
    a physical model (e.g. a bispectrum template) with a modal basis expansion.
    It constructs a hybrid class that inherits from both `Model` and `Basis`,
    initializes it with the supplied arguments, runs the decomposition, and
    optionally writes results to disk.

    Parameters
    ----------
    N_modes : int
        Number of modes in the modal expansion.
    n_s : float
        Scalar spectral index used in basis scaling.
    Lbox : float
        Side length of the simulation box in comoving units (e.g. Mpc/h).
    Nmax : int
        Maximum index of modes to include.
    ModeTol : float
        Tolerance threshold for truncating modes.
    MaxModeCount : int
        Hard cutoff on the number of modes kept.
    **kwargs : dict, optional
        Extra keyword arguments passed through to the combined `Model` and
        `Basis` classes when constructed.

    Methods
    -------
    go(Model, Basis, outdir=None, **kwargs)
        Run the decomposition by instantiating a combined class inheriting
        from both `Model` and `Basis`. Returns the decomposition result, and
        if `outdir` is provided, writes output using the instance’s `finalize`
        method.

    Notes
    -----
    * The `outdir` path must already exist; the code will raise an `AssertionError`
      otherwise.

    Examples
    --------
    >>> dec = Decomposer(N_modes=50, n_s=0.965, Lbox=1000.0,
    ...                  Nmax=30, ModeTol=1e-6, MaxModeCount=500)
    >>> result = dec.go(Model=SomeBispectrumModel, Basis=PolynomialBasis)
    >>> # Optionally save results to disk
    >>> result = dec.go(Model=SomeBispectrumModel, Basis=PolynomialBasis,
    ...                 outdir="./results")
    """

    def __init__(self, N_modes, n_s, Lbox, Nmax, ModeTol, MaxModeCount, **kwargs):
        
        args = dict(N_modes = N_modes, n_s = n_s, Lbox = Lbox, Nmax = Nmax, 
                    ModeTol = ModeTol, MaxModeCount = MaxModeCount)
        args.update(**kwargs)

        self.args = args


    def go(self, Model, Basis, outdir = None, **kwargs):
        """
        Run the modal decomposition for a chosen model and basis.

        Parameters
        ----------
        Model : type
            A class implementing the physical bispectrum/trispectrum model. See the
            ``Aarambam.models`` module for examples.
        Basis : type
            A class implementing the modal basis expansion. This simply uses our modified
            CMB-BEST formalism.
        outdir : str, optional
            Path to an existing directory where results should be written via
            the model’s ``finalize`` method. If None (default), no files are written.
        **kwargs : dict
            Additional keyword arguments to override or supplement stored
            parameters in ``self.args``. This is where all model parameters
            are included.

        Returns
        -------
        result : object
            The output of the combined model-basis instance’s ``go()`` method.
            The exact structure depends on the `Model` and `Basis` implementation.
        """
        inpars = self.args.copy()
        inpars.update(**kwargs)

        decomp = type('C', (Model, Basis), {})
        instan = decomp(**inpars)
        result = instan.go()

        if outdir is not None:
            assert os.path.exists(outdir), f"Path {outdir} does not exist"
            instan.finalize(result, directory = outdir)

        return result

