import numpy as np, os, sys, gc
import textwrap, glob, tqdm

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

    return textwrap.dedent(DEFAULT_BASIS_CONFIG % locals())


def make_config_resbasis(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                         seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                         AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses,
                         Pk_A_res, Pk_w_res, Pk_phi_res, Pk_mode):

    return textwrap.dedent(DEFAULT_RESBASIS_CONFIG % locals())


def make_example_config_basis():
    return make_config_basis(**DEFAULT_BASIS_INPUTS)

def make_example_config_resbasis():
    return make_config_resbasis(**DEFAULT_RESBASIS_INPUTS)


def camb2input(path):
    return np.loadtxt(path, usecols = [0, 6])
    

def collate_potential(OutputDir):

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

    def __init__(self, N_modes, n_s, Lbox, Nmax, ModeTol, MaxModeCount, **kwargs):
        
        args = dict(N_modes = N_modes, n_s = n_s, Lbox = Lbox, Nmax = Nmax, 
                    ModeTol = ModeTol, MaxModeCount = MaxModeCount)
        args.update(**kwargs)

        self.args = args


    def go(self, Model, Basis, outdir = None, **kwargs):

        inpars = self.args.copy()
        inpars.update(**kwargs)

        decomp = type('C', (Model, Basis), {})
        instan = decomp(**inpars)
        result = instan.go()

        if outdir is not None:
            assert os.path.exists(outdir), f"Path {outdir} does not exist"
            instan.finalize(result, directory = outdir)

        return result

