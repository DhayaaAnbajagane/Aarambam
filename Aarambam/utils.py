import numpy as np, os, sys, gc
import textwrap, glob, tqdm

DEFAULT_CONFIG = """
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

DEFAULT_INPUTS = dict(Nmesh = 256, Nsample = 256, Nmax = 256, Lbox = 512, FileBase = 'Aarambam_ics',
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

def make_config(Nmesh, Nsample, Nmax, Lbox, FileBase, OutputDir, GlassFile, GlassTileFac,
                seed, Om, Ode, h, sigma8, n_s, z_ini, Fnl, N_modes, TransferPath,
                AlphaPath, AiPath, SavePotentialField, NWriteoutProcesses):

    return textwrap.dedent(DEFAULT_CONFIG % locals())


def make_example_config():
    return make_config(**DEFAULT_INPUTS)


def camb2input(path):
    return np.loadtxt(path, usecols = [0, 6])
    

def collate_potential(OutputDir):

    #Now save the initial conditions that you have generated
    dtype = np.dtype([('ind', 'i4'), ('pot', 'f8')])
    out   = {}
    for t in ['Gauss_potential', 'Nongauss_potential']:
        
        files = sorted(glob.glob(OutputDir + f'/{t}*'))
        Nmesh = sum([int(f[-1]) for f in files])
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
