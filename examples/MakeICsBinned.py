import Aarambam as Am
import numpy as np, subprocess as sp, os, shutil, tempfile


if __name__ == '__main__':
    outdir = os.environ.get('TMPDIR', tempfile.gettempdir()) + '/'
    print(f"USING {outdir} AS BASE FOR CALCULATIONS")
    
    #Specify some defaults that get used everywhere
    #Cosmology matches what we used for the example CAMB transfer function
    n_s     = 0.9660499
    OmegaM  = 0.3175
    OmegaDE = 1 - OmegaM
    h       = 0.6711
    sigma8  = 0.834
    z_ini   = 127
    Fnl     = 100

    Binned_Nk = 20
    Binned_filter = 'hybrid_cubic_k3cubic'
    Lbox    = 1000
    Nmax    = 256
    Nmesh   = 512
    Nsample = 256

    Nprocs  = os.cpu_count()
    
    #Copy the correct transfer function to the working directory
    shutil.copy(os.path.dirname(__file__) + '/../Aarambam/defaults/TransferFunctionConverted.dat', outdir)
    shutil.copy(os.path.dirname(__file__) + '/../Aarambam/defaults/dummy_glass_dmonly_64.dat', outdir)
    
    print("\n\n===================================")
    print("STARTING BASIS DECOMPOSITION")
    print("===================================\n\n")

    #Setup the binned decomposition with some default values
    #  n_s: spectral index. Used to account for mild breaking of scale invariance
    #  Lbox: Sim box in Mpc/h
    #  Nmax: The maximum wavenumber (in units of kF = 2*pi/Lbox) the decomposition goes up to.
    #  Binned_Nk: The number of bins used to decompose the kernel. Play with this a bit.
    #  Binned_filter: How we interpolate over the k1, k2, k3 legs. Default is hybrid_cubic_k3cubic which
    #                 works fine for most target kernels. Using 'heaviside' reverts to no interpolation
    #                 while replacing each 'cubic' with linear provides alternative options.
    Unit  = Am.utils.Decomposer(n_s = n_s, Lbox = Lbox, Nmax = Nmax, Binned_Nk = Binned_Nk, Binned_filter = Binned_filter)

    #Pass in the relevant model and decomposer. Pass any args the model requires.
    #And finally, if you pass in outdir, then the decomposed coeffs will be written out
    #in the format ready for running LPT!
    coeff = Unit.go(Am.models.ScalarI, Am.binned.BasicBinnedDecompose, mass = 1, outdir = outdir)

    #Now make the LPT config file. This command is a utility that takes
    #in the arguments and puts it in the right text format
    CONFIG = Am.utils.make_config_binned(Nmesh = Nmesh, Nsample= Nsample, Nmax = Nmax, Lbox = Lbox, FileBase = 'Aarambam_ics',
                                         OutputDir = outdir, 
                                         GlassFile = outdir + '/dummy_glass_dmonly_64.dat', GlassTileFac = 4,
                                         seed = 42, Om = OmegaM, Ode = OmegaDE, h = h, sigma8 = sigma8, n_s = n_s,
                                         z_ini = z_ini, Fnl = Fnl, 
                                         TransferPath = outdir + '/TransferFunctionConverted.dat',
                                         BinnedK1EdgesPath         = outdir + '/BinnedKernel_k1_edges.txt',
                                         BinnedK3EdgesPath         = outdir + '/BinnedKernel_k3_edges.txt',
                                         BinnedKernelValuesPath    = outdir + '/BinnedKernel_hybrid_values.txt',
                                         BinnedPNGFilterCode       = 3,
                                         BinnedPNGOutputFilterCode = 3,
                                         BinnedPNGCacheBlockSize   = 'auto', #all bins up to Nmesh 512, 4 above
                                         SavePotentialField = int(True), NWriteoutProcesses = Nprocs,
                                         ModeTables = 'auto') #1: per-mode tables (fast, ~1 extra grid); 0: least memory
    
    with open(outdir + '/LPTconfig', 'w') as f: f.write(CONFIG)


    print("\n\n===================================")
    print("STARTING LPT")
    print("===================================\n\n")

    #Now we just run the basis-enhanced LPT! This comes packaged within Aarambam, so
    #you can just use the executable. You can run it from the command line, but
    #let's do it via a subprocess here.
    sp.run(f"mpirun -np {Nprocs} Aarambam-2LPT-Binned {outdir + '/LPTconfig'}", shell = True, env = os.environ)


    #If you asked to save a potential (which we did above, SavePotentialField = True) then
    #let's collate the potentials into one file instead of [Nproc] files
    sp.run(f"Aarambam-collate-potential --file_dir {outdir}", shell = True, env = os.environ)

    #Note that the output ICs will also be spread across files. This follows
    #exactly what 2LPTIc used to do. It is up to the user to decide how they
    #want to handle that organization of the files. We don't pass utils
    #for this as they often require additional dependencies that we don't 
    #want to enforce.