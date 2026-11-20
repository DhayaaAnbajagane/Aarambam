#ifndef AARAMBAM_ALLVARS_H
#define AARAMBAM_ALLVARS_H

/* Limits set by 32-bit integers in this code and in the output formats. Overridable with -D (tests use small
 * values). */
#ifndef AARAMBAM_MAX_SLAB_REALS            /* per-rank grid size: TotalSizePlusAdditional and the potential-dump header */
#define AARAMBAM_MAX_SLAB_REALS 2147483647LL
#endif
#ifndef AARAMBAM_MAX_INT_ID                /* particle IDs written as 4-byte ints (NO64BITID) */
#define AARAMBAM_MAX_INT_ID 2147483647LL
#endif
#ifndef AARAMBAM_MAX_BLOCK_BYTES           /* Gadget block markers are 4-byte ints */
#define AARAMBAM_MAX_BLOCK_BYTES 2147483647LL
#endif

#include <fftw3-mpi.h>
#include <stdio.h>
#include <time.h>

/* Grid types. lpt_complex has the memory layout of FFTW3's fftw_complex (two doubles), so grids are handed to
 * FFTW by a cast; the .re / .im members keep the k-space loops readable. */
typedef double lpt_real;
typedef struct
{
  double re, im;
}
lpt_complex;

#define  PI          3.14159265358979323846
#define  GRAVITY     6.672e-8
#define  HUBBLE      3.2407789e-18   /* in h/sec */

double PowerSpec(double kmag);
double GrowthFactor(double astart, double aend);
double F_Omega(double a);
double F2_Omega(double a);
int    read_parameter_file(char *fname);


#ifdef T3E
typedef short int int4byte;     /* Note: int has 8 Bytes on the T3E ! */
typedef unsigned short int uint4byte;   /* Note: int has 8 Bytes on the T3E ! */
#else
typedef int int4byte;
typedef unsigned int uint4byte;
#endif


extern struct io_header_1
{
  uint4byte npart[6];      /*!< npart[1] gives the number of particles in the present file, other particle types are ignored */
  double mass[6];          /*!< mass[1] gives the particle mass */
  double time;             /*!< time (=cosmological scale factor) of snapshot */
  double redshift;         /*!< redshift of snapshot */
  int4byte flag_sfr;       /*!< flags whether star formation is used (not available in L-Gadget2) */
  int4byte flag_feedback;  /*!< flags whether feedback from star formation is included */
  uint4byte npartTotal[6]; /*!< npart[1] gives the total number of particles in the run. If this number exceeds 2^32, the npartTotal[2] stores
                                the result of a division of the particle number by 2^32, while npartTotal[1] holds the remainder. */
  int4byte flag_cooling;   /*!< flags whether radiative cooling is included */
  int4byte num_files;      /*!< determines the number of files that are used for a snapshot */
  double BoxSize;          /*!< Simulation box size (in code units) */
  double Omega0;           /*!< matter density */
  double OmegaLambda;      /*!< vacuum energy density */
  double HubbleParam;      /*!< little 'h' */
 /*  double Fnl;  out of header   !< nongaussian parametrization */
  int4byte flag_stellarage;     /*!< flags whether the age of newly formed stars is recorded and saved */
  int4byte flag_metals;         /*!< flags whether metal enrichment is included */
  int4byte hashtabsize;         /*!< gives the size of the hashtable belonging to this snapshot file */
  char fill[84];                /*!< fills to 256 Bytes */
}
header, header1;


extern double   start_time;          /* MPI_Wtime() at start-up, and at the last print_timed_done */
extern double   previous_time;

extern int      Nglass;
extern int      *Local_nx_table;
extern int      WhichSpectrum;
extern int      WhichTransfer;
extern int      SavePotentialField;
extern int      ModeTables;          /* 0: per-mode quantities on the fly (least memory); 1: precomputed tables (faster) */
extern char     FFTWWisdomDir[500];  /* FFT plan directory: "auto" (cache directory), a path, or "none" */
extern char     FFTBackend[20];      /* FFTs of the non-Gaussian stage: "fftw" (FFTW3-MPI) or "band" (band_fft.c) */
extern char     NGPrecision[20];     /* non-Gaussian stage's fields and FFTs: "double" or "float" (lpt_fft.h) */

extern FILE     *FdTmp, *FdTmpInput;

extern int      Nmesh, Nsample, Nmax;

extern int      SphereMode;

extern long long IDStart;


extern char     GlassFile[500];
extern char     FileWithInputSpectrum[500];
extern char     FileWithInputTransfer[500];
extern char     FileWithAlphaCoeff[500];
extern char     FileWithX0Coeff[500];
extern double   A_res, w_res, phi_res;   /* oscillating primordial P(k), WhichSpectrum 1/2 */

extern int      GlassTileFac;

extern double   Box;
extern int      Seed;
extern int      N_modes;

extern long long TotNumPart;

extern int      NumPart;

extern int      NTaskWithN;


extern int      *Slab_to_task;


extern struct part_data
{
  float Pos[3];
  float Vel[3];
#ifdef  MULTICOMPONENTGLASSFILE
  int   Type;
#endif
  long long ID;
} *P;


extern double InitTime;
extern double Redshift;
extern double MassTable[6];
extern double Fnl;
// ******* FAVN/DSJ ******
extern int FixedAmplitude;
extern int PhaseFlip;
// ******* FAVN/DSJ ******

extern char OutputDir[500], FileBase[500];
extern int  NumFilesWrittenInParallel;


extern int      ThisTask, NTask;

extern int      Local_nx, Local_x_start;

extern int  IdStart;

extern unsigned int TotalSizePlusAdditional;
extern fftw_plan Inverse_plan;       /* c2r, normal (x-slab) input */
extern fftw_plan Forward_plan;       /* r2c, normal (x-slab) output */
extern fftw_plan Forward_plan_t;     /* r2c, transposed (y-slab) output */
extern fftw_plan Transpose_plan;     /* transposed -> normal k-space layout */


extern double UnitTime_in_s, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
extern double InputSpectrum_UnitLength_in_cm;
extern double G, Hubble;
extern double RhoCrit;

extern double Omega, OmegaLambda, OmegaDM_2ndSpecies, Sigma8;
extern double OmegaBaryon, HubbleParam;
extern double PrimordialIndex;
extern double ShapeGamma;
extern double Anorm;

extern double Dplus; /* growth factor where output set by InitTime */
extern double D0;    /* matter/DE domination growth factor normalized to the scale factor at early times */

#ifdef DIFFERENT_TRANSFER_FUNC
extern int Type, MinType, MaxType;
#endif

extern int    WDM_On;
extern int    WDM_Vtherm_On;
extern double WDM_PartMass_in_kev;

#endif
