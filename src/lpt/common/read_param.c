#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "allvars.h"
#include "proto.h"
#include "params.h"

#define MAXLINE 2048

/* Tags shared by every method. Method-specific tags are in src/lpt/<method>/params_<method>.c */
static const struct lpt_param common_params[] = {
  {"Omega",                          &Omega,                         PARAM_DOUBLE, 0, 1},
  {"OmegaLambda",                    &OmegaLambda,                   PARAM_DOUBLE, 0, 1},
  {"OmegaBaryon",                    &OmegaBaryon,                   PARAM_DOUBLE, 0, 1},
  {"OmegaDM_2ndSpecies",             &OmegaDM_2ndSpecies,            PARAM_DOUBLE, 0, 1},
  {"HubbleParam",                    &HubbleParam,                   PARAM_DOUBLE, 0, 1},
  {"ShapeGamma",                     &ShapeGamma,                    PARAM_DOUBLE, 0, 1},
  {"Sigma8",                         &Sigma8,                        PARAM_DOUBLE, 0, 1},
  {"PrimordialIndex",                &PrimordialIndex,               PARAM_DOUBLE, 0, 1},
  {"Box",                            &Box,                           PARAM_DOUBLE, 0, 1},
  {"Redshift",                       &Redshift,                      PARAM_DOUBLE, 0, 1},
  {"Fnl",                            &Fnl,                           PARAM_DOUBLE, 0, 1},
  {"SavePotentialField",             &SavePotentialField,            PARAM_INT,    0, 1},
  {"ModeTables",                     &ModeTables,                    PARAM_INT,    0, 0},
  {"FFTWWisdomDir",                  FFTWWisdomDir,                  PARAM_STRING, sizeof(FFTWWisdomDir), 0},
  {"FFTBackend",                     FFTBackend,                     PARAM_STRING, sizeof(FFTBackend),    0},
  {"NGPrecision",                    NGPrecision,                    PARAM_STRING, sizeof(NGPrecision),   0},
  {"FixedAmplitude",                 &FixedAmplitude,                PARAM_INT,    0, 1},
  {"PhaseFlip",                      &PhaseFlip,                     PARAM_INT,    0, 1},
  {"Nmesh",                          &Nmesh,                         PARAM_INT,    0, 1},
  {"Nsample",                        &Nsample,                       PARAM_INT,    0, 1},
  {"Nmax",                           &Nmax,                          PARAM_INT,    0, 1},
  {"GlassFile",                      GlassFile,                      PARAM_STRING, sizeof(GlassFile), 1},
  {"FileWithInputSpectrum",          FileWithInputSpectrum,          PARAM_STRING, sizeof(FileWithInputSpectrum), 1},
  {"FileWithInputTransfer",          FileWithInputTransfer,          PARAM_STRING, sizeof(FileWithInputTransfer), 1},
  {"GlassTileFac",                   &GlassTileFac,                  PARAM_INT,    0, 1},
  {"Seed",                           &Seed,                          PARAM_INT,    0, 1},
  {"SphereMode",                     &SphereMode,                    PARAM_INT,    0, 1},
  {"NumFilesWrittenInParallel",      &NumFilesWrittenInParallel,     PARAM_INT,    0, 1},
  {"OutputDir",                      OutputDir,                      PARAM_STRING, sizeof(OutputDir), 1},
  {"FileBase",                       FileBase,                       PARAM_STRING, sizeof(FileBase), 1},
  {"WhichSpectrum",                  &WhichSpectrum,                 PARAM_INT,    0, 1},
  {"WhichTransfer",                  &WhichTransfer,                 PARAM_INT,    0, 1},
  {"UnitVelocity_in_cm_per_s",       &UnitVelocity_in_cm_per_s,      PARAM_DOUBLE, 0, 1},
  {"UnitLength_in_cm",               &UnitLength_in_cm,              PARAM_DOUBLE, 0, 1},
  {"UnitMass_in_g",                  &UnitMass_in_g,                 PARAM_DOUBLE, 0, 1},
  {"InputSpectrum_UnitLength_in_cm", &InputSpectrum_UnitLength_in_cm, PARAM_DOUBLE, 0, 1},
  {"WDM_On",                         &WDM_On,                        PARAM_INT,    0, 1},
  {"WDM_Vtherm_On",                  &WDM_Vtherm_On,                 PARAM_INT,    0, 1},
  {"WDM_PartMass_in_kev",            &WDM_PartMass_in_kev,           PARAM_DOUBLE, 0, 1},
};
static const int common_nparams = sizeof(common_params) / sizeof(common_params[0]);


static const struct lpt_param *param_entry(int i)
{
  return (i < common_nparams) ? &common_params[i] : &lpt_method_params[i - common_nparams];
}


/* Reads the parameter file. Called on task 0 only; broadcast_parameters() then sends every tag to all tasks. */
void read_parameterfile(char *fname)
{
  FILE *fd;
  char buf[MAXLINE], buf1[MAXLINE], buf2[MAXLINE], buf3[MAXLINE];
  int i, j, nt = common_nparams + lpt_method_nparams;
  int errorFlag = 0;
  char *seen = calloc(nt, 1);

  ModeTables = 0;
  strcpy(FFTWWisdomDir, "auto");
  strcpy(FFTBackend, "fftw");
  strcpy(NGPrecision, "double");
  lpt_method_param_defaults();

  if((fd = fopen(fname, "r")))
    {
      while(fgets(buf, sizeof(buf), fd))
        {
          if(strlen(buf) == sizeof(buf) - 1 && buf[sizeof(buf) - 2] != '\n')
            {
              fprintf(stdout, "Error in file %s: line longer than %d characters.\n", fname, MAXLINE - 1);
              errorFlag = 1;
              break;
            }

          if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
            continue;

          if(buf1[0] == '%' || buf1[0] == '#')     /* comment line */
            continue;

          for(i = 0, j = -1; i < nt; i++)
            if(!seen[i] && strcmp(buf1, param_entry(i)->tag) == 0)
              {
                j = i;
                seen[i] = 1;
                break;
              }

          if(j >= 0)
            {
              const struct lpt_param *p = param_entry(j);
              switch (p->type)
                {
                case PARAM_DOUBLE:
                  *((double *) p->addr) = atof(buf2);
                  break;
                case PARAM_STRING:
                  if(strlen(buf2) >= p->size)
                    {
                      fprintf(stdout, "Error in file %s: value of '%s' longer than %d characters.\n",
                              fname, buf1, (int) p->size - 1);
                      errorFlag = 1;
                    }
                  else
                    strcpy(p->addr, buf2);
                  break;
                case PARAM_INT:
                  *((int *) p->addr) = atoi(buf2);
                  break;
                }
            }
          else
            {
              fprintf(stdout, "Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
              errorFlag = 1;
            }
        }
      fclose(fd);
    }
  else
    {
      fprintf(stdout, "Parameter file %s not found.\n", fname);
      errorFlag = 1;
    }

  for(i = 0; i < nt; i++)
    if(!seen[i] && param_entry(i)->required)
      {
        fprintf(stdout, "Error. I miss a value for tag '%s' in parameter file '%s'.\n", param_entry(i)->tag, fname);
        errorFlag = 1;
      }
  free(seen);

  if(lpt_method_validate(fname))
    errorFlag = 1;

  if(ModeTables < 0 || ModeTables > 1)
    {
      fprintf(stdout, "Error. ModeTables must be 0 (compute per-mode quantities on the fly) or 1 (precompute "
                      "tables) in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  /* the k-space indexing (wavenumbers, Nyquist planes, the seed table) assumes an even Nmesh, and the sampled band
   * must fit on the grid */
  if(Nmesh < 2 || Nmesh % 2)
    {
      fprintf(stdout, "Error. Nmesh must be even (Nmesh = %d) in parameter file '%s'.\n", Nmesh, fname);
      errorFlag = 1;
    }
  if(Nsample < 1 || Nsample > Nmesh)
    {
      fprintf(stdout, "Error. Nsample must be between 1 and Nmesh = %d (Nsample = %d) in parameter file '%s'.\n",
              Nmesh, Nsample, fname);
      errorFlag = 1;
    }
  if(GlassTileFac < 1 || !(Box > 0))
    {
      fprintf(stdout, "Error. GlassTileFac must be at least 1 and Box positive in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }
  if(WDM_On != 0)          /* the displacements use the transfer table alone: the WDM cut-off would only rescale sigma8 */
    {
      fprintf(stdout, "Error. WDM_On is not supported (use a WDM transfer function table instead) in parameter file "
              "'%s'.\n", fname);
      errorFlag = 1;
    }

  if(errorFlag)            /* called on task 0 only: abort all tasks with a non-zero status */
    {
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


/* Sends every parameter of the common and method tables from task 0 to all tasks */
void broadcast_parameters(void)
{
  int i, nt = common_nparams + lpt_method_nparams;

  for(i = 0; i < nt; i++)
    {
      const struct lpt_param *p = param_entry(i);
      switch (p->type)
        {
        case PARAM_DOUBLE:
          MPI_Bcast(p->addr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          break;
        case PARAM_INT:
          MPI_Bcast(p->addr, 1, MPI_INT, 0, MPI_COMM_WORLD);
          break;
        case PARAM_STRING:
          MPI_Bcast(p->addr, (int) p->size, MPI_CHAR, 0, MPI_COMM_WORLD);
          break;
        }
    }
}

#undef MAXLINE
