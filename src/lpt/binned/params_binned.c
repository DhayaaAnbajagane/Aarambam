#include <stdio.h>
#include "allvars.h"
#include "params.h"
#include "binned_vars.h"

char FileWithBinnedK1Edges[500];
char FileWithBinnedK2Edges[500];
char FileWithBinnedK3Edges[500];
char FileWithBinnedKernelValues[500];
int BinnedPNGFilter;
int BinnedPNGOutputFilter;
int BinnedPNGCoordinate;
int BinnedPNGUseInternalSymmetry;
int BinnedPNGCacheBlockSize;

/* Binned method: kernel tables and filters. The modal-basis tags are accepted but not needed, so that a
 * config written for the Legendre method can be reused. */
const struct lpt_param lpt_method_params[] = {
  {"N_modes",                        &N_modes,                       PARAM_INT,    0, 0},
  {"FileWithAlphaCoeff",             FileWithAlphaCoeff,             PARAM_STRING, sizeof(FileWithAlphaCoeff), 0},
  {"FileWithX0Coeff",                FileWithX0Coeff,                PARAM_STRING, sizeof(FileWithX0Coeff), 0},
  {"FileWithBinnedK1Edges",          FileWithBinnedK1Edges,          PARAM_STRING, sizeof(FileWithBinnedK1Edges), 1},
  {"FileWithBinnedK2Edges",          FileWithBinnedK2Edges,          PARAM_STRING, sizeof(FileWithBinnedK2Edges), 1},
  {"FileWithBinnedK3Edges",          FileWithBinnedK3Edges,          PARAM_STRING, sizeof(FileWithBinnedK3Edges), 1},
  {"FileWithBinnedKernelValues",     FileWithBinnedKernelValues,     PARAM_STRING, sizeof(FileWithBinnedKernelValues), 1},
  {"BinnedPNGFilter",                &BinnedPNGFilter,               PARAM_INT,    0, 1},
  {"BinnedPNGOutputFilter",          &BinnedPNGOutputFilter,         PARAM_INT,    0, 1},
  {"BinnedPNGCoordinate",            &BinnedPNGCoordinate,           PARAM_INT,    0, 1},
  {"BinnedPNGUseInternalSymmetry",   &BinnedPNGUseInternalSymmetry,  PARAM_INT,    0, 1},
  {"BinnedPNGCacheBlockSize",        &BinnedPNGCacheBlockSize,       PARAM_INT,    0, 1},
};
const int lpt_method_nparams = sizeof(lpt_method_params) / sizeof(lpt_method_params[0]);

void lpt_method_param_defaults(void)
{
  /* Invalid defaults: these binned-PNG controls are required in the parameter file. */
  BinnedPNGFilter = 0;
  BinnedPNGOutputFilter = 0;
  BinnedPNGCoordinate = 0;
  BinnedPNGUseInternalSymmetry = 0;
  BinnedPNGCacheBlockSize = 0;
  FileWithBinnedK1Edges[0] = 0;
  FileWithBinnedK2Edges[0] = 0;
  FileWithBinnedK3Edges[0] = 0;
  FileWithBinnedKernelValues[0] = 0;
}

int lpt_method_validate(const char *fname)
{
  int errorFlag = 0;

  if(BinnedPNGFilter < 1 || BinnedPNGFilter > 3)
    {
      fprintf(stdout, "Error. BinnedPNGFilter must be 1=heaviside, 2=linear, or 3=cubic in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(BinnedPNGOutputFilter < 1 || BinnedPNGOutputFilter > 3)
    {
      fprintf(stdout, "Error. BinnedPNGOutputFilter must be 1=heaviside, 2=linear, or 3=cubic in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(BinnedPNGUseInternalSymmetry < 0 || BinnedPNGUseInternalSymmetry > 1)
    {
      fprintf(stdout, "Error. BinnedPNGUseInternalSymmetry must be 0 or 1 in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(BinnedPNGCacheBlockSize < 0)
    {
      fprintf(stdout, "Error. BinnedPNGCacheBlockSize must be >=0 in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(BinnedPNGCoordinate < 1 || BinnedPNGCoordinate > 2)
    {
      fprintf(stdout, "Error. BinnedPNGCoordinate must be 1=k or 2=log(k) in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(FileWithBinnedK1Edges[0] == 0 ||
     FileWithBinnedK2Edges[0] == 0 ||
     FileWithBinnedK3Edges[0] == 0 ||
     FileWithBinnedKernelValues[0] == 0)
    {
      fprintf(stdout, "Error. Missing one or more required binned table file tags: FileWithBinnedK1Edges, FileWithBinnedK2Edges, FileWithBinnedK3Edges, FileWithBinnedKernelValues in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  if(WhichSpectrum != 0)
    {
      fprintf(stdout, "Error. WhichSpectrum must be 0 for the binned method in parameter file '%s'.\n", fname);
      errorFlag = 1;
    }

  return errorFlag;
}
