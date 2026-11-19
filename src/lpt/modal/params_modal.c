#include <stdio.h>
#include "allvars.h"
#include "params.h"

/* Legendre-basis (modal) method: decomposition tables, plus an optional oscillating primordial P(k)
 * (WhichSpectrum 1 = linear-k, 2 = log-k; amplitude A_res, frequency w_res, phase phi_res). */
const struct lpt_param lpt_method_params[] = {
  {"N_modes",                        &N_modes,                       PARAM_INT,    0, 1},
  {"FileWithAlphaCoeff",             FileWithAlphaCoeff,             PARAM_STRING, sizeof(FileWithAlphaCoeff), 1},
  {"FileWithX0Coeff",                FileWithX0Coeff,                PARAM_STRING, sizeof(FileWithX0Coeff), 1},
  {"A_res",                          &A_res,                         PARAM_DOUBLE, 0, 0},
  {"w_res",                          &w_res,                         PARAM_DOUBLE, 0, 0},
  {"phi_res",                        &phi_res,                       PARAM_DOUBLE, 0, 0},
};
const int lpt_method_nparams = sizeof(lpt_method_params) / sizeof(lpt_method_params[0]);

void lpt_method_param_defaults(void)
{
  A_res = 0;
  w_res = 0;
  phi_res = 0;
}

int lpt_method_validate(const char *fname)
{
  if(WhichSpectrum < 0 || WhichSpectrum > 2)
    {
      fprintf(stdout, "Error. WhichSpectrum must be 0 (no oscillation), 1 (linear-k) or 2 (log-k oscillation) in parameter file '%s'.\n", fname);
      return 1;
    }
  return 0;
}
