#include "allvars.h"
#include "params.h"

/* Legendre-basis method with an oscillating primordial P(k) */
const struct lpt_param lpt_method_params[] = {
  {"N_modes",                        &N_modes,                       PARAM_INT,    0, 1},
  {"FileWithAlphaCoeff",             FileWithAlphaCoeff,             PARAM_STRING, sizeof(FileWithAlphaCoeff), 1},
  {"FileWithX0Coeff",                FileWithX0Coeff,                PARAM_STRING, sizeof(FileWithX0Coeff), 1},
  {"A_res",                          &A_res,                         PARAM_DOUBLE, 0, 1},
  {"w_res",                          &w_res,                         PARAM_DOUBLE, 0, 1},
  {"phi_res",                        &phi_res,                       PARAM_DOUBLE, 0, 1},
};
const int lpt_method_nparams = sizeof(lpt_method_params) / sizeof(lpt_method_params[0]);

void lpt_method_param_defaults(void) {}

int lpt_method_validate(const char *fname) { return 0; }
