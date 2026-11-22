#ifndef AARAMBAM_PARAMS_H
#define AARAMBAM_PARAMS_H

#include <stddef.h>

/* One parameter-file tag. read_param.c reads and broadcasts every entry of the common table plus the
 * method's table, so a parameter cannot be read on task 0 and then forgotten in the broadcast. */
enum lpt_param_type { PARAM_DOUBLE = 1, PARAM_STRING = 2, PARAM_INT = 3 };

struct lpt_param
{
  const char *tag;
  void *addr;
  int type;        /* lpt_param_type */
  size_t size;     /* capacity of a PARAM_STRING target (including the terminating 0), else 0 */
  int required;    /* 1: missing tag is an error. 0: optional, keeps the value set by lpt_method_param_defaults() */
};

/* Supplied by each method directory (src/lpt/<method>/params_<method>.c) */
extern const struct lpt_param lpt_method_params[];
extern const int lpt_method_nparams;
void lpt_method_param_defaults(void);       /* called on task 0 before the file is read */
int  lpt_method_validate(const char *fname); /* called on task 0 after reading; returns nonzero on error */

#endif
