#include <stdio.h>
#include <stdlib.h>
#include "arrays.h"

/* d1 doubles; stops the program if they cannot be allocated (the callers run inside OpenMP regions) */
double *create_array(int d1){
	double *arr = (double *) malloc(d1 * sizeof(double));
	if (arr == NULL){
		fprintf(stderr, "Insufficient memory for an array of %d doubles\n", d1);
		abort();
	}
	return arr;
}

void free_array(double *arr){
	free(arr);
}
