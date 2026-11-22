/* Inner products of mode bispectra on the tetrapyd grid (tetrapyd.c) */
#ifndef TETRAPYD_H
#define TETRAPYD_H

void compute_mode_bispectra_covariance(double *mode_bispectra_covariance,
							double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
							int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
							double *mode_evals, int mode_p_max, int k_npts);
void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts);

#endif /* TETRAPYD_H */
