/* Inner products of the mode bispectra on the tetrapyd grid (the covariance QQ and the projections QS of shapes),
 * parallel over modes with OpenMP. Weights are given per tetrapyd point (cmbbest.pyx). */
#include <omp.h>
#include "arrays.h"
#include "tetrapyd.h"

void compute_mode_bispectra_covariance(double *mode_bispectra_covariance, 
							double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
							int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
							double *mode_evals, int mode_p_max, int k_npts) {
	/* Dimensions:
	 * bispectra_covariance: (n_modes, n_modes) flattened
	 * tetra_weights: 		 (tetra_npts)
	 * tetra_i1, 2 3:		 (tetra_npts)
	 * mode_evaluations:	 (mode_p_max, k_npts) flattened
	 * Note that the flattened arrays are assumed to have contiguous memory allocations.
	 * */
	const int N_K = k_npts;
	const int N_MODES = n_modes;
	
	#pragma omp parallel
	{
		double *Q_vec = create_array(tetra_npts);

		#pragma omp for
		for (int n = 0; n < n_modes; n++) {
			int p1 = mode_p1[n];
			int p2 = mode_p2[n];
			int p3 = mode_p3[n];

			/* Evaluate one of the Q(k1,k2,k3)s first */
			for (int t = 0; t < tetra_npts; t++) {
				int i1 = tetra_i1[t];
				int i2 = tetra_i2[t];
				int i3 = tetra_i3[t];

				double mode_11 = mode_evals[p1 * N_K + i1];
				double mode_12 = mode_evals[p1 * N_K + i2];
				double mode_13 = mode_evals[p1 * N_K + i3];

				double mode_21 = mode_evals[p2 * N_K + i1];
				double mode_22 = mode_evals[p2 * N_K + i2];
				double mode_23 = mode_evals[p2 * N_K + i3];

				double mode_31 = mode_evals[p3 * N_K + i1];
				double mode_32 = mode_evals[p3 * N_K + i2];
				double mode_33 = mode_evals[p3 * N_K + i3];

				double Q_eval = ( mode_11 * mode_22 * mode_33 + mode_11 * mode_23 * mode_32 
								+ mode_12 * mode_21 * mode_33 + mode_12 * mode_23 * mode_31
								+ mode_13 * mode_21 * mode_32 + mode_13 * mode_22 * mode_31) / 6.0;
				
				Q_vec[t] = tetra_weights[t] * Q_eval;
			}

			/* Now loop over the second Q and compute the inner product */
			for (int m = 0; m < n_modes; m++) {
				int q1 = mode_p1[m];
				int q2 = mode_p2[m];
				int q3 = mode_p3[m];

				double sum = 0.0;
				for (int t = 0; t < tetra_npts; t++) {
					int j1 = tetra_i1[t];
					int j2 = tetra_i2[t];
					int j3 = tetra_i3[t];

					double q_11 = mode_evals[q1 * N_K + j1];
					double q_12 = mode_evals[q1 * N_K + j2];
					double q_13 = mode_evals[q1 * N_K + j3];

					double q_21 = mode_evals[q2 * N_K + j1];
					double q_22 = mode_evals[q2 * N_K + j2];
					double q_23 = mode_evals[q2 * N_K + j3];

					double q_31 = mode_evals[q3 * N_K + j1];
					double q_32 = mode_evals[q3 * N_K + j2];
					double q_33 = mode_evals[q3 * N_K + j3];

					double Q2_eval = (q_11 * q_22 * q_33 + q_11 * q_23 * q_32 
									+ q_12 * q_21 * q_33 + q_12 * q_23 * q_31
									+ q_13 * q_21 * q_32 + q_13 * q_22 * q_31) / 6.0;
					
					sum += Q2_eval * Q_vec[t];
				}
				// Below should be thread-safe since n is distinct across threads
				mode_bispectra_covariance[n * N_MODES + m] = sum;
			}
		}

		free_array(Q_vec);
	}

	return;
}

void compute_QS(double *QS, double *S, int n_shapes,
                    double *tetra_weights, int *tetra_i1, int *tetra_i2, int *tetra_i3, int tetra_npts,
                    int *mode_p1, int *mode_p2, int *mode_p3, int n_modes,
                    double *mode_evals, int mode_p_max, int k_npts) {

	/* Dimensions:
	 * QS: (n_shapes, n_modes) flattened
	 # S: (n_shapes, tetra_npts) flattened
	 * tetra_weights: 		 (tetra_npts)
	 * tetra_i1, 2 3:		 (tetra_npts)
	 * mode_evaluations:	 (mode_p_max, k_npts) flattened
	 * Note that the flattened arrays are assumed to have contiguous memory allocations.
	 * */


	const int N_T = tetra_npts;
	const int N_K = k_npts;
	const int N_MODES = n_modes;
	
	#pragma omp parallel
	{
		double *Q_vec = create_array(tetra_npts);

		#pragma omp for
		for (int n = 0; n < n_modes; n++) {
			int p1 = mode_p1[n];
			int p2 = mode_p2[n];
			int p3 = mode_p3[n];

			/* Evaluate the mode bispectra Q_n(k1,k2,k3) */
			for (int t = 0; t < tetra_npts; t++) {
				int i1 = tetra_i1[t];
				int i2 = tetra_i2[t];
				int i3 = tetra_i3[t];

				double mode_11 = mode_evals[p1 * N_K + i1];
				double mode_12 = mode_evals[p1 * N_K + i2];
				double mode_13 = mode_evals[p1 * N_K + i3];

				double mode_21 = mode_evals[p2 * N_K + i1];
				double mode_22 = mode_evals[p2 * N_K + i2];
				double mode_23 = mode_evals[p2 * N_K + i3];

				double mode_31 = mode_evals[p3 * N_K + i1];
				double mode_32 = mode_evals[p3 * N_K + i2];
				double mode_33 = mode_evals[p3 * N_K + i3];

				double Q_eval = ( mode_11 * mode_22 * mode_33 + mode_11 * mode_23 * mode_32 
								+ mode_12 * mode_21 * mode_33 + mode_12 * mode_23 * mode_31
								+ mode_13 * mode_21 * mode_32 + mode_13 * mode_22 * mode_31) / 6.0;
				
				Q_vec[t] = tetra_weights[t] * Q_eval;
			}

			/* Now loop over S and compute the inner product */
			for (int s = 0; s < n_shapes; s++) {

				double sum = 0.0;
				for (int t = 0; t < tetra_npts; t++) {
					sum += S[s * N_T + t] * Q_vec[t];
				}
				// Below should be thread-safe since n is distinct across threads
				QS[s * N_MODES + n] = sum;
			}
		}

		free_array(Q_vec);
	}

	return;

}
