#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// gcc -shared -fPIC -O3 qr.c -o libqr.so -lm

// -------- Math Helpers --------
static inline double pythag(double a, double b) {
    double absa = fabs(a);
    double absb = fabs(b);

    if (absa > absb) {
        if (absa == 0.0) return 0.0; // Only way to get here is if absa and absb are 0
        return absa * sqrt(1.0 + (absb / absa) * (absb / absa));
    } else { // absb >= absa
        if (absb == 0.0) return 0.0; // Both are zero
        return absb * sqrt(1.0 + (absa / absb) * (absa / absb));
    }
}

static inline double sign_transfer(double a, double b) {
    return (b >= 0.0) ? fabs(a) : -fabs(a); // if b>=0 then |a|, b<0 then -|a|
}

// Tridiagonal Matrix Structure:
//
// | m1  e1   0   0   ...      0 |
// | e1  m2  e2   0   ...      0 |
// |  0  e2  m3  e3   ...      0 |
// |  0   0  e3  m4   ...      0 |
// | ... ... ... ...  ... e_{n-1}|
// |  0   0   0  ... e_{n-1} mn |

// -------- Phase 1: Householder Tridiagonalization --------
void tred2(double *A, int n, double *d, double *e) { // matrix A in R(nxn), diagonal d and off diagonal e
    for (int i = n - 1; i > 0; i--) { // we start with last row and go up [0< i <=n-1]
        int l = i - 1; // guard - perform a Hr if there're at least 2 elems in row before diag
        double h = 0.0; // L2(X)^2/l1(X)^2
        double scale = 0.0; // the L1 norm

        // Numerical Stability:
        // We want ||x||^2 = Σ x_k^2, but squaring large x_k risks overflow.
        // To avoid this, compute L1(x) = ||x||_1 = Σ |x_k|,
        // then normalize y_k = x_k / L1(x) so that |y_k| ≤ 1.
        // Next, h = Σ y_k^2 = <x,x>/L_1(x)^2
        // Finally, recover ||x||_2^2 = h * L1(x)^2 = (<x,x>/L_1(x)^2) * L1(x)^2
        // This guarantees stable computation of the L2 norm by bridging through the L1 norm.

        if (l > 0) {
            for (int k = 0; k < i; k++) scale += fabs(A[i * n + k]);
        }

        if (scale == 0.0) { // easy case - store the lower-diagonal elem: A[i][i-1]
            e[i] = A[i * n + l]; // e -is a vector standing for the off-diagonals: upper and lower
        } else { // hard case - do the HAH transformation on the A submatrix
            for (int k = 0; k < i; k++) {
                A[i * n + k] /= scale; // i=const 0=<k<i: A[i][k]/scale = x/L1(x) - we divide each elem of target vec x by L1(x)
                h += A[i * n + k] * A[i * n + k]; // h = ||x/L1(x)||^2 = ||x||^2/L1(x)^2
            }
            double f = A[i * n + l]; // f = A[i][i-1]
            double g = (f >= 0.0 ? -sqrt(h) : sqrt(h)); // sign(A[i][i-1]) - for num stability
            e[i] = scale * g; // e = [...-sign(A[i][i-1])||x_i||...] - off diagonal is full of 
            h -= f * g; // essentially 1/2 * v.T@v - denominator in Householder matrix construction;  h = ||x||^2/L1(x)^2 - sign(A[i][i-1]) * A[i][i-1] * ||x||/L1(x)
            A[i * n + l] = f - g; // v_i-1 = x_i-1 - sign(x_i-1)e_i-1 - overwrite the last element of x to get the Householder vector v
            f = 0.0; // reuse f

            for (int j = 0; j < i; j++) {
                A[j * n + i] = A[i * n + j] / h;
                double g_val = 0.0;
                for (int k = 0; k <= j; k++) g_val += A[j * n + k] * A[i * n + k];
                for (int k = j + 1; k < i; k++) g_val += A[k * n + j] * A[i * n + k];
                e[j] = g_val / h;
                f += e[j] * A[i * n + j];
            }
            double hh = f / (h + h);
            for (int j = 0; j < i; j++) {
                f = A[i * n + j];
                double ej = e[j] - hh * f;
                e[j] = ej;
                for (int k = 0; k <= j; k++) A[j * n + k] -= (f * e[k] + A[i * n + k] * ej);
            }
        }
        d[i] = h;
    }
    d[0] = 0.0; e[0] = 0.0;

    for (int i = 0; i < n; i++) {
        int l = i - 1;
        if (d[i] != 0.0) {
            for (int j = 0; j < l + 1; j++) {
                double g = 0.0;
                for (int k = 0; k < l + 1; k++) g += A[i * n + k] * A[k * n + j];
                for (int k = 0; k < l + 1; k++) A[k * n + j] -= g * A[k * n + i];
            }
        }
        d[i] = A[i * n + i];
        A[i * n + i] = 1.0;
        for (int j = 0; j < i; j++) A[j * n + i] = A[i * n + j] = 0.0;
    }
}

// -------- Phase 2: QL Algorithm --------
void tql2(double *d, double *e, int n, double *V) {
    for (int i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0.0;

    for (int l = 0; l < n; l++) {
        int iter = 0;
        while (1) {
            int m;
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if ((double)(fabs(e[m]) + dd) == dd) break;
            }
            if (m == l) break;
            
            // Note: Standard QL converges very fast (avg 2-3 iters). 
            // If iter > 30, matrix is likely ill-conditioned or logic error.
            if (iter++ == 30) break; 

            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = pythag(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + sign_transfer(r, g));
            double s = 1.0, c = 1.0, p = 0.0;

            for (int i = m - 1; i >= l; i--) {
                double f = s * e[i];
                double b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;
                if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                
                // Accumulate vectors
                for (int k = 0; k < n; k++) {
                    double f_val = V[k * n + (i + 1)];
                    V[k * n + (i + 1)] = s * V[k * n + i] + c * f_val;
                    V[k * n + i] = c * V[k * n + i] - s * f_val;
                }
            }
            if (r == 0.0 && m != l) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

// -------- BRIDGE FUNCTION --------
// This matches the Python ctypes signature exactly.
// It adapts the inputs to the fast algorithms above.
void qr_wilkinson_symmetric(double *A, int n, double tol, int max_iter, double *eigvals, double *V) {
    // 1. Allocate off-diagonal vector
    double *e = (double*)malloc(sizeof(double) * n);
    
    // 2. Prepare workspace for Eigenvectors.
    // The fast algorithm overwrites the matrix with eigenvectors.
    // If user wants vectors (V != NULL), we copy A into V and work on V.
    // If user doesn't want vectors, we malloc a temp buffer to preserve A.
    double *WorkMat;
    int free_work = 0;
    
    if (V != NULL) {
        WorkMat = V;
        // Copy A input to V, because we need the data, but we don't want to destroy A
        memcpy(WorkMat, A, sizeof(double) * n * n);
    } else {
        WorkMat = (double*)malloc(sizeof(double) * n * n);
        memcpy(WorkMat, A, sizeof(double) * n * n);
        free_work = 1;
    }

    // 3. Run Fast Algorithms
    // Phase 1: Reduce to Tridiagonal (WorkMat becomes Q)
    tred2(WorkMat, n, eigvals, e);
    
    // Phase 2: Solve Tridiagonal (WorkMat updates to Eigenvectors, eigvals updates to Eigenvalues)
    // Note: We ignore 'tol' and 'max_iter' because the implicit QL algorithm 
    // is self-adjusting and converges cubically (usually < 5 iters per val).
    tql2(eigvals, e, n, WorkMat);

    // 4. Cleanup
    free(e);
    if (free_work) free(WorkMat);
}