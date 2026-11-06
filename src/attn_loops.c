/*
 * attn_loops.c -- Q3a: loop reordering so the compiler can vectorise.
 *
 * Same three steps as attn_naive.c, with the loops reordered so that the
 * innermost loop of each matrix product walks memory with unit stride and has
 * independent iterations (no reduction), which clang vectorises at -O3:
 *
 *   QK^T: i-j-k (dot product over k)  ->  i-k-j: S[i,:] += Q[i,k] * K^T[k,:]
 *         K is transposed once into Kt (d x N) so that the j loop is unit-stride.
 *   PV:   i-k-j (dot product over j, V read down a column)
 *                                     ->  i-j-k: O[i,:] += P[i,j] * V[j,:]
 *
 * Every S[i,j] and O[i,k] is still accumulated in the same order (k = 0..d-1,
 * j = 0..N-1) as in naive, so the output is bitwise identical to naive.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = malloc((size_t)d * N * sizeof(float));

    /* 0. Kt = K^T (d x N): row k of Kt holds column k of K */
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    /* 1. S = Q K^T * scale, i-k-j order: the j loop is unit-stride in S and Kt */
    for (int i = 0; i < N; i++) {
        float *s = S + (size_t)i * N;
        for (int j = 0; j < N; j++)
            s[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            const float q = Q[(size_t)i * d + k];
            const float *kt = Kt + (size_t)k * N;
            for (int j = 0; j < N; j++)
                s[j] += q * kt[j];
        }
        for (int j = 0; j < N; j++)
            s[j] *= scale;
    }

    /* 2. softmax of each row of S (unchanged from naive) */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
    }

    /* 3. O = S V, i-j-k order: the k loop is unit-stride in O and V */
    for (int i = 0; i < N; i++) {
        float *o = O + (size_t)i * d;
        const float *p = S + (size_t)i * N;
        for (int k = 0; k < d; k++)
            o[k] = 0.0f;
        for (int j = 0; j < N; j++) {
            const float pij = p[j];
            const float *v = V + (size_t)j * d;
            for (int k = 0; k < d; k++)
                o[k] += pij * v[k];
        }
    }

    free(Kt);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
