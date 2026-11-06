/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Same three steps as attn_loops.c (S = QK^T * scale is stored, softmax,
 * O = S V), with both matrix products tiled:
 *
 *   register tiles: a QK_MR x QK_NR tile of S and a PV_MR x PV_NR tile of O are
 *     accumulated in local arrays with compile-time sizes, which clang fully
 *     unrolls into YMM registers, so each 8-wide FMA needs no load/store of
 *     its accumulator;
 *   cache blocks:   QK_JB keys (packed K, QK_JB*d*4 bytes) and PV_JB rows of V
 *     (PV_JB*d*4 bytes) are reused by every query row while they sit in L1/L2.
 *
 * K is packed into panels of QK_NR keys (Kp[panel][k][c]) so that the
 * micro-kernel reads one contiguous QK_NR-float row per k.
 *
 * Every S[i,j] is still summed over k = 0..d-1 and every O[i,k] over
 * j = 0..N-1 in the same order as in naive, so the output is bitwise identical.
 * Tile sizes can be overridden with -D for experiments; the defaults below are
 * the fastest configuration measured on the Ryzen 7 5800X.
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
#ifndef QK_MR
#define QK_MR 4 /* query rows per QK^T register tile */
#endif
#ifndef QK_NR
#define QK_NR 16 /* keys per QK^T register tile (multiple of 8) */
#endif
#ifndef QK_JB
#define QK_JB 256 /* keys per QK^T cache block (multiple of QK_NR) */
#endif
#ifndef PV_MR
#define PV_MR 5 /* query rows per PV register tile */
#endif
#ifndef PV_NR
#define PV_NR 16 /* output columns per PV register tile (multiple of 8) */
#endif
#ifndef PV_JB
#define PV_JB 128 /* rows of V per PV cache block */
#endif

static inline int imin(int a, int b) { return a < b ? a : b; }

/* s[r*N + c] = scale * sum_k q[r*d + k] * kp[k*QK_NR + c]  for r < mr, c < nc */
static inline __attribute__((always_inline)) void qk_tile(const float *q, const float *kp, float *s,
                                                          int N, int d, float scale, const int mr, int nc)
{
    float acc[QK_MR][QK_NR];
    for (int r = 0; r < mr; r++)
        for (int c = 0; c < QK_NR; c++)
            acc[r][c] = 0.0f;
    for (int k = 0; k < d; k++)
        for (int r = 0; r < mr; r++) {
            const float a = q[(size_t)r * d + k];
            for (int c = 0; c < QK_NR; c++)
                acc[r][c] += a * kp[(size_t)k * QK_NR + c];
        }
    if (nc == QK_NR) {
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < QK_NR; c++)
                s[(size_t)r * N + c] = acc[r][c] * scale;
    } else {
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < nc; c++)
                s[(size_t)r * N + c] = acc[r][c] * scale;
    }
}

/* o[r*d + c] (+)= sum_{j < jn} p[r*N + j] * v[j*d + c]  for r < mr, c < PV_NR */
static inline __attribute__((always_inline)) void pv_tile(const float *p, const float *v, float *o, int N,
                                                          int d, int jn, int first, const int mr)
{
    float acc[PV_MR][PV_NR];
    for (int r = 0; r < mr; r++)
        for (int c = 0; c < PV_NR; c++)
            acc[r][c] = first ? 0.0f : o[(size_t)r * d + c];
    for (int j = 0; j < jn; j++)
        for (int r = 0; r < mr; r++) {
            const float a = p[(size_t)r * N + j];
            for (int c = 0; c < PV_NR; c++)
                acc[r][c] += a * v[(size_t)j * d + c];
        }
    for (int r = 0; r < mr; r++)
        for (int c = 0; c < PV_NR; c++)
            o[(size_t)r * d + c] = acc[r][c];
}

/* Columns d - d % PV_NR .. d-1 of O for rows r < mr, one column at a time (same summation order). */
static void pv_cols(const float *p, const float *v, float *o, int N, int d, int jn, int first, int mr,
                    int c0)
{
    for (int r = 0; r < mr; r++)
        for (int c = c0; c < d; c++) {
            float acc = first ? 0.0f : o[(size_t)r * d + c];
            for (int j = 0; j < jn; j++)
                acc += p[(size_t)r * N + j] * v[(size_t)j * d + c];
            o[(size_t)r * d + c] = acc;
        }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int np = (N + QK_NR - 1) / QK_NR; /* number of key panels */
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kp = malloc((size_t)np * d * QK_NR * sizeof(float));

    /* 0. pack K into panels: Kp[p][k][c] = K[p*QK_NR + c][k], zero-padded past N */
    for (int p = 0; p < np; p++)
        for (int k = 0; k < d; k++)
            for (int c = 0; c < QK_NR; c++) {
                const int j = p * QK_NR + c;
                Kp[((size_t)p * d + k) * QK_NR + c] = j < N ? K[(size_t)j * d + k] : 0.0f;
            }

    /* 1. S = Q K^T * scale: for each block of QK_JB keys, sweep all query rows */
    for (int j0 = 0; j0 < N; j0 += QK_JB) {
        const int j1 = imin(j0 + QK_JB, N);
        int i = 0;
        for (; i + QK_MR <= N; i += QK_MR)
            for (int j = j0; j < j1; j += QK_NR)
                qk_tile(Q + (size_t)i * d, Kp + (size_t)(j / QK_NR) * d * QK_NR, S + (size_t)i * N + j, N, d,
                        scale, QK_MR, imin(QK_NR, N - j));
        for (; i < N; i++)
            for (int j = j0; j < j1; j += QK_NR)
                qk_tile(Q + (size_t)i * d, Kp + (size_t)(j / QK_NR) * d * QK_NR, S + (size_t)i * N + j, N, d,
                        scale, 1, imin(QK_NR, N - j));
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

    /* 3. O = S V: for each block of PV_JB rows of V, sweep all query rows */
    const int dn = d - d % PV_NR;
    for (int j0 = 0; j0 < N; j0 += PV_JB) {
        const int jn = imin(PV_JB, N - j0), first = j0 == 0;
        int i = 0;
        for (; i + PV_MR <= N; i += PV_MR) {
            for (int c = 0; c < dn; c += PV_NR)
                pv_tile(S + (size_t)i * N + j0, V + (size_t)j0 * d + c, O + (size_t)i * d + c, N, d, jn, first,
                        PV_MR);
            pv_cols(S + (size_t)i * N + j0, V + (size_t)j0 * d, O + (size_t)i * d, N, d, jn, first, PV_MR, dn);
        }
        for (; i < N; i++) {
            for (int c = 0; c < dn; c += PV_NR)
                pv_tile(S + (size_t)i * N + j0, V + (size_t)j0 * d + c, O + (size_t)i * d + c, N, d, jn, first, 1);
            pv_cols(S + (size_t)i * N + j0, V + (size_t)j0 * d, O + (size_t)i * d, N, d, jn, first, 1, dn);
        }
    }

    free(Kp);
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
