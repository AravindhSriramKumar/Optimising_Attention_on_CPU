/*
 * attn_fused.c -- Q3c: row-wise fusion (the N x N matrix is never stored).
 *
 * The query rows are processed in blocks of FB rows. For each block the three
 * steps run back to back on an FB x N score buffer that stays in cache
 * (L2 at N=1024, L3 at N=4096 for FB=80):
 *
 *   1. scores  P[r,:] = Q[i0+r,:] K^T * scale   (register tiles from attn_tiled.c)
 *   2. softmax of each row of P, in place       (same two-pass code as naive)
 *   3. O[i0+r,:] = P[r,:] V                     (register tiles from attn_tiled.c)
 *
 * Temporary memory is FB*N + N*d floats (score buffer + packed K) instead of
 * the N*N floats of S. Each score and each output element is still summed in
 * naive's order and the softmax is the same code, so the output is bitwise
 * identical to naive. Sizes can be overridden with -D for experiments; the
 * defaults below are the fastest configuration measured on the Ryzen 7 5800X.
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
#ifndef FB
#define FB 80 /* query rows per fused block (score buffer: FB*N floats) */
#endif
#ifndef QK_MR
#define QK_MR 4 /* query rows per QK^T register tile */
#endif
#ifndef QK_NR
#define QK_NR 16 /* keys per QK^T register tile (multiple of 8) */
#endif
#ifndef PV_MR
#define PV_MR 5 /* query rows per PV register tile */
#endif
#ifndef PV_NR
#define PV_NR 16 /* output columns per PV register tile (multiple of 8) */
#endif
#ifndef PV_JB
#define PV_JB 128 /* rows of V per PV cache block, shared by the FB rows of a block */
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
    float *Kp = malloc((size_t)np * d * QK_NR * sizeof(float));
    float *P = malloc((size_t)FB * N * sizeof(float)); /* scores of one block of query rows */

    /* 0. pack K into panels: Kp[p][k][c] = K[p*QK_NR + c][k], zero-padded past N */
    for (int p = 0; p < np; p++)
        for (int k = 0; k < d; k++)
            for (int c = 0; c < QK_NR; c++) {
                const int j = p * QK_NR + c;
                Kp[((size_t)p * d + k) * QK_NR + c] = j < N ? K[(size_t)j * d + k] : 0.0f;
            }

    for (int i0 = 0; i0 < N; i0 += FB) {
        const int rb = imin(FB, N - i0);
        const float *q = Q + (size_t)i0 * d;
        float *o = O + (size_t)i0 * d;

        /* 1. scores of rows i0..i0+rb-1: each key panel is reused by every row of the block */
        for (int j = 0; j < N; j += QK_NR) {
            const float *kp = Kp + (size_t)(j / QK_NR) * d * QK_NR;
            const int nc = imin(QK_NR, N - j);
            int r = 0;
            for (; r + QK_MR <= rb; r += QK_MR)
                qk_tile(q + (size_t)r * d, kp, P + (size_t)r * N + j, N, d, scale, QK_MR, nc);
            for (; r < rb; r++)
                qk_tile(q + (size_t)r * d, kp, P + (size_t)r * N + j, N, d, scale, 1, nc);
        }

        /* 2. softmax of each row of the block (same code as naive) */
        for (int r = 0; r < rb; r++) {
            float *row = P + (size_t)r * N;
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

        /* 3. O rows = P V: each block of PV_JB rows of V is reused by every row of the block */
        const int dn = d - d % PV_NR;
        for (int j0 = 0; j0 < N; j0 += PV_JB) {
            const int jn = imin(PV_JB, N - j0), first = j0 == 0;
            const float *v = V + (size_t)j0 * d;
            int r = 0;
            for (; r + PV_MR <= rb; r += PV_MR) {
                for (int c = 0; c < dn; c += PV_NR)
                    pv_tile(P + (size_t)r * N + j0, v + c, o + (size_t)r * d + c, N, d, jn, first, PV_MR);
                pv_cols(P + (size_t)r * N + j0, v, o + (size_t)r * d, N, d, jn, first, PV_MR, dn);
            }
            for (; r < rb; r++) {
                for (int c = 0; c < dn; c += PV_NR)
                    pv_tile(P + (size_t)r * N + j0, v + c, o + (size_t)r * d + c, N, d, jn, first, 1);
                pv_cols(P + (size_t)r * N + j0, v, o + (size_t)r * d, N, d, jn, first, 1, dn);
            }
        }
    }

    free(P);
    free(Kp);
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
