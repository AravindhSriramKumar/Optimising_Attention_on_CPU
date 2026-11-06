/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
 *
 * attn_fused.c (row blocks of FB queries; register-tiled QK^T and PV; the N x N
 * matrix is never stored) with a vectorised softmax:
 *
 *   - row max in 8-lane AVX2 registers (exact: max does not depend on order);
 *   - exp(x) computed in-house, 8 lanes at a time: n = round(x*log2(e)),
 *     r = x - n*ln2 (Cody-Waite, two FMAs), exp(r) by a degree-5 polynomial
 *     (Cephes expf coefficients), times 2^n built from the exponent bits;
 *     arguments are always <= 0 (max subtracted), x < -87.3 gives 0;
 *   - the row sum accumulated in vector lanes alongside the exp;
 *   - no normalisation pass: O = (sum_j e_j V_j) * (1/sum) in the PV epilogue.
 *
 * SOFTMAX selects the softmax for the ablation in results/stage6 (-DSOFTMAX=n):
 *   0  naive softmax: scalar max, libm expf, scalar sum, divide (= attn_fused.c)
 *   1  0 + 8-lane max (AVX2)
 *   2  1 + polynomial exp and lane-wise sum (plain C, auto-vectorised), divide
 *   3  2 with a multiply by 1/sum instead of the divide
 *   4  3 with 1/sum folded into the PV epilogue (no normalisation pass)
 *   5  4 with the exp/sum loop written with AVX2 intrinsics
 * The default is the fastest variant measured on the Ryzen 7 5800X.
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
#include <immintrin.h>

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#ifndef SOFTMAX
#define SOFTMAX 5
#endif
#ifndef FB
#define FB 160 /* query rows per fused block (score buffer: FB*N floats) */
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

/* exp(x) for x <= 0, |relative error| of a few ulp (measured in results/stage6). Plain C that clang
   vectorises: the magic-number add rounds to nearest, the memcpy is a bit cast, and each a*b + c is
   contracted to an FMA (fmaf() itself is a libm call that may set errno, which blocks vectorisation). */
static inline float exp_poly(float x)
{
    const float xc = x < -87.3f ? -87.3f : x; /* keeps n in range (no UB in the int conversion); NaN passes */
    const float t = xc * 1.44269504088896341f;
    const float n = (t + 12582912.0f) - 12582912.0f; /* round(t), valid for |t| < 2^22 */
    float r = n * -0.693359375f + xc; /* x - n*ln2, Cody-Waite in two parts */
    r = n * 2.12194440e-4f + r;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * (r * r) + (r + 1.0f);
    const int32_t e = ((int32_t)(n == n ? n : 0.0f) + 127) << 23; /* 2^n (n >= -126); NaN: p is NaN anyway */
    float s;
    memcpy(&s, &e, sizeof s);
    return x < -87.3f ? 0.0f : p * s;
}

/* The same computation on 8 lanes with AVX2 intrinsics (SOFTMAX 5). */
static inline __m256 exp_poly8(__m256 x)
{
    const __m256 xc = _mm256_max_ps(_mm256_set1_ps(-87.3f), x); /* max(lo, x) returns x if x is NaN */
    const __m256 t = _mm256_mul_ps(xc, _mm256_set1_ps(1.44269504088896341f));
    const __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-0.693359375f), xc);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    const __m256 y = _mm256_mul_ps(p, _mm256_castsi256_ps(e));
    return _mm256_andnot_ps(_mm256_cmp_ps(x, _mm256_set1_ps(-87.3f), _CMP_LT_OQ), y);
}

/* max of row[0..N) in 2 x 8 lanes (same result as a sequential max). Intrinsics, because clang 14
   does not vectorise a float max reduction without -ffast-math. */
static inline float row_max(const float *row, int N)
{
    __m256 m0 = _mm256_set1_ps(row[0]), m1 = m0;
    int j = 0;
    for (; j + 16 <= N; j += 16) {
        m0 = _mm256_max_ps(m0, _mm256_loadu_ps(row + j));
        m1 = _mm256_max_ps(m1, _mm256_loadu_ps(row + j + 8));
    }
    float m8[8];
    _mm256_storeu_ps(m8, _mm256_max_ps(m0, m1));
    float m = m8[0];
    for (int l = 1; l < 8; l++)
        m = m8[l] > m ? m8[l] : m;
    for (; j < N; j++)
        m = row[j] > m ? row[j] : m;
    return m;
}

/* row[j] = exp(row[j] - m); returns sum_j row[j] (8-lane partial sums). Not inlined: once inlined
   into attention(), clang 14 no longer vectorises the plain-C loop ("unsafe dependent memory
   operations"); as a separate function it does. */
static __attribute__((noinline)) float row_exp_sum(float *row, int N, float m)
{
#if SOFTMAX == 5
    __m256 s = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    const __m256 mv = _mm256_set1_ps(m);
    int j = 0;
    for (; j + 16 <= N; j += 16) {
        const __m256 e0 = exp_poly8(_mm256_sub_ps(_mm256_loadu_ps(row + j), mv));
        const __m256 e1 = exp_poly8(_mm256_sub_ps(_mm256_loadu_ps(row + j + 8), mv));
        _mm256_storeu_ps(row + j, e0);
        _mm256_storeu_ps(row + j + 8, e1);
        s = _mm256_add_ps(s, e0);
        s1 = _mm256_add_ps(s1, e1);
    }
    float s8[8];
    _mm256_storeu_ps(s8, _mm256_add_ps(s, s1));
    float sum = ((s8[0] + s8[1]) + (s8[2] + s8[3])) + ((s8[4] + s8[5]) + (s8[6] + s8[7]));
    for (; j < N; j++) {
        row[j] = exp_poly(row[j] - m);
        sum += row[j];
    }
    return sum;
#else
    float sum = 0.0f;
#pragma clang loop vectorize(enable) /* lets clang split the sum into vector lanes */
    for (int j = 0; j < N; j++) {
        const float e = exp_poly(row[j] - m);
        row[j] = e;
        sum += e;
    }
    return sum;
#endif
}

/* softmax of the rb rows of P; for SOFTMAX >= 4 the rows are left unnormalised and inv[r] = 1/sum */
static void softmax_rows(float *P, int rb, int N, float *inv)
{
    for (int r = 0; r < rb; r++) {
        float *row = P + (size_t)r * N;
#if SOFTMAX == 0
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
#elif SOFTMAX == 1
        const float max = row_max(row, N);
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
#else
        const float sum = row_exp_sum(row, N, row_max(row, N));
#endif
#if SOFTMAX <= 2
        for (int j = 0; j < N; j++)
            row[j] = row[j] / sum;
        inv[r] = 1.0f;
#elif SOFTMAX == 3
        const float is = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] = row[j] * is;
        inv[r] = 1.0f;
#else
        inv[r] = 1.0f / sum;
#endif
    }
}

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

/* o[r*d + c] (+)= sum_{j < jn} p[r*N + j] * v[j*d + c]  for r < mr, c < PV_NR,
   then o[r*d + c] *= inv[r] if inv != NULL (last block of j) */
static inline __attribute__((always_inline)) void pv_tile(const float *p, const float *v, float *o, int N,
                                                          int d, int jn, int first, const int mr,
                                                          const float *inv)
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
    if (inv) {
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < PV_NR; c++)
                o[(size_t)r * d + c] = acc[r][c] * inv[r];
    } else {
        for (int r = 0; r < mr; r++)
            for (int c = 0; c < PV_NR; c++)
                o[(size_t)r * d + c] = acc[r][c];
    }
}

/* Columns d - d % PV_NR .. d-1 of O for rows r < mr, one column at a time (same summation order). */
static void pv_cols(const float *p, const float *v, float *o, int N, int d, int jn, int first, int mr,
                    int c0, const float *inv)
{
    for (int r = 0; r < mr; r++)
        for (int c = c0; c < d; c++) {
            float acc = first ? 0.0f : o[(size_t)r * d + c];
            for (int j = 0; j < jn; j++)
                acc += p[(size_t)r * N + j] * v[(size_t)j * d + c];
            o[(size_t)r * d + c] = inv ? acc * inv[r] : acc;
        }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int np = (N + QK_NR - 1) / QK_NR; /* number of key panels */
    float *Kp = malloc((size_t)np * d * QK_NR * sizeof(float));
    float *P = malloc((size_t)FB * N * sizeof(float)); /* scores of one block of query rows */
    float inv[FB];                                     /* 1/row sum, applied in the PV epilogue */

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

        /* 2. softmax of each row of the block (see SOFTMAX above) */
        softmax_rows(P, rb, N, inv);

        /* 3. O rows = P V: each block of PV_JB rows of V is reused by every row of the block */
        const int dn = d - d % PV_NR;
        for (int j0 = 0; j0 < N; j0 += PV_JB) {
            const int jn = imin(PV_JB, N - j0), first = j0 == 0;
            const float *sc = (SOFTMAX >= 4 && j0 + jn == N) ? inv : NULL; /* 1/sum on the last block */
            const float *v = V + (size_t)j0 * d;
            int r = 0;
            for (; r + PV_MR <= rb; r += PV_MR) {
                for (int c = 0; c < dn; c += PV_NR)
                    pv_tile(P + (size_t)r * N + j0, v + c, o + (size_t)r * d + c, N, d, jn, first, PV_MR,
                            sc ? sc + r : NULL);
                pv_cols(P + (size_t)r * N + j0, v, o + (size_t)r * d, N, d, jn, first, PV_MR, dn, sc ? sc + r : NULL);
            }
            for (; r < rb; r++) {
                for (int c = 0; c < dn; c += PV_NR)
                    pv_tile(P + (size_t)r * N + j0, v + c, o + (size_t)r * d + c, N, d, jn, first, 1,
                            sc ? sc + r : NULL);
                pv_cols(P + (size_t)r * N + j0, v, o + (size_t)r * d, N, d, jn, first, 1, dn, sc ? sc + r : NULL);
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
