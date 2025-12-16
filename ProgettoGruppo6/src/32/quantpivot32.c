#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <float.h>
#include "common.h"

// DEBUG: abilitate funzioni ASM
#define USE_ASM_APPROX 1
#define USE_ASM_EUCLIDEAN 1
#define USE_ASM_LOWER_BOUND 1

extern float approx_distance_asm(const float* vplus, const float* vminus,
                                 const float* wplus, const float* wminus,
                                 int D);

extern float euclidean_distance_asm(const float* v, const float* w, int D);

extern float compute_lower_bound_asm(const float* idx_v, const float* qpivot, int h);

// malloc allineata che controlla NULL
static inline void* checked_alloc(size_t size) {
    void* p = _mm_malloc(size, align);
    if (!p) {
        printf("ERRORE: impossibile allocare %lu bytes\n", (unsigned long)size);
        fflush(stdout);
        exit(1);
    }
    return p;
}

// ============================================================
// QUICKSELECT per Quantizzazione - O(D) in media
// ============================================================

static inline void swap_pair(type* vals, int* indices, int i, int j) {
    type tmp_val = vals[i];
    vals[i] = vals[j];
    vals[j] = tmp_val;
    
    int tmp_idx = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp_idx;
}

static inline int partition(type* vals, int* indices, int left, int right, int pivot_idx) {
    type pivot_value = vals[pivot_idx];
    swap_pair(vals, indices, pivot_idx, right);
    
    int store_idx = left;
    for(int i = left; i < right; i++) {
        if(vals[i] > pivot_value) {
            swap_pair(vals, indices, i, store_idx);
            store_idx++;
        }
    }
    
    swap_pair(vals, indices, store_idx, right);
    return store_idx;
}

static void quickselect_top_x(type* vals, int* indices, int left, int right, int x) {
    if(left >= right || x <= 0) return;
    
    while(left < right) {
        int mid = left + (right - left) / 2;
        if(vals[mid] > vals[left]) swap_pair(vals, indices, left, mid);
        if(vals[right] > vals[left]) swap_pair(vals, indices, left, right);
        if(vals[mid] > vals[right]) swap_pair(vals, indices, mid, right);
        
        int pivot_idx = mid;
        int new_pivot = partition(vals, indices, left, right, pivot_idx);
        
        if(new_pivot == x - 1) {
            return;
        } else if(new_pivot > x - 1) {
            right = new_pivot - 1;
        } else {
            left = new_pivot + 1;
        }
    }
}

// ============================================================
// QUANTIZZAZIONE OTTIMIZZATA SSE con Quickselect O(D)
// ============================================================

static inline void quantize_vector_scratch(const type* v,
                                          type* vplus,
                                          type* vminus,
                                          int x,
                                          int D,
                                          int* indices,
                                          type* abs_vals)
{
    // reset iniziale
    memset(vplus,  0, (size_t)D * sizeof(type));
    memset(vminus, 0, (size_t)D * sizeof(type));

    if (x <= 0) return;
    if (x > D) x = D;

    // Calcolo valori assoluti + indici con SSE
    int i = 0;
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    for (; i <= D - 4; i += 4) {
        __m128 vals  = _mm_loadu_ps((const float*)&v[i]);
        __m128 abs_v = _mm_andnot_ps(sign_mask, vals);
        _mm_storeu_ps((float*)&abs_vals[i], abs_v);

        indices[i]   = i;
        indices[i+1] = i+1;
        indices[i+2] = i+2;
        indices[i+3] = i+3;
    }
    for (; i < D; i++) {
        indices[i]  = i;
        abs_vals[i] = (type)fabsf((float)v[i]);
    }

    // Quickselect per partizionare i top-x - O(D)
    quickselect_top_x(abs_vals, indices, 0, D - 1, x);
    
    // Insertion Sort SOLO sui primi x elementi - O(x²) ma x << D
    for (int j = 1; j < x; j++) {
        type key_val = abs_vals[j];
        int key_idx = indices[j];
        int k = j - 1;
        
        while (k >= 0 && abs_vals[k] < key_val) {
            abs_vals[k + 1] = abs_vals[k];
            indices[k + 1] = indices[k];
            k--;
        }
        abs_vals[k + 1] = key_val;
        indices[k + 1] = key_idx;
    }

    // Imposta i bit nei vettori quantizzati
    for (int count = 0; count < x; count++) {
        int idx = indices[count];
        if (v[idx] >= (type)0) vplus[idx] = (type)1.0f;
        else                  vminus[idx] = (type)1.0f;
    }
}

// ============================================================
// DISTANZE
// ============================================================

static inline type approx_distance_c(type* vplus, type* vminus,
                                    type* wplus, type* wminus, int D)
{
    __m128 sum_pp = _mm_setzero_ps();
    __m128 sum_mm = _mm_setzero_ps();
    __m128 sum_pm = _mm_setzero_ps();
    __m128 sum_mp = _mm_setzero_ps();

    int i = 0;
    for (; i <= D - 4; i += 4) {
        __m128 vp = _mm_loadu_ps((float*)&vplus[i]);
        __m128 vm = _mm_loadu_ps((float*)&vminus[i]);
        __m128 wp = _mm_loadu_ps((float*)&wplus[i]);
        __m128 wm = _mm_loadu_ps((float*)&wminus[i]);

        sum_pp = _mm_add_ps(sum_pp, _mm_mul_ps(vp, wp));
        sum_mm = _mm_add_ps(sum_mm, _mm_mul_ps(vm, wm));
        sum_pm = _mm_add_ps(sum_pm, _mm_mul_ps(vp, wm));
        sum_mp = _mm_add_ps(sum_mp, _mm_mul_ps(vm, wp));
    }

    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, sum_pp);
    type dot_pp = (type)(temp[0] + temp[1] + temp[2] + temp[3]);

    _mm_store_ps(temp, sum_mm);
    type dot_mm = (type)(temp[0] + temp[1] + temp[2] + temp[3]);

    _mm_store_ps(temp, sum_pm);
    type dot_pm = (type)(temp[0] + temp[1] + temp[2] + temp[3]);

    _mm_store_ps(temp, sum_mp);
    type dot_mp = (type)(temp[0] + temp[1] + temp[2] + temp[3]);

    for (; i < D; i++) {
        dot_pp += vplus[i] * wplus[i];
        dot_mm += vminus[i] * wminus[i];
        dot_pm += vplus[i] * wminus[i];
        dot_mp += vminus[i] * wplus[i];
    }

    return dot_pp + dot_mm - dot_pm - dot_mp;
}

static inline float approx_distance(const float* vplus, const float* vminus,
                                   const float* wplus, const float* wminus,
                                   int D)
{
#ifdef USE_ASM_APPROX
    if (D <= 0) return 0.0f;
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return (float)approx_distance_c((type*)vplus, (type*)vminus, (type*)wplus, (type*)wminus, D);
#endif
}

static inline type euclidean_distance_c(type* v, type* w, int D) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i <= D - 4; i += 4) {
        __m128 v_vec = _mm_loadu_ps((float*)&v[i]);
        __m128 w_vec = _mm_loadu_ps((float*)&w[i]);
        __m128 diff  = _mm_sub_ps(v_vec, w_vec);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, sum);
    type result = (type)(temp[0] + temp[1] + temp[2] + temp[3]);

    for (; i < D; i++) {
        type d = v[i] - w[i];
        result += d * d;
    }
    return (type)sqrtf((float)result);
}

static inline type euclidean_distance(type* v, type* w, int D) {
#ifdef USE_ASM_EUCLIDEAN
    if (D <= 0) return 0.0f;
    return (type)euclidean_distance_asm((float*)v, (float*)w, D);
#else
    return euclidean_distance_c(v, w, D);
#endif
}

// ============================================================
// LOWER BOUND
// ============================================================

static inline type compute_lower_bound_c(type* idx_v, type* qpivot, int h) {
    type LB = 0.0f;

    __m128 max_lb = _mm_setzero_ps();
    __m128 sign_mask = _mm_set1_ps(-0.0f);

    int j = 0;
    for (; j <= h - 4; j += 4) {
        __m128 iv = _mm_loadu_ps((float*)&idx_v[j]);
        __m128 qp = _mm_loadu_ps((float*)&qpivot[j]);
        __m128 diff = _mm_sub_ps(iv, qp);
        __m128 abs_diff = _mm_andnot_ps(sign_mask, diff);
        max_lb = _mm_max_ps(max_lb, abs_diff);
    }

    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, max_lb);
    LB = (type)temp[0];
    if (temp[1] > (float)LB) LB = (type)temp[1];
    if (temp[2] > (float)LB) LB = (type)temp[2];
    if (temp[3] > (float)LB) LB = (type)temp[3];

    for (; j < h; j++) {
        type diff = (type)fabsf((float)(idx_v[j] - qpivot[j]));
        if (diff > LB) LB = diff;
    }

    return LB;
}

static inline type compute_lower_bound(type* idx_v, type* qpivot, int h) {
#ifdef USE_ASM_LOWER_BOUND
    if (h <= 0) return 0.0f;
    return (type)compute_lower_bound_asm((float*)idx_v, (float*)qpivot, h);
#else
    return compute_lower_bound_c(idx_v, qpivot, h);
#endif
}

static inline type compute_lower_bound_thresh(const type* idx_v,
                                              const type* qpivot,
                                              int h,
                                              type worst_dist)
{
    type LB = 0.0f;
    for (int j = 0; j < h; j++) {
        type diff = (type)fabsf((float)(idx_v[j] - qpivot[j]));
        if (diff > LB) {
            LB = diff;
            if (LB >= worst_dist) return LB; // EARLY EXIT: pruning garantito
        }
    }
    return LB;
}


// ============================================================
// FIT
// ============================================================

void fit(params* input) {
    if (!input->silent) {
        printf("DEBUG: Entrato in fit() - VERSIONE OTTIMIZZATA (NO MALLOC IN QUANTIZE)\n");
        fflush(stdout);
    }

    #ifdef USE_ASM_APPROX
        if(!input->silent) printf("[DEBUG] approx_distance: ASM attiva\n");
    #else
        if(!input->silent) printf("[DEBUG] approx_distance: C baseline attiva\n");
    #endif

    #ifdef USE_ASM_EUCLIDEAN
        if(!input->silent) printf("[DEBUG] euclidean_distance: ASM attiva\n");
    #else
        if(!input->silent) printf("[DEBUG] euclidean_distance: C baseline attiva\n");
    #endif

    #ifdef USE_ASM_LOWER_BOUND
        if(!input->silent) printf("[DEBUG] lower_bound: ASM attiva\n");
    #else
        if(!input->silent) printf("[DEBUG] lower_bound: C SSE attiva\n");
    #endif

    if(!input->silent) {
        printf("[OTTIMIZZAZIONE] quantize_vector: Quickselect O(D) + scratch riusato\n");
        fflush(stdout);
    }

    // 0. init puntatori prima chiamata
    if (input->first_fit_call == false) {
        if(!input->silent) printf("DEBUG: Prima chiamata a fit(), inizializzo puntatori...\n");
        input->P = NULL;
        input->ds_plus = NULL;
        input->ds_minus = NULL;
        input->index = NULL;
        input->first_fit_call = true;
    }

    // 1. parametri
    int N = input->N;
    int D = input->D;
    int h = input->h;
    int x = input->x;

    if(!input->silent) {
        printf("FIT PARAMS: N=%d, D=%d, h=%d, x=%d\n", N, D, h, x);
        fflush(stdout);
    }

    if (input->DS == NULL) {
        printf("ERRORE: input->DS è NULL! Abort.\n");
        exit(1);
    }

    // 2. selezione pivot
    if (input->P != NULL) {
        if(!input->silent) printf("DEBUG: libero P precedente...\n");
        _mm_free(input->P);
    }

    input->P = checked_alloc((size_t)h * sizeof(int));
    if(!input->silent) printf("DEBUG: P allocato = %p\n", input->P);

    int step = (h > 0) ? (N / h) : 0;
    for (int j = 0; j < h; j++) input->P[j] = j * step;

    if(!input->silent) printf("DEBUG: Pivot generati correttamente.\n");

    // 3. quantizzazione dataset
    if (input->ds_plus != NULL) {
        if(!input->silent) printf("DEBUG: libero ds_plus precedente...\n");
        _mm_free(input->ds_plus);
    }
    if (input->ds_minus != NULL) {
        if(!input->silent) printf("DEBUG: libero ds_minus precedente...\n");
        _mm_free(input->ds_minus);
    }

    input->ds_plus  = checked_alloc((size_t)N * (size_t)D * sizeof(type));
    input->ds_minus = checked_alloc((size_t)N * (size_t)D * sizeof(type));
    if(!input->silent) printf("DEBUG: Allocati ds_plus=%p, ds_minus=%p\n", input->ds_plus, input->ds_minus);

    // scratch riusato (una sola alloc)
    int*  scratch_idx = (int*)malloc((size_t)D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc((size_t)D * sizeof(type));
    if (!scratch_idx || !scratch_abs) {
        printf("ERRORE: scratch alloc fallita.\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        if (!input->silent && (i % 500 == 0)) {
            printf("DEBUG: Quantizzo DS[%d/%d]\n", i, N);
            fflush(stdout);
        }

        quantize_vector_scratch(&input->DS[(size_t)i * (size_t)D],
                                &input->ds_plus[(size_t)i * (size_t)D],
                                &input->ds_minus[(size_t)i * (size_t)D],
                                x, D,
                                scratch_idx, scratch_abs);

        if (i + 4 < N) {
            __builtin_prefetch(&input->DS[(size_t)(i+4) * (size_t)D], 0, 1);
            __builtin_prefetch(&input->ds_plus[(size_t)(i+4) * (size_t)D], 1, 1);
            __builtin_prefetch(&input->ds_minus[(size_t)(i+4) * (size_t)D], 1, 1);
        }
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    if(!input->silent) printf("DEBUG: Quantizzazione dataset completata.\n");

    // 4. costruzione indice
    if (input->index != NULL) {
        if(!input->silent) printf("DEBUG: libero index precedente...\n");
        _mm_free(input->index);
    }

    input->index = checked_alloc((size_t)N * (size_t)h * sizeof(type));
    if(!input->silent) printf("DEBUG: index allocato = %p\n", input->index);

    for (int i = 0; i < N; i++) {
        if (!input->silent && (i % 500 == 0)) {
            printf("DEBUG: costruzione indice [%d/%d]\n", i, N);
            fflush(stdout);
        }

        const type* vplus_i  = &input->ds_plus[(size_t)i * (size_t)D];
        const type* vminus_i = &input->ds_minus[(size_t)i * (size_t)D];
        type* out = &input->index[(size_t)i * (size_t)h];

        for (int j = 0; j < h; j++) {
            int pivot_idx = input->P[j];
            const type* pplus  = &input->ds_plus[(size_t)pivot_idx * (size_t)D];
            const type* pminus = &input->ds_minus[(size_t)pivot_idx * (size_t)D];

            out[j] = (type)approx_distance((const float*)vplus_i, (const float*)vminus_i,
                                           (const float*)pplus,  (const float*)pminus, D);
        }

        if (i + 4 < N) {
            __builtin_prefetch(&input->index[(size_t)(i+4) * (size_t)h], 1, 1);
        }
    }

    if(!input->silent) {
        printf("DEBUG: Index costruito.\n");
        printf("FIT COMPLETATO.\n");
        fflush(stdout);
    }
}

// ============================================================
// PREDICT
// ============================================================

typedef struct {
    int id;
    type dist;
} neighbor;

void predict(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in predict() - VERSIONE BATCH QUERY + PIVOT CONTIGUI\n");
        fflush(stdout);
    }

    int nq = input->nq;
    int N  = input->N;
    int D  = input->D;
    int h  = input->h;
    int k  = input->k;
    int x  = input->x;

    if (input->ds_plus == NULL || input->ds_minus == NULL) {
        printf("ERRORE: predict() chiamata prima di fit()!\n");
        exit(1);
    }
    if (input->Q == NULL) {
        printf("ERRORE: input->Q è NULL!\n");
        exit(1);
    }

    // ============================================================
    // OTTIMIZZAZIONE 1: BATCH QUANTIZATION - pre-quantizza TUTTE le query
    // ============================================================
    type* all_q_plus  = (type*)checked_alloc((size_t)nq * (size_t)D * sizeof(type));
    type* all_q_minus = (type*)checked_alloc((size_t)nq * (size_t)D * sizeof(type));

    // scratch per quantizzazione (riusato)
    int*  scratch_idx = (int*)malloc((size_t)D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc((size_t)D * sizeof(type));
    if (!scratch_idx || !scratch_abs) {
        printf("ERRORE: scratch alloc fallita (predict).\n");
        exit(1);
    }

    // Pre-quantizza tutte le query in un blocco
    for (int q = 0; q < nq; q++) {
        quantize_vector_scratch(&input->Q[(size_t)q * (size_t)D],
                                &all_q_plus[(size_t)q * (size_t)D],
                                &all_q_minus[(size_t)q * (size_t)D],
                                x, D,
                                scratch_idx, scratch_abs);
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    if(!input->silent) {
        printf("[OPT] Batch quantization: %d query pre-quantizzate\n", nq);
        fflush(stdout);
    }

    // ============================================================
    // OTTIMIZZAZIONE 2: MEMORIA CONTIGUA PER I PIVOT
    // ============================================================
    type* pivot_plus_contig  = (type*)checked_alloc((size_t)h * (size_t)D * sizeof(type));
    type* pivot_minus_contig = (type*)checked_alloc((size_t)h * (size_t)D * sizeof(type));

    for (int j = 0; j < h; j++) {
        int p = input->P[j];
        memcpy(&pivot_plus_contig[(size_t)j * (size_t)D],
               &input->ds_plus[(size_t)p * (size_t)D],
               (size_t)D * sizeof(type));
        memcpy(&pivot_minus_contig[(size_t)j * (size_t)D],
               &input->ds_minus[(size_t)p * (size_t)D],
               (size_t)D * sizeof(type));
    }

    if(!input->silent) {
        printf("[OPT] Pivot contigui: %d pivot copiati in memoria contigua\n", h);
        fflush(stdout);
    }

    neighbor* knn = (neighbor*)malloc((size_t)k * sizeof(neighbor));
    type* qpivot  = (type*)malloc((size_t)h * sizeof(type));
    if (!knn || !qpivot) {
        printf("ERRORE: alloc knn/qpivot fallita.\n");
        exit(1);
    }

    for (int q = 0; q < nq; q++) {
        // Usa query pre-quantizzata
        const type* q_plus  = &all_q_plus[(size_t)q * (size_t)D];
        const type* q_minus = &all_q_minus[(size_t)q * (size_t)D];

        // init kNN
        for (int i = 0; i < k; i++) {
            knn[i].id = -1;
            knn[i].dist = (type)FLT_MAX;
        }

        // precalcolo d~(q, pivot_j) - usa pivot contigui
        for (int j = 0; j < h; j++) {
            const type* pplus  = &pivot_plus_contig[(size_t)j * (size_t)D];
            const type* pminus = &pivot_minus_contig[(size_t)j * (size_t)D];
            qpivot[j] = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                             (const float*)pplus, (const float*)pminus,
                                             D);
        }

        type worst_dist = (type)FLT_MAX;
        int  worst_idx  = 0;

        // ============================================================
        // OTTIMIZZAZIONE 3: LOOP UNROLLING x4 sulla scansione dataset
        // ============================================================
        int v = 0;
        
        // Loop unrolled x4 - processa 4 candidati per iterazione
        for (; v <= N - 4; v += 4) {
            // Prefetch per le prossime 4 iterazioni
            if (v + 12 < N) {
                __builtin_prefetch(&input->index[(size_t)(v+12) * (size_t)h], 0, 1);
                __builtin_prefetch(&input->ds_plus[(size_t)(v+12) * (size_t)D], 0, 1);
            }

            // --- Candidato v+0 ---
            type* idx_v0 = &input->index[(size_t)(v+0) * (size_t)h];
            type LB0 = compute_lower_bound(idx_v0, qpivot, h);
            if (LB0 < worst_dist) {
                type* vplus_v0  = &input->ds_plus[(size_t)(v+0) * (size_t)D];
                type* vminus_v0 = &input->ds_minus[(size_t)(v+0) * (size_t)D];
                type d0 = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                               (const float*)vplus_v0, (const float*)vminus_v0, D);
                if (d0 < worst_dist) {
                    knn[worst_idx].id = v+0;
                    knn[worst_idx].dist = d0;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+1 ---
            type* idx_v1 = &input->index[(size_t)(v+1) * (size_t)h];
            type LB1 = compute_lower_bound(idx_v1, qpivot, h);
            if (LB1 < worst_dist) {
                type* vplus_v1  = &input->ds_plus[(size_t)(v+1) * (size_t)D];
                type* vminus_v1 = &input->ds_minus[(size_t)(v+1) * (size_t)D];
                type d1 = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                               (const float*)vplus_v1, (const float*)vminus_v1, D);
                if (d1 < worst_dist) {
                    knn[worst_idx].id = v+1;
                    knn[worst_idx].dist = d1;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+2 ---
            type* idx_v2 = &input->index[(size_t)(v+2) * (size_t)h];
            type LB2 = compute_lower_bound(idx_v2, qpivot, h);
            if (LB2 < worst_dist) {
                type* vplus_v2  = &input->ds_plus[(size_t)(v+2) * (size_t)D];
                type* vminus_v2 = &input->ds_minus[(size_t)(v+2) * (size_t)D];
                type d2 = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                               (const float*)vplus_v2, (const float*)vminus_v2, D);
                if (d2 < worst_dist) {
                    knn[worst_idx].id = v+2;
                    knn[worst_idx].dist = d2;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }

            // --- Candidato v+3 ---
            type* idx_v3 = &input->index[(size_t)(v+3) * (size_t)h];
            type LB3 = compute_lower_bound(idx_v3, qpivot, h);
            if (LB3 < worst_dist) {
                type* vplus_v3  = &input->ds_plus[(size_t)(v+3) * (size_t)D];
                type* vminus_v3 = &input->ds_minus[(size_t)(v+3) * (size_t)D];
                type d3 = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                               (const float*)vplus_v3, (const float*)vminus_v3, D);
                if (d3 < worst_dist) {
                    knn[worst_idx].id = v+3;
                    knn[worst_idx].dist = d3;
                    worst_dist = knn[0].dist; worst_idx = 0;
                    for (int i = 1; i < k; i++) {
                        if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                    }
                }
            }
        }

        // Coda: elementi rimanenti (N non multiplo di 4)
        for (; v < N; v++) {
            type* idx_v = &input->index[(size_t)v * (size_t)h];
            type LB = compute_lower_bound(idx_v, qpivot, h);
            if (LB >= worst_dist) continue;

            type* vplus_v  = &input->ds_plus[(size_t)v * (size_t)D];
            type* vminus_v = &input->ds_minus[(size_t)v * (size_t)D];
            type d_approx = (type)approx_distance((const float*)q_plus, (const float*)q_minus,
                                                 (const float*)vplus_v, (const float*)vminus_v, D);
            if (d_approx < worst_dist) {
                knn[worst_idx].id = v;
                knn[worst_idx].dist = d_approx;
                worst_dist = knn[0].dist; worst_idx = 0;
                for (int i = 1; i < k; i++) {
                    if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                }
            }
        }
        type* query_base = &input->Q[(size_t)q * (size_t)D];
        for (int i = 0; i < k; i++) {
            if (knn[i].id >= 0) {
                knn[i].dist = euclidean_distance(query_base,
                                                &input->DS[(size_t)knn[i].id * (size_t)D],
                                                D);
            }
        }

        // salvataggio risultati (stesso ordine interno)
        for (int i = 0; i < k; i++) {
            input->id_nn[(size_t)q * (size_t)k + (size_t)i]   = knn[i].id;
            input->dist_nn[(size_t)q * (size_t)k + (size_t)i] = knn[i].dist;
        }
    }

    free(qpivot);
    free(knn);

    _mm_free(pivot_plus_contig);
    _mm_free(pivot_minus_contig);

    _mm_free(all_q_plus);
    _mm_free(all_q_minus);

    if(!input->silent) {
        printf("DEBUG: PREDICT COMPLETATO (batch query + pivot contigui)\n");
        fflush(stdout);
    }
}
