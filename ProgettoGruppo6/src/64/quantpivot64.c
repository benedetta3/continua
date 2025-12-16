#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <float.h>
#include "common.h"

#define USE_ASM_APPROX 1
#define USE_ASM_EUCLIDEAN 1
#define USE_ASM_LOWER_BOUND 1
#define PREFETCH_DIST 16  // Distanza di prefetching

extern double approx_distance_asm(const double* vplus, const double* vminus,
                                  const double* wplus, const double* wminus,
                                  int D);

extern double euclidean_distance_asm(const double* v, const double* w, int D);

extern double compute_lower_bound_asm(const double* idx_v, const double* qpivot, int h);

void* checked_alloc(size_t size) {
    void* p = _mm_malloc(size, align);
    if (!p) {
        printf("ERRORE: impossibile allocare %lu bytes\n", size);
        fflush(stdout);
        exit(1);
    }
    return p;
}

// ==============================
// MAX-HEAP per K-NN
// ==============================

typedef struct {
    int id;
    type dist;
} neighbor;

typedef struct {
    neighbor* heap;
    int size;
    int capacity;
} MaxHeap;

static inline void heap_init(MaxHeap* h, int k) {
    h->heap = (neighbor*)malloc(k * sizeof(neighbor));
    h->size = 0;
    h->capacity = k;
    for(int i = 0; i < k; i++) {
        h->heap[i].id = -1;
        h->heap[i].dist = DBL_MAX;
    }
}

static inline void heap_free(MaxHeap* h) {
    free(h->heap);
}

static inline void heap_swap(neighbor* a, neighbor* b) {
    neighbor temp = *a;
    *a = *b;
    *b = temp;
}

static inline void heap_sift_down(MaxHeap* h, int idx) {
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if(left < h->size && h->heap[left].dist > h->heap[largest].dist)
        largest = left;
    if(right < h->size && h->heap[right].dist > h->heap[largest].dist)
        largest = right;

    if(largest != idx) {
        heap_swap(&h->heap[idx], &h->heap[largest]);
        heap_sift_down(h, largest);
    }
}

static inline void heap_sift_up(MaxHeap* h, int idx) {
    while(idx > 0) {
        int parent = (idx - 1) / 2;
        if(h->heap[parent].dist >= h->heap[idx].dist)
            break;
        heap_swap(&h->heap[parent], &h->heap[idx]);
        idx = parent;
    }
}

static inline int heap_try_insert(MaxHeap* h, int id, type dist) {
    if(h->size < h->capacity) {
        h->heap[h->size].id = id;
        h->heap[h->size].dist = dist;
        heap_sift_up(h, h->size);
        h->size++;
        return 1;
    } else if(dist < h->heap[0].dist) {
        h->heap[0].id = id;
        h->heap[0].dist = dist;
        heap_sift_down(h, 0);
        return 1;
    }
    return 0;
}

static inline type heap_max_dist(MaxHeap* h) {
    return (h->size > 0) ? h->heap[0].dist : DBL_MAX;
}

// ==============================
// QUICKSELECT per Quantizzazione
// ==============================

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

// ==============================
// QUANTIZZAZIONE OTTIMIZZATA AVX con Quickselect
// ==============================

// Versione che riusa buffer scratch (NO malloc/free per ogni chiamata)
static inline void quantize_vector_scratch(const type* v,
                                           type* vplus,
                                           type* vminus,
                                           int x,
                                           int D,
                                           int* indices,
                                           type* abs_vals)
{
    memset(vplus, 0, D * sizeof(type));
    memset(vminus, 0, D * sizeof(type));

    if(x <= 0) return;
    if(x > D) x = D;

    // Calcolo valori assoluti con AVX
    int i = 0;
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    
    for(; i <= D - 4; i += 4) {
        __m256d vals = _mm256_loadu_pd(&v[i]);
        __m256d abs_v = _mm256_andnot_pd(sign_mask, vals);
        _mm256_storeu_pd(&abs_vals[i], abs_v);
        indices[i] = i;
        indices[i+1] = i+1;
        indices[i+2] = i+2;
        indices[i+3] = i+3;
    }
    
    for(; i < D; i++) {
        indices[i] = i;
        abs_vals[i] = fabs(v[i]);
    }
    
    // Quickselect per partizionare i top-x - O(D)
    quickselect_top_x(abs_vals, indices, 0, D - 1, x);
    
    // Insertion Sort SOLO sui primi x elementi - O(x²)
    for(int i = 1; i < x; i++) {
        type key_val = abs_vals[i];
        int key_idx = indices[i];
        int j = i - 1;
        
        while(j >= 0 && abs_vals[j] < key_val) {
            abs_vals[j + 1] = abs_vals[j];
            indices[j + 1] = indices[j];
            j--;
        }
        abs_vals[j + 1] = key_val;
        indices[j + 1] = key_idx;
    }
    
    // Imposta bit
    for(int count = 0; count < x; count++) {
        int idx = indices[count];
        if(v[idx] >= 0) {
            vplus[idx] = 1.0;
        } else {
            vminus[idx] = 1.0;
        }
    }
}

// Versione originale per compatibilità (usata in fit)
void quantize_vector(type* v, type* vplus, type* vminus, int x, int D) {
    int* indices = (int*)malloc(D * sizeof(int));
    type* abs_vals = (type*)_mm_malloc(D * sizeof(type), align);
    
    quantize_vector_scratch(v, vplus, vminus, x, D, indices, abs_vals);
    
    _mm_free(abs_vals);
    free(indices);
}

// ==============================
// DISTANZE OTTIMIZZATE AVX
// ==============================

type approx_distance_c(type* vplus, type* vminus, type* wplus, type* wminus, int D) {
    __m256d sum_pp = _mm256_setzero_pd();
    __m256d sum_mm = _mm256_setzero_pd();
    __m256d sum_pm = _mm256_setzero_pd();
    __m256d sum_mp = _mm256_setzero_pd();
    
    int i = 0;
    for(; i <= D - 4; i += 4) {
        __m256d vp = _mm256_loadu_pd(&vplus[i]);
        __m256d vm = _mm256_loadu_pd(&vminus[i]);
        __m256d wp = _mm256_loadu_pd(&wplus[i]);
        __m256d wm = _mm256_loadu_pd(&wminus[i]);
        
        sum_pp = _mm256_add_pd(sum_pp, _mm256_mul_pd(vp, wp));        
        sum_mm = _mm256_add_pd(sum_mm, _mm256_mul_pd(vm, wm));
        sum_pm = _mm256_add_pd(sum_pm, _mm256_mul_pd(vp, wm));
        sum_mp = _mm256_add_pd(sum_mp, _mm256_mul_pd(vm, wp));
    }
    
    double temp[4] __attribute__((aligned(32)));
    _mm256_store_pd(temp, sum_pp);
    type dot_pp = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm256_store_pd(temp, sum_mm);
    type dot_mm = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm256_store_pd(temp, sum_pm);
    type dot_pm = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm256_store_pd(temp, sum_mp);
    type dot_mp = temp[0] + temp[1] + temp[2] + temp[3];
    
    for(; i < D; i++) {
        dot_pp += vplus[i] * wplus[i];
        dot_mm += vminus[i] * wminus[i];
        dot_pm += vplus[i] * wminus[i];
        dot_mp += vminus[i] * wplus[i];
    }
    
    return dot_pp + dot_mm - dot_pm - dot_mp;
}

double approx_distance(const double* vplus, const double* vminus,
                       const double* wplus, const double* wminus,
                       int D)
{
#ifdef USE_ASM_APPROX
    if (D <= 0) return 0.0;
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance_c((double*)vplus, (double*)vminus, (double*)wplus, (double*)wminus, D);
#endif
}

type euclidean_distance_c(type* v, type* w, int D) {
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    
    int i = 0;
    for(; i <= D - 8; i += 8) {
        __m256d v0 = _mm256_loadu_pd(&v[i]);
        __m256d w0 = _mm256_loadu_pd(&w[i]);
        __m256d v1 = _mm256_loadu_pd(&v[i+4]);
        __m256d w1 = _mm256_loadu_pd(&w[i+4]);
        
        __m256d diff0 = _mm256_sub_pd(v0, w0);
        __m256d diff1 = _mm256_sub_pd(v1, w1);
        
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(diff0, diff0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(diff1, diff1));
    }
    
    __m256d sum = _mm256_add_pd(sum0, sum1);  // combina accumulatori
    
    double temp[4] __attribute__((aligned(32)));
    _mm256_store_pd(temp, sum);
    type result = temp[0] + temp[1] + temp[2] + temp[3];
    
    for(; i < D; i++) {
        type d = v[i] - w[i];
        result += d * d;
    }
    
    return sqrt(result);
}

type euclidean_distance(type* v, type* w, int D) {
#ifdef USE_ASM_EUCLIDEAN
    if (D <= 0) return 0.0;
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance_c(v, w, D);
#endif
}

type compute_lower_bound_c(type* idx_v, type* qpivot, int h) {
    type LB = 0.0;
    
    __m256d max_lb = _mm256_setzero_pd();
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    
    int j = 0;

    for(; j <= h - 8; j += 8) {
        __m256d iv0 = _mm256_loadu_pd(&idx_v[j]);
        __m256d qp0 = _mm256_loadu_pd(&qpivot[j]);
        __m256d iv1 = _mm256_loadu_pd(&idx_v[j+4]);
        __m256d qp1 = _mm256_loadu_pd(&qpivot[j+4]);
        
        __m256d diff0 = _mm256_sub_pd(iv0, qp0);
        __m256d diff1 = _mm256_sub_pd(iv1, qp1);
        
        __m256d abs0 = _mm256_andnot_pd(sign_mask, diff0);
        __m256d abs1 = _mm256_andnot_pd(sign_mask, diff1);
        
        max_lb = _mm256_max_pd(max_lb, abs0);
        max_lb = _mm256_max_pd(max_lb, abs1);
    }

    for(; j <= h - 4; j += 4) {
        __m256d iv = _mm256_loadu_pd(&idx_v[j]);
        __m256d qp = _mm256_loadu_pd(&qpivot[j]);
        __m256d diff = _mm256_sub_pd(iv, qp);
        __m256d abs_diff = _mm256_andnot_pd(sign_mask, diff);
        max_lb = _mm256_max_pd(max_lb, abs_diff);
    }
    
    double temp[4] __attribute__((aligned(32)));
    _mm256_store_pd(temp, max_lb);
    LB = temp[0];
    if(temp[1] > LB) LB = temp[1];
    if(temp[2] > LB) LB = temp[2];
    if(temp[3] > LB) LB = temp[3];
    
    for(; j < h; j++) {
        type diff = fabs(idx_v[j] - qpivot[j]);
        if(diff > LB) LB = diff;
    }
    
    return LB;
}

type compute_lower_bound(type* idx_v, type* qpivot, int h) {
#ifdef USE_ASM_LOWER_BOUND
    if (h <= 0) return 0.0;
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound_c(idx_v, qpivot, h);
#endif
}

// Lower bound con early exit - esce appena LB >= worst_dist
static inline type compute_lower_bound_thresh(const type* idx_v,
                                              const type* qpivot,
                                              int h,
                                              type worst_dist)
{
    type LB = 0.0;
    for (int j = 0; j < h; j++) {
        type diff = fabs(idx_v[j] - qpivot[j]);
        if (diff > LB) {
            LB = diff;
            if (LB >= worst_dist) return LB; // EARLY EXIT: pruning garantito
        }
    }
    return LB;
}

// ==============================
// FIT
// ==============================

void fit(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in fit() OTTIMIZZATO (AVX)\n");
        fflush(stdout);
    }
    
    #ifndef USE_ASM_APPROX
        printf("[DEBUG] approx_distance: versione C baseline AVX attiva\n");
    #else
        printf("[DEBUG] approx_distance: versione ASM AVX attiva (unrolling x4)\n");
    #endif
    
    #ifndef USE_ASM_EUCLIDEAN
        printf("[DEBUG] euclidean_distance: versione C baseline AVX attiva\n");
    #else
        printf("[DEBUG] euclidean_distance: versione ASM AVX attiva (unrolling x4)\n");
    #endif
    
    #ifndef USE_ASM_LOWER_BOUND
        printf("[DEBUG] lower_bound: versione C AVX attiva\n");
    #else
        printf("[DEBUG] lower_bound: versione ASM AVX attiva\n");
    #endif
    
    printf("[OPTIMIZATION] Quantizzazione: Quickselect O(D)\n");
    printf("[OPTIMIZATION] KNN Search: Max-Heap O(log k)\n");
    
    fflush(stdout);

    if (input->first_fit_call == false) {
        if(!input->silent) printf("DEBUG: Prima chiamata a fit(), inizializzo puntatori...\n");
        input->P = NULL;
        input->ds_plus = NULL;
        input->ds_minus = NULL;
        input->index = NULL;
        input->first_fit_call = true;
    }

    int N = input->N;
    int D = input->D;
    int h = input->h;
    int x = input->x;

    if(!input->silent) {
        printf("FIT PARAMS: N=%d, D=%d, h=%d, x=%d\n", N, D, h, x);
        fflush(stdout);
    }

    if(input->DS == NULL){
        printf("ERRORE: input->DS è NULL! Abort.\n");
        exit(1);
    }

    if(input->P != NULL){
        if(!input->silent) printf("DEBUG: libero P precedente...\n");
        _mm_free(input->P);
    }

    input->P = checked_alloc(h * sizeof(int));
    if(!input->silent) printf("DEBUG: P allocato = %p\n", input->P);

    int step = N / h;
    for(int j = 0; j < h; j++){
        input->P[j] = j * step;
    }

    if(!input->silent) printf("DEBUG: Pivot generati correttamente.\n");

    if(input->ds_plus != NULL){
        if(!input->silent) printf("DEBUG: libero ds_plus precedente...\n");
        _mm_free(input->ds_plus);
    }
    if(input->ds_minus != NULL){
        if(!input->silent) printf("DEBUG: libero ds_minus precedente...\n");
        _mm_free(input->ds_minus);
    }

    input->ds_plus = checked_alloc(N * D * sizeof(type));
    input->ds_minus = checked_alloc(N * D * sizeof(type));

    if(!input->silent) printf("DEBUG: Allocati ds_plus=%p, ds_minus=%p\n", input->ds_plus, input->ds_minus);

    // Scratch buffer riusato (evita malloc per ogni vettore)
    int*  scratch_idx = (int*)malloc(D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc(D * sizeof(type));

    for(int i = 0; i < N; i++){
        if(!input->silent && i % 500 == 0){
            printf("DEBUG: Quantizzo DS[%d/%d]\n", i, N);
            fflush(stdout);
        }

        quantize_vector_scratch(&input->DS[i * D],
                                &input->ds_plus[i * D],
                                &input->ds_minus[i * D],
                                x, D,
                                scratch_idx, scratch_abs);

        if (i + 4 < N) {
            __builtin_prefetch(&input->DS[(i+4) * D], 0, 1);
            __builtin_prefetch(&input->ds_plus[(i+4) * D], 1, 1);
            __builtin_prefetch(&input->ds_minus[(i+4) * D], 1, 1);
        }
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    if(!input->silent) printf("DEBUG: Quantizzazione dataset completata.\n");

    if(input->index != NULL){
        if(!input->silent) printf("DEBUG: libero index precedente...\n");
        _mm_free(input->index);
    }

    input->index = checked_alloc(N * h * sizeof(type));
    if(!input->silent) printf("DEBUG: index allocato = %p\n", input->index);

    for(int i = 0; i < N; i++){
        if(!input->silent && i % 500 == 0){
            printf("DEBUG: costruzione indice [%d/%d]\n", i, N);
            fflush(stdout);
        }

        for(int j = 0; j < h; j++){
            int pivot_idx = input->P[j];

            input->index[i*h + j] = approx_distance(
                &input->ds_plus[i * D],    
                &input->ds_minus[i * D],
                &input->ds_plus[pivot_idx * D], 
                &input->ds_minus[pivot_idx * D],
                D
            );
        }
    }

    if(!input->silent) {
        printf("DEBUG: Index costruito.\n");
        printf("FIT COMPLETATO.\n");
        fflush(stdout);
    }
}

void predict(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in predict() OTTIMIZZATO (AVX + Batch)\n");
        fflush(stdout);
    }

    int nq = input->nq;
    int N  = input->N;
    int D  = input->D;
    int h  = input->h;
    int k  = input->k;
    int x  = input->x;

    if(input->ds_plus == NULL || input->ds_minus == NULL){
        printf("ERRORE: predict() chiamata prima di fit()!\n");
        exit(1);
    }

    if(input->Q == NULL){
        printf("ERRORE: input->Q è NULL!\n");
        exit(1);
    }

    // ============================================================
    // BATCH QUANTIZZAZIONE QUERY (evita alloc per ogni query)
    // ============================================================
    
    MATRIX q_plus  = (type*)checked_alloc(nq * D * sizeof(type));
    MATRIX q_minus = (type*)checked_alloc(nq * D * sizeof(type));

    // Scratch per quantizzazione (riusato per ogni query)
    int*  scratch_idx = (int*)malloc(D * sizeof(int));
    type* scratch_abs = (type*)checked_alloc(D * sizeof(type));

    for(int q = 0; q < nq; q++){
        quantize_vector_scratch(&input->Q[q * D],
                                &q_plus[q * D],
                                &q_minus[q * D],
                                x, D, scratch_idx, scratch_abs);
    }

    free(scratch_idx);
    _mm_free(scratch_abs);

    // ============================================================
    // COPIA PIVOT IN MEMORIA CONTIGUA (cache-friendly)
    // ============================================================
    
    MATRIX pivot_plus  = (type*)checked_alloc(h * D * sizeof(type));
    MATRIX pivot_minus = (type*)checked_alloc(h * D * sizeof(type));

    for(int j = 0; j < h; j++){
        int p = input->P[j];
        memcpy(&pivot_plus[j * D],  &input->ds_plus[p * D],  D * sizeof(type));
        memcpy(&pivot_minus[j * D], &input->ds_minus[p * D], D * sizeof(type));
    }

    // ============================================================
    // STRUTTURA K-NN LINEARE (identica al prof)
    // ============================================================
    
    neighbor* knn = (neighbor*)malloc(k * sizeof(neighbor));
    type* qpivot = (type*)malloc(h * sizeof(type));

    // ============================================================
    // LOOP QUERY
    // ============================================================
    
    for(int q = 0; q < nq; q++){
        
        // Inizializza k-NN a infinito
        for(int i = 0; i < k; i++){
            knn[i].id = -1;
            knn[i].dist = DBL_MAX;
        }

        // Puntatori alla query quantizzata
        type* qplus_q  = &q_plus[q * D];
        type* qminus_q = &q_minus[q * D];

        // Precalcolo distanze query-pivot
        for(int j = 0; j < h; j++){
            qpivot[j] = approx_distance(
                qplus_q, qminus_q,
                &pivot_plus[j * D], &pivot_minus[j * D],
                D
            );
        }

        type worst_dist = DBL_MAX;
        int  worst_idx  = 0;

        // ============================================================
        // SCANSIONE DATASET - LOOP UNROLLING x4 (come versione 32-bit)
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
                type d0 = approx_distance(qplus_q, qminus_q, vplus_v0, vminus_v0, D);
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
                type d1 = approx_distance(qplus_q, qminus_q, vplus_v1, vminus_v1, D);
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
                type d2 = approx_distance(qplus_q, qminus_q, vplus_v2, vminus_v2, D);
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
                type d3 = approx_distance(qplus_q, qminus_q, vplus_v3, vminus_v3, D);
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
            type d_approx = approx_distance(qplus_q, qminus_q, vplus_v, vminus_v, D);
            if (d_approx < worst_dist) {
                knn[worst_idx].id = v;
                knn[worst_idx].dist = d_approx;
                worst_dist = knn[0].dist; worst_idx = 0;
                for (int i = 1; i < k; i++) {
                    if (knn[i].dist > worst_dist) { worst_dist = knn[i].dist; worst_idx = i; }
                }
            }
        }

        // ============================================================
        // RAFFINAMENTO con distanze euclidee
        // ============================================================
        
        type* query_base = &input->Q[q * D];
        for(int i = 0; i < k; i++){
            if(knn[i].id >= 0) {
                knn[i].dist = euclidean_distance(
                    query_base,
                    &input->DS[knn[i].id * D],
                    D
                );
            }
        }

        // ============================================================
        // SALVATAGGIO - ORDINE INTERNO INVARIATO
        // ============================================================
        
        for(int i = 0; i < k; i++){
            input->id_nn[q * k + i]   = knn[i].id;
            input->dist_nn[q * k + i] = knn[i].dist;
        }
    }

    // Cleanup
    free(qpivot);
    free(knn);
    _mm_free(q_plus);
    _mm_free(q_minus);
    _mm_free(pivot_plus);
    _mm_free(pivot_minus);

    if(!input->silent) {
        printf("DEBUG: PREDICT COMPLETATO (Batch + AVX)\n");
        fflush(stdout);
    }
}