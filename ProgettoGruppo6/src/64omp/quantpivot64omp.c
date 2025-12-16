#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <float.h>
#include <omp.h>
#include "common.h"

#define USE_ASM_APPROX 1
#define USE_ASM_EUCLIDEAN 1
#define USE_ASM_LOWER_BOUND 1

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
    
    // Insertion Sort SOLO sui primi x elementi - O(x^2) ma x << D
    for(int j = 1; j < x; j++) {
        type key_val = abs_vals[j];
        int key_idx = indices[j];
        int k = j - 1;
        
        while(k >= 0 && abs_vals[k] < key_val) {
            abs_vals[k + 1] = abs_vals[k];
            indices[k + 1] = indices[k];
            k--;
        }
        abs_vals[k + 1] = key_val;
        indices[k + 1] = key_idx;
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

// Versione originale per compatibilita
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
        __m256d vp = _mm256_load_pd(&vplus[i]);
        __m256d vm = _mm256_load_pd(&vminus[i]);
        __m256d wp = _mm256_load_pd(&wplus[i]);
        __m256d wm = _mm256_load_pd(&wminus[i]);
        
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
    __m256d sum = _mm256_setzero_pd();
    
    int i = 0;
    for(; i <= D - 4; i += 4) {
        __m256d v_vec = _mm256_loadu_pd(&v[i]);
        __m256d w_vec = _mm256_loadu_pd(&w[i]);
        __m256d diff = _mm256_sub_pd(v_vec, w_vec);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }
    
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

// ==============================
// FIT con OpenMP
// ==============================

void fit(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in fit() OTTIMIZZATO - VERSIONE OPENMP\n");
        fflush(stdout);
    }
    
    #ifndef USE_ASM_APPROX
        printf("[DEBUG] approx_distance: versione C baseline AVX attiva\n");
    #else
        printf("[DEBUG] approx_distance: versione ASM AVX attiva\n");
    #endif
    
    #ifndef USE_ASM_EUCLIDEAN
        printf("[DEBUG] euclidean_distance: versione C baseline AVX attiva\n");
    #else
        printf("[DEBUG] euclidean_distance: versione ASM AVX attiva\n");
    #endif
    
    #ifndef USE_ASM_LOWER_BOUND
        printf("[DEBUG] lower_bound: versione C AVX attiva\n");
    #else
        printf("[DEBUG] lower_bound: versione ASM AVX attiva\n");
    #endif
    
    printf("[OPTIMIZATION] Quantizzazione: Quickselect O(D) + OpenMP\n");
    printf("[OPTIMIZATION] KNN Search: Max-Heap O(log k) + OpenMP\n");
    
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
        printf("OpenMP threads: %d\n", omp_get_max_threads());
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

    input->ds_plus  = checked_alloc(N * D * sizeof(type));
    input->ds_minus = checked_alloc(N * D * sizeof(type));

    if(!input->silent) printf("DEBUG: Allocati ds_plus=%p, ds_minus=%p\n", input->ds_plus, input->ds_minus);

    // PARALLELIZZAZIONE con OpenMP
    #pragma omp parallel for schedule(dynamic, 64)
    for(int i = 0; i < N; i++){
        quantize_vector(&input->DS[i * D],
                        &input->ds_plus[i * D],
                        &input->ds_minus[i * D],
                        x, D);
    }

    if(!input->silent) printf("DEBUG: Quantizzazione dataset completata (OpenMP).\n");

    if(input->index != NULL){
        if(!input->silent) printf("DEBUG: libero index precedente...\n");
        _mm_free(input->index);
    }

    input->index = checked_alloc(N * h * sizeof(type));
    if(!input->silent) printf("DEBUG: index allocato = %p\n", input->index);

    #pragma omp parallel for schedule(dynamic, 64)
    for(int i = 0; i < N; i++){
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
        printf("DEBUG: Index costruito (OpenMP).\n");
        printf("FIT COMPLETATO.\n");
        fflush(stdout);
    }
}

// ==============================

void predict(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in predict() OTTIMIZZATO - VERSIONE OPENMP\n");
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

    MATRIX q_plus  = checked_alloc(nq * D * sizeof(type));
    MATRIX q_minus = checked_alloc(nq * D * sizeof(type));

    #pragma omp parallel for schedule(static)
    for(int q = 0; q < nq; q++){
        quantize_vector(&input->Q[q * D],
                        &q_plus[q * D],
                        &q_minus[q * D],
                        x, D);
    }

    MATRIX pivot_plus  = checked_alloc(h * D * sizeof(type));
    MATRIX pivot_minus = checked_alloc(h * D * sizeof(type));

    for(int j = 0; j < h; j++){
        int p = input->P[j];
        memcpy(&pivot_plus[j * D],  &input->ds_plus[p * D],  D * sizeof(type));
        memcpy(&pivot_minus[j * D], &input->ds_minus[p * D], D * sizeof(type));
    }

    // RICERCA KNN PARALLELIZZATA con Max-Heap
    #pragma omp parallel
    {
        MaxHeap heap;
        heap_init(&heap, k);
        type* qpivot = (type*)malloc(h * sizeof(type));

        #pragma omp for schedule(dynamic, 4)
        for(int q = 0; q < nq; q++){
            // Reset heap per nuova query
            heap.size = 0;
            for(int i = 0; i < k; i++) {
                heap.heap[i].id = -1;
                heap.heap[i].dist = DBL_MAX;
            }

            for(int j = 0; j < h; j++){
                qpivot[j] = approx_distance(
                    &q_plus[q*D], 
                    &q_minus[q*D],
                    &pivot_plus[j*D], 
                    &pivot_minus[j*D],
                    D
                );
            }

            type* qplus_q = &q_plus[q*D];
            type* qminus_q = &q_minus[q*D];
            
            const int PREFETCH_DIST = 16;
            
            for(int v = 0; v < N; v++){
                if(v % PREFETCH_DIST == 0 && v + PREFETCH_DIST < N) {
                    __builtin_prefetch(&input->ds_plus[(v + PREFETCH_DIST) * D], 0, 3);
                    __builtin_prefetch(&input->ds_minus[(v + PREFETCH_DIST) * D], 0, 3);
                    __builtin_prefetch(&input->index[(v + PREFETCH_DIST) * h], 0, 3);
                }

                type worst_dist = heap_max_dist(&heap);
                type* idx_v = &input->index[v*h];
                type LB = compute_lower_bound(idx_v, qpivot, h);

                if(LB >= worst_dist) {
                    continue;
                }

                type* vplus_v = &input->ds_plus[v*D];
                type* vminus_v = &input->ds_minus[v*D];
                
                type d_approx = approx_distance(qplus_q, qminus_q, vplus_v, vminus_v, D);

                heap_try_insert(&heap, v, d_approx);
            }

            // RAFFINAMENTO con distanze euclidee esatte
            type* query_base = &input->Q[q * D];
            
            for(int i = 0; i < heap.size; i++){
                if(heap.heap[i].id >= 0) {
                    heap.heap[i].dist = euclidean_distance(
                        query_base,
                        &input->DS[heap.heap[i].id * D],
                        D
                    );
                }
            }

            // Ri-heapify dopo raffinamento - O(k)
            for(int i = heap.size / 2 - 1; i >= 0; i--) {
                heap_sift_down(&heap, i);
            }

            // HEAP SORT - Estrazione ordinata in O(k log k)
            int original_size = heap.size;

            // Estrai in ordine decrescente (max-heap)
            for(int i = original_size - 1; i > 0; i--) {
                heap_swap(&heap.heap[0], &heap.heap[i]);
                heap.size--;
                heap_sift_down(&heap, 0);
            }

            // Inverti per ordine crescente
            for(int i = 0; i < original_size / 2; i++) {
                heap_swap(&heap.heap[i], &heap.heap[original_size - 1 - i]);
            }

            // Salvataggio risultati
            for(int i = 0; i < k; i++){
                if(i < original_size) {
                    input->id_nn[q*k + i]   = heap.heap[i].id;
                    input->dist_nn[q*k + i] = heap.heap[i].dist;
                } else {
                    input->id_nn[q*k + i]   = -1;
                    input->dist_nn[q*k + i] = DBL_MAX;
                }
            }
        }

        free(qpivot);
        heap_free(&heap);
    }

    _mm_free(q_plus);
    _mm_free(q_minus);
    _mm_free(pivot_plus);
    _mm_free(pivot_minus);

    if(!input->silent) {
        printf("DEBUG: PREDICT COMPLETATO (OpenMP + Max-Heap + Quickselect)\n");
        fflush(stdout);
    }
}