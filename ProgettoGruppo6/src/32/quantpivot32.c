#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <float.h>
#include "common.h"

#define USE_ASM_APPROX 1
#define USE_ASM_EUCLIDEAN 1
#define USE_ASM_LOWER_BOUND 1

extern float approx_distance_asm(const float* vplus, const float* vminus,
                                 const float* wplus, const float* wminus,
                                 int D);

extern float euclidean_distance_asm(const float* v, const float* w, int D);

extern float compute_lower_bound_asm(const float* idx_v, const float* qpivot, int h);

// malloc allineata che controlla NULL
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
// QUANTIZZAZIONE OTTIMIZZATA
// ==============================

void quantize_vector(type* v, type* vplus, type* vminus, int x, int D) {
    // reset iniziale
    memset(vplus, 0, D * sizeof(type));
    memset(vminus, 0, D * sizeof(type));

    if(x <= 0) return;
    if(x > D) x = D;

    // Array di indici e valori assoluti
    int* indices = (int*)malloc(D * sizeof(int));
    type* abs_vals = (type*)malloc(D * sizeof(type));
    
    // OTTIMIZZAZIONE SSE: calcolo valori assoluti con SIMD
    int i = 0;
    __m128 sign_mask = _mm_set1_ps(-0.0f);  // 0x80000000
    
    // Processa 4 float alla volta
    for(; i <= D - 4; i += 4) {
        __m128 vals = _mm_loadu_ps(&v[i]);
        __m128 abs_v = _mm_andnot_ps(sign_mask, vals);  // azzera bit segno
        _mm_storeu_ps(&abs_vals[i], abs_v);
        indices[i] = i;
        indices[i+1] = i+1;
        indices[i+2] = i+2;
        indices[i+3] = i+3;
    }
    
    // Resto scalare
    for(; i < D; i++) {
        indices[i] = i;
        abs_vals[i] = fabsf(v[i]);
    }
    
    // Selection sort parziale per trovare i primi x massimi
    // (questa parte è difficile da vettorizzare efficacemente)
    for(int count = 0; count < x; count++) {
        int max_idx = count;
        for(int i = count + 1; i < D; i++) {
            if(abs_vals[i] > abs_vals[max_idx]) {
                max_idx = i;
            }
        }
        
        // Swap
        if(max_idx != count) {
            type temp_val = abs_vals[count];
            abs_vals[count] = abs_vals[max_idx];
            abs_vals[max_idx] = temp_val;
            
            int temp_idx = indices[count];
            indices[count] = indices[max_idx];
            indices[max_idx] = temp_idx;
        }
    }
    
    // Imposta i bit nei vettori quantizzati
    for(int count = 0; count < x; count++) {
        int idx = indices[count];
        if(v[idx] >= 0) {
            vplus[idx] = 1.0f;
        } else {
            vminus[idx] = 1.0f;
        }
    }
    
    free(indices);
    free(abs_vals);
}

// ==============================
// DISTANZE OTTIMIZZATE
// ==============================

// Versione SSE baseline per approx_distance
type approx_distance_c(type* vplus, type* vminus, type* wplus, type* wminus, int D) {
    __m128 sum_pp = _mm_setzero_ps();
    __m128 sum_mm = _mm_setzero_ps();
    __m128 sum_pm = _mm_setzero_ps();
    __m128 sum_mp = _mm_setzero_ps();
    
    int i = 0;
    for(; i <= D - 4; i += 4) {
        __m128 vp = _mm_load_ps(&vplus[i]);
        __m128 vm = _mm_load_ps(&vminus[i]);
        __m128 wp = _mm_load_ps(&wplus[i]);
        __m128 wm = _mm_load_ps(&wminus[i]);
        
        sum_pp = _mm_add_ps(sum_pp, _mm_mul_ps(vp, wp));
        sum_mm = _mm_add_ps(sum_mm, _mm_mul_ps(vm, wm));
        sum_pm = _mm_add_ps(sum_pm, _mm_mul_ps(vp, wm));
        sum_mp = _mm_add_ps(sum_mp, _mm_mul_ps(vm, wp));
    }
    
    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, sum_pp);
    type dot_pp = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm_store_ps(temp, sum_mm);
    type dot_mm = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm_store_ps(temp, sum_pm);
    type dot_pm = temp[0] + temp[1] + temp[2] + temp[3];
    
    _mm_store_ps(temp, sum_mp);
    type dot_mp = temp[0] + temp[1] + temp[2] + temp[3];
    
    for(; i < D; i++) {
        dot_pp += vplus[i] * wplus[i];
        dot_mm += vminus[i] * wminus[i];
        dot_pm += vplus[i] * wminus[i];
        dot_mp += vminus[i] * wplus[i];
    }
    
    return dot_pp + dot_mm - dot_pm - dot_mp;
}

float approx_distance(const float* vplus, const float* vminus,
                      const float* wplus, const float* wminus,
                      int D)
{
#ifdef USE_ASM_APPROX
    if (D <= 0) return 0.0f;
    return approx_distance_asm(vplus, vminus, wplus, wminus, D);
#else
    return approx_distance_c(vplus, vminus, wplus, wminus, D);
#endif
}

// Versione SSE baseline per euclidean_distance
type euclidean_distance_c(type* v, type* w, int D) {
    __m128 sum = _mm_setzero_ps();
    
    int i = 0;
    for(; i <= D - 4; i += 4) {
        __m128 v_vec = _mm_loadu_ps(&v[i]);
        __m128 w_vec = _mm_loadu_ps(&w[i]);
        __m128 diff = _mm_sub_ps(v_vec, w_vec);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    
    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, sum);
    type result = temp[0] + temp[1] + temp[2] + temp[3];
    
    for(; i < D; i++) {
        type d = v[i] - w[i];
        result += d * d;
    }
    
    return sqrtf(result);
}

type euclidean_distance(type* v, type* w, int D) {
#ifdef USE_ASM_EUCLIDEAN
    if (D <= 0) return 0.0f;
    return euclidean_distance_asm(v, w, D);
#else
    return euclidean_distance_c(v, w, D);
#endif
}

// ==============================
// LOWER BOUND OTTIMIZZATO
// ==============================

type compute_lower_bound_c(type* idx_v, type* qpivot, int h) {
    type LB = 0.0f;
    
    // Versione SSE manuale
    __m128 max_lb = _mm_setzero_ps();
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    
    int j = 0;
    for(; j <= h - 4; j += 4) {
        __m128 iv = _mm_loadu_ps(&idx_v[j]);
        __m128 qp = _mm_loadu_ps(&qpivot[j]);
        __m128 diff = _mm_sub_ps(iv, qp);
        __m128 abs_diff = _mm_andnot_ps(sign_mask, diff);
        max_lb = _mm_max_ps(max_lb, abs_diff);
    }
    
    // Riduzione orizzontale
    float temp[4] __attribute__((aligned(16)));
    _mm_store_ps(temp, max_lb);
    LB = temp[0];
    if(temp[1] > LB) LB = temp[1];
    if(temp[2] > LB) LB = temp[2];
    if(temp[3] > LB) LB = temp[3];
    
    // Resto scalare
    for(; j < h; j++) {
        type diff = fabsf(idx_v[j] - qpivot[j]);
        if(diff > LB) LB = diff;
    }
    
    return LB;
}

type compute_lower_bound(type* idx_v, type* qpivot, int h) {
#ifdef USE_ASM_LOWER_BOUND
    if (h <= 0) return 0.0f;
    return compute_lower_bound_asm(idx_v, qpivot, h);
#else
    return compute_lower_bound_c(idx_v, qpivot, h);
#endif
}


void fit(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in fit()\n");
        fflush(stdout);
    }
    
    // Debug ottimizzazioni attive
    #ifndef USE_ASM_APPROX
        printf("[DEBUG] approx_distance: versione C baseline attiva\n");
    #else
        printf("[DEBUG] approx_distance: versione ASM attiva (unrolling x4)\n");
    #endif
    
    #ifndef USE_ASM_EUCLIDEAN
        printf("[DEBUG] euclidean_distance: versione C baseline attiva\n");
    #else
        printf("[DEBUG] euclidean_distance: versione ASM attiva (unrolling x4)\n");
    #endif
    
    #ifndef USE_ASM_LOWER_BOUND
        printf("[DEBUG] lower_bound: versione C SSE attiva\n");
    #else
        printf("[DEBUG] lower_bound: versione ASM attiva\n");
    #endif
    
    fflush(stdout);

    // ============================================================
    // 0. INIZIALIZZAZIONE PUNTATORI ALLA PRIMA CHIAMATA
    // ============================================================
    if (input->first_fit_call == false) {
        if(!input->silent) printf("DEBUG: Prima chiamata a fit(), inizializzo puntatori...\n");
        input->P = NULL;
        input->ds_plus = NULL;
        input->ds_minus = NULL;
        input->index = NULL;
        input->first_fit_call = true;
    }

    // ============================================================
    // 1. PARAMETRI
    // ============================================================
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

    // ============================================================
    // 2. SELEZIONE DEI PIVOT
    // ============================================================
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

    // ============================================================
    // 3. QUANTIZZAZIONE COMPLETA DEL DATASET
    // ============================================================
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

    for(int i = 0; i < N; i++){
        if(!input->silent && i % 500 == 0){
            printf("DEBUG: Quantizzo DS[%d/%d]\n", i, N);
            fflush(stdout);
        }

        quantize_vector(&input->DS[i * D],
                        &input->ds_plus[i * D],
                        &input->ds_minus[i * D],
                        x, D);
    }

    if(!input->silent) printf("DEBUG: Quantizzazione dataset completata.\n");

    // ============================================================
    // 4. COSTRUZIONE INDICE
    // ============================================================
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

typedef struct {
    int id;
    type dist;
} neighbor;

void predict(params* input) {
    if(!input->silent) {
        printf("DEBUG: Entrato in predict()\n");
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

    // ====== 1. Quantizzazione delle query ======
    MATRIX q_plus  = checked_alloc(nq * D * sizeof(type));
    MATRIX q_minus = checked_alloc(nq * D * sizeof(type));

    for(int q = 0; q < nq; q++){
        quantize_vector(&input->Q[q * D],
                        &q_plus[q * D],
                        &q_minus[q * D],
                        x, D);
    }

    // ====== 2. Copia pivot quantizzati ======
    MATRIX pivot_plus  = checked_alloc(h * D * sizeof(type));
    MATRIX pivot_minus = checked_alloc(h * D * sizeof(type));

    for(int j = 0; j < h; j++){
        int p = input->P[j];
        memcpy(&pivot_plus[j * D],  &input->ds_plus[p * D],  D * sizeof(type));
        memcpy(&pivot_minus[j * D], &input->ds_minus[p * D], D * sizeof(type));
    }

    // Allocazioni fuori dal loop
    neighbor* knn = (neighbor*)malloc(k * sizeof(neighbor));
    type* qpivot = (type*)malloc(h * sizeof(type));

    // ====== 3. Ricerca KNN per ogni query ======
    for(int q = 0; q < nq; q++){
        // Inizializza k-NN
        for(int i = 0; i < k; i++){
            knn[i].id = -1;
            knn[i].dist = FLT_MAX;
        }

        // Precalcolo d̃(q,p_j) per tutti i pivot
        for(int j = 0; j < h; j++){
            qpivot[j] = approx_distance(
                &q_plus[q*D], 
                &q_minus[q*D],
                &pivot_plus[j*D], 
                &pivot_minus[j*D],
                D
            );
        }

        // === SCANSIONE DATASET CON LOWER BOUND OTTIMIZZATO ===
        type* qplus_q = &q_plus[q*D];
        type* qminus_q = &q_minus[q*D];
        
        for(int v = 0; v < N; v++){
            // 1) Trova peggiore in KNN
            int worst_idx = 0;
            type worst_dist = knn[0].dist;
            for(int i = 1; i < k; i++) {
                if(knn[i].dist > worst_dist) {
                    worst_dist = knn[i].dist;
                    worst_idx = i;
                }
            }
            
            // 2) Calcola Lower bound con funzione ottimizzata
            type* idx_v = &input->index[v*h];
            type LB = compute_lower_bound(idx_v, qpivot, h);

            // 3) Pruning
            if(LB >= worst_dist) {
                continue;
            }

            // 4) Calcolo distanza approssimata
            type* vplus_v = &input->ds_plus[v*D];
            type* vminus_v = &input->ds_minus[v*D];
            
            type d_approx = approx_distance(qplus_q, qminus_q, vplus_v, vminus_v, D);

            // 5) Aggiorna KNN
            if(d_approx < worst_dist) {
                knn[worst_idx].id = v;
                knn[worst_idx].dist = d_approx;
            }
        }

        // === 6. RAFFINAMENTO con distanze euclidee esatte ===
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

        // === 7. Salvataggio risultati ===
        for(int i = 0; i < k; i++){
            input->id_nn[q*k + i]   = knn[i].id;
            input->dist_nn[q*k + i] = knn[i].dist;
        }
    }

    free(qpivot);
    free(knn);

    _mm_free(q_plus);
    _mm_free(q_minus);
    _mm_free(pivot_plus);
    _mm_free(pivot_minus);

    if(!input->silent) {
        printf("DEBUG: PREDICT COMPLETATO\n");
        fflush(stdout);
    }
}