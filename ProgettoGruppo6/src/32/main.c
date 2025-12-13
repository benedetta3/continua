#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <xmmintrin.h>

#include <omp.h>

#include "common.h"

#include "quantpivot32.c"

/*
*
* 	load_data
* 	=========
*
*	Legge da file una matrice di N righe
* 	e M colonne e la memorizza in un array lineare in row-major order
*
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero
* 	successivi 4 byte: numero di colonne (M) --> numero intero
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri floating-point a precisione singola
*/
MATRIX load_data(char* filename, int *n, int *k) {
	FILE* fp;
	int rows, cols, status, i;
	
	fp = fopen(filename, "rb");
	
	if (fp == NULL){
		printf("'%s': bad data file name!\n", filename);
		exit(0);
	}
	
	status = fread(&rows, sizeof(int), 1, fp);
	status = fread(&cols, sizeof(int), 1, fp);
	
	MATRIX data = _mm_malloc(rows*cols*sizeof(type), align);
	status = fread(data, sizeof(type), rows*cols, fp);
	fclose(fp);
	
	*n = rows;
	*k = cols;
	
	return data;
}

/*
* 	save_data
* 	=========
* 
*	Salva su file un array lineare in row-major order
*	come matrice di N righe e M colonne
* 
* 	Codifica del file:
* 	primi 4 byte: numero di righe (N) --> numero intero a 32 bit
* 	successivi 4 byte: numero di colonne (M) --> numero intero a 32 bit
* 	successivi N*M*4 byte: matrix data in row-major order --> numeri interi o floating-point a precisione singola
*/
void save_data(char* filename, void* X, int n, int k) {
	FILE* fp;
	int i;
	fp = fopen(filename, "wb");
	if(X != NULL){
		fwrite(&n, 4, 1, fp);
		fwrite(&k, 4, 1, fp);
		for (i = 0; i < n; i++) {
			fwrite(X, sizeof(type), k, fp);
			//printf("%i %i\n", ((int*)X)[0], ((int*)X)[1]);
			X += sizeof(type)*k;
		}
	}
	else{
		int x = 0;
		fwrite(&x, 4, 1, fp);
		fwrite(&x, 4, 1, fp);
	}
	fclose(fp);
}

int main(int argc, char** argv) {
    // ================= Parametri di ingresso =================
    char* dsfilename = "ds.ds2";
    char* queryfilename = "query.ds2";
    int h = 2;
    int k = 3;
    int x = 2;
    int silent = 0;
    // =========================================================
    
    params* input = malloc(sizeof(params));
    input->DS = load_data(dsfilename, &input->N, &input->D);
		// ============================================================
	// Validazione ROBUSTA dei parametri
	printf("=== VALIDAZIONE PARAMETRI ===\n");
	printf("Dataset: N=%d, D=%d\n", input->N, input->D);
	printf("Query: nq=%d\n", input->nq);
	printf("Richiesti: h=%d, k=%d, x=%d\n", h, k, x);

	// 1. Verifica h (numero pivot)
	if(h <= 0) {
		printf("ERRORE: h=%d deve essere positivo (h > 0)\n", h);
		exit(1);
	}
	if(h > input->N) {
		printf("ERRORE: h=%d non può essere maggiore di N=%d\n", h, input->N);
		exit(1);
	}

	// 2. Verifica k (numero vicini)
	if(k <= 0) {
		printf("ERRORE: k=%d deve essere positivo (k > 0)\n", k);
		exit(1);
	}
	if(k > input->N) {
		printf("ERRORE: k=%d non può essere maggiore di N=%d\n", k, input->N);
		printf("Suggerimento: usa k <= N\n");
		exit(1);
	}

	// 3. Verifica x (parametro quantizzazione)
	if(x <= 0) {
		printf("ERRORE: x=%d deve essere positivo (x > 0)\n", x);
		exit(1);
	}
	if(x > input->D) {
		printf("WARNING: x=%d > D=%d, verrà limitato a x=D\n", x, input->D);
		x = input->D;
	}

	// 4. Verifica dimensioni minime
	if(input->D < 1) {
		printf("ERRORE: D=%d deve essere almeno 1\n", input->D);
		exit(1);
	}
	if(input->N < 2) {
		printf("ERRORE: N=%d deve essere almeno 2\n", input->N);
		exit(1);
	}
	if(input->nq < 1) {
		printf("ERRORE: nq=%d deve essere almeno 1\n", input->nq);
		exit(1);
	}

	// 5. Verifica compatibilità query
	if(input->nq > 0) {
		// La dimensione delle query viene verificata automaticamente nel load
		printf("Validazione completata con successo\n\n");
	}

	// Inizializza flag per fit (SE NON ANCORA FATTO)
	input->first_fit_call = false;

	// RESTO DEL CODICE INVARIATO...
    input->Q = load_data(queryfilename, &input->nq, &input->D);
    
    // ========== AGGIUNGI SOLO QUESTO BLOCCO ==========
    // Validazione parametri
    if(h <= 0 || h > input->N) {
        printf("ERRORE: h=%d non valido (deve essere 1..%d)\n", h, input->N);
        exit(1);
    }
    if(k <= 0 || k > input->N) {
        printf("ERRORE: k=%d non valido (deve essere 1..%d)\n", k, input->N);
        exit(1);
    }
    if(x <= 0) {
        printf("ERRORE: x=%d deve essere positivo\n", x);
        exit(1);
    }
    if(x > input->D) {
        printf("WARNING: x=%d > D=%d, verrÃ  limitato a D\n", x, input->D);
        x = input->D;
    }
    // Inizializza flag per fit
    input->first_fit_call = false;
    // =================================================
    
    // RESTO DEL CODICE DEL PROF INVARIATO
    input->id_nn = _mm_malloc(input->nq*k*sizeof(int), align);
    input->dist_nn = _mm_malloc(input->nq*k*sizeof(type), align);
    input->h = h;
    input->k = k;
    input->x = x;
    input->silent = silent;
    
    clock_t t;
    float time;
    
    t = omp_get_wtime();
    // =========================================================
    fit(input);
    // =========================================================
    t = omp_get_wtime() - t;
    time = ((float)t)/CLOCKS_PER_SEC;
    
    if(!input->silent)
        printf("FIT time = %.5f secs\n", time);
    else
        printf("%.3f\n", time);
    
    t = omp_get_wtime();
    // =========================================================
    predict(input);
    // =========================================================
    t = omp_get_wtime() - t;
    time = ((float)t)/CLOCKS_PER_SEC;
    
    if(!input->silent)
        printf("PREDICT time = %.5f secs\n", time);
    else
        printf("%.3f\n", time);
    
    // Salva il risultato
    char* outname_id = "out_idnn.ds2";
    char* outname_k = "out_distnn.ds2";
    save_data(outname_id, input->id_nn, input->nq, input->k);
    save_data(outname_k, input->dist_nn, input->nq, input->k);
    
    if(!input->silent){
        for(int i=0; i<input->nq; i++){
            printf("ID NN Q%3i: ( ", i);
            for(int j=0; j<input->k; j++)
                printf("%i ", input->id_nn[i*input->k + j]);
            printf(")\n");
        }
        for(int i=0; i<input->nq; i++){
            printf("Dist NN Q%3i: ( ", i);
            for(int j=0; j<input->k; j++)
                printf("%f ", input->dist_nn[i*input->k + j]);
            printf(")\n");
        }
    }
    
    _mm_free(input->DS);
    _mm_free(input->Q);
    _mm_free(input->P);
    _mm_free(input->index);
    _mm_free(input->id_nn);
    _mm_free(input->dist_nn);
    free(input);
    
    return 0;
}