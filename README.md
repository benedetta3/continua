CODICI DA ESEGUIRE NEL TERMINALE DI VISUAL STUDIO CODE

------------------------------------------------------------------------------

INSTALLAZIONE VENV E DIPENDENZE:                    

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy
python3 setup.py build_ext --inplace
sudo apt update
sudo apt install libfftw3-dev libfftw3-doc
pip install pyfftw

------------------------------------------------------------------------------

EESECUZIONE 32:

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot32/*.so src/32/*.o
python3 setup.py build_ext --inplace
cd ~/Scrivania/Progetto/Mio
python3 test.py dataset_2000x256_32.ds2 query_2000x256_32.ds2 16 8 64 32
python3 compare_results.py

------------------------------------------------------------------------------

EESECUZIONE 64:

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64/*.so src/64/*.o
python3 setup.py build_ext --inplace
cd ~/Scrivania/Progetto/Mio
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64
python3 compare_results.py

------------------------------------------------------------------------------

EESECUZIONE 64 OMP:

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
rm -rf build/ gruppo6/quantpivot64omp/*.so src/64omp/*.o
python3 setup.py build_ext --inplace
cd ~/Scrivania/Progetto/Mio
python3 test.py dataset_2000x256_64.ds2 query_2000x256_64.ds2 16 8 64 64
python3 compare_results.py

------------------------------------------------------------------------------

TEST VALIDITÀ PROGRAMMA SU DATASET GRANDI, DISPARI, NON MULTIPLI DI 4...

cd ~/Scrivania/Progetto/Mio/ProgettoGruppo6
source venv/bin/activate
pip install -e .
cd ~/Scrivania/Progetto/Mio
python3 test_edge_cases.py

------------------------------------------------------------------------------

CREAZIONE DI DATASET E QUERY DI DIMENSIONI DIFFERENTI

gcc make_ds2.c -O3 -o make_ds2
./make_ds2 200000 256 DS_200k_256.ds2
./make_ds2 2000 256 Q_2000_256.ds2
python3 test.py DS_200k_256.ds2 Q_2000_256.ds2 16 8 64 32
 
===============================================================================

GUIDA AL PROGETTO – Architetture Avanzate (Gruppo 6)

Il progetto implementa l’algoritmo QuantPivot per k-NN: quantizza i vettori in (v+, v-), usa una distanza approssimata
per ottenere un lower bound, fa pruning dei candidati e infine calcola la distanza euclidea reale solo sui candidati rimasti.

Perché è implementato così (scelte progettuali)
------------------------------------------------------------------------------
1) Stessa API e stessa logica in tutte le versioni (32 / 64 / 64omp)
   - Questo riduce bug, facilita il confronto e rende i risultati riproducibili.
2) Separazione tra:
   - logica “algoritmica” in C (fit/query/quantizzazione/indice)
   - kernel ottimizzati in Assembly (SSE/AVX) per le parti numericamente pesanti
   - wrapper Python (py.c) del template, mantenuto invariato per compatibilità con la valutazione
3) Prestazioni:
   - la parte costosa è: quantizzazione + calcolo distanza approssimata + distanza euclidea finale
   - per questo esistono kernel ASM (SSE/AVX) e, in 64omp, parallelismo sui loop principali (OpenMP).

Struttura delle directory
------------------------------------------------------------------------------
Nel progetto trovate tipicamente:

ProgettoGruppo6/
  gruppo6/                  -> package Python (import gruppo6...)
  src/
    32/                     -> versione float + SSE
    64/                     -> versione double + AVX
    64omp/                  -> versione double + AVX + OpenMP
  test.py                   -> script di test (template)
  compare_results.py         -> confronto risultati (se presente/ricreato)
  clean                      -> pulizia artefatti build (consigliato prima dello zip)

Le 3 directory src/* hanno nomi di file simili (stessa “intestazione”), ma implementano tipi e ottimizzazioni diverse.

Differenze tra src/32, src/64, src/64omp
------------------------------------------------------------------------------
1) src/32  (float + SSE)
   - Dati: float (32 bit)
   - Vettorizzazione: SSE (registri XMM, 4 float per volta)
   - Target: baseline ottimizzata per 32 bit

2) src/64  (double + AVX)
   - Dati: double (64 bit)
   - Vettorizzazione: AVX (registri YMM, 4 double per volta)
   - Target: maggiore precisione + throughput su double

3) src/64omp (double + AVX + OpenMP)
   - Come src/64 ma con #pragma omp parallel for sui loop “grandi”
   - Target: sfruttare più core su fit/query, mantenendo kernel AVX per inner-loop

Nota: i nomi dei file sono simili per uniformità (e per rimanere aderenti al template), ma:
- cambiano i tipi (float vs double)
- cambiano i kernel ASM (SSE vs AVX)
- in 64omp cambiano i loop (parallelizzati) e i flag di compilazione (OpenMP)

API Python (cosa vede chi usa il progetto)
------------------------------------------------------------------------------
Il progetto esporta 3 classi “gemelle”, una per directory:

- QuantPivot32
- QuantPivot64
- QuantPivot64OMP

Uso tipico:

qp = QuantPivotXX(...)
qp.fit(DS, h, x, ...)    # costruisce pivot, quantizza, costruisce indice
ids, dists = qp.query(Q, k, ...)   # risponde alle query con k-NN

Dove:
- DS è il dataset (N x D)
- Q  è il query set (nq x D)
- h  è il numero di pivot
- k  è il numero di vicini richiesti
- x  è il parametro di quantizzazione (soglia/numero selezioni non-zero)

Che cosa fa ogni componente (alto livello)

1) Wrapper Python (file *_py.c)
   - Converte gli array numpy -> puntatori C
   - Controlla parametri e chiama le funzioni C
   - Gestisce la vita dell’oggetto (alloc/free)
   - Nota: lasciato uguale al template per compatibilità

2) C “core” (file quantpivotXX.c)
   Contiene le funzioni principali, in genere:

   a) fit()
      - alloca e inizializza strutture
      - genera i pivot P (h vettori)
      - quantizza DS in (v_plus, v_minus)
      - costruisce l’indice idx[v, j] = d̃(v, Pj) per tutti i punti v e pivot j

   b) query()
      - per ogni query q:
        1) quantizza q in (q_plus, q_minus)
        2) calcola qpivot[j] = d̃(q, Pj)
        3) per ogni v nel dataset calcola lower bound:
             LB(v,q) = max_j |idx[v,j] - qpivot[j]|
        4) seleziona candidati migliori per LB (pruning)
        5) calcola distanza euclidea reale sui candidati
        6) restituisce i k vicini (id + dist)

   c) quantizing() (o equivalente)
      - trasforma un vettore reale in due vettori non-negativi:
        v+ contiene i contributi positivi
        v- contiene i contributi negativi (in valore assoluto)
      - tipicamente “sparse” controllata da x (sceglie le componenti più rilevanti)

   d) approx_distance()
      - implementa la distanza approssimata d̃:
        (v+·w+) + (v-·w-) – (v+·w-) – (v-·w+)

   e) euclidean_distance()
      - distanza reale, usata solo in finale su pochi candidati

3) ASM (file quantpivotXX.nasm + utils)
   - Implementa i kernel “hot” (dot product / distanze) in SSE o AVX
   - Riduce overhead dei loop e sfrutta SIMD (unrolling, prefetch, ecc.)

4) Script run/benchmark (se presenti)
   - Servono per test locali, confronto C vs ASM, e misurazioni tempo

Differenze tra file con “stessa intestazione” (esempi)
------------------------------------------------------------------------------
- quantpivot32.c vs quantpivot64.c:
  * stesso algoritmo, ma float vs double
  * cambiamento di tipi e, spesso, step SIMD (4 float vs 4 double per AVX)

- quantpivot32.nasm vs quantpivot64.nasm:
  * SSE (XMM) per 32
  * AVX (YMM) per 64

- quantpivot64.nasm vs quantpivot64omp.nasm:
  * kernel AVX simili (a volte identici)
  * la differenza principale è nel C: parallelizzazione OpenMP sui loop esterni

Note pratiche per chi clona e compila
------------------------------------------------------------------------------
1) Prima di zippare o pushare:
   - eseguire ./clean
   - non includere build/, *.o, *.so, __pycache__/ (evita problemi a chi compila)

2) Installazione/editable:
   - dentro ProgettoGruppo6:  pip install -e .
   - poi usare test.py dalla root per eseguire i test

3) Debug:
   - se servono log, farli in C (printf/fflush) nelle funzioni chiave:
     fit(): pivot/alloc/quantizzazione/indice
     query(): lower bound/candidati/kNN finale

Glossario rapido:
- P: matrice dei pivot (h x D)
- v+, v-: vettori quantizzati non-negativi
- idx[v,j]: valore d̃ tra punto v e pivot j (indice)
- qpivot[j]: valore d̃ tra query q e pivot j
- LB: lower bound per pruning (massimo scarto sugli h pivot)
- pruning: elimina punti sicuramente non nei k migliori prima della distanza reale

Fine.
