#!/usr/bin/env python3
"""
BENCHMARK SCALABILITÀ - Progetto QuantPivot Gruppo 6
Esegue test su dataset di dimensioni crescenti per tutte e 3 le versioni.
Genera dataset al volo e usa direttamente le librerie Python.
"""

import sys
import os
import time
import numpy as np
import pyfftw

# Configurazione
DATASET_SIZES = [2000, 10000, 50000, 100000, 250000, 500000, 600000, 700000, 800000]
D = 256
NQ = 2000
H = 16
K = 8
X = 64

# Risultati: {size: {version: (fit, prd)}}
results = {}

def format_size(n):
    """Formatta dimensione in formato leggibile."""
    if n >= 1000000:
        return f"{n//1000000}M"
    elif n >= 1000:
        return f"{n//1000}K"
    return str(n)

def generate_dataset(n, d, dtype, alignment):
    """Genera dataset random allineato."""
    data = pyfftw.empty_aligned((n, d), dtype=dtype, n=alignment)
    data[:] = np.random.randn(n, d).astype(dtype)
    return data

def run_test_32(dataset, query):
    """Esegue test versione 32-bit."""
    try:
        from gruppo6.quantpivot32 import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 32: {e}")
        return None

def run_test_64(dataset, query):
    """Esegue test versione 64-bit."""
    try:
        from gruppo6.quantpivot64 import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 64: {e}")
        return None

def run_test_64omp(dataset, query):
    """Esegue test versione 64-bit OpenMP."""
    try:
        from gruppo6.quantpivot64omp import QuantPivot
        
        model = QuantPivot()
        
        start = time.perf_counter()
        model.fit(dataset, H, X, True)  # silent=True
        fit_time = time.perf_counter() - start
        
        start = time.perf_counter()
        ids, dists = model.predict(query, K, True)  # silent=True
        prd_time = time.perf_counter() - start
        
        return (fit_time, prd_time)
    except Exception as e:
        print(f"ERRORE 64omp: {e}")
        return None

def print_table():
    """Stampa tabella riepilogativa."""
    
    print("\n" + "="*80)
    print("  BENCHMARK SCALABILITÀ - Progetto QuantPivot Gruppo 6")
    print(f"  Query: {NQ}, h={H}, k={K}, x={X}")
    print("="*80)
    
    # Header
    print(f"\n{'Dataset':<12} | {'32-bit SSE':<20} | {'64-bit AVX':<20} | {'64-bit OMP':<20}")
    print(f"{'(N×256)':<12} | {'FIT / PRD (sec)':<20} | {'FIT / PRD (sec)':<20} | {'FIT / PRD (sec)':<20}")
    print("-"*80)
    
    for size in DATASET_SIZES:
        if size not in results:
            continue
        label = format_size(size)
        row = f"{label:<12} |"
        
        for version in ["32", "64", "64omp"]:
            if version in results[size]:
                res = results[size][version]
                if res is None:
                    row += f" {'ERRORE':<20}|"
                elif res == "SKIP":
                    row += f" {'SKIP (OOM)':<20}|"
                else:
                    fit, prd = res
                    row += f" {fit:>6.2f} / {prd:<10.2f} |"
            else:
                row += f" {'-':<20}|"
        print(row)
    
    print("-"*80)
    
    # Speedup OMP vs 32-bit
    print("\nSPEEDUP OpenMP vs 32-bit SSE:")
    for size in DATASET_SIZES:
        if size not in results:
            continue
        label = format_size(size)
        if "32" in results[size] and "64omp" in results[size]:
            res32 = results[size]["32"]
            res_omp = results[size]["64omp"]
            if res32 and res_omp and res32 != "SKIP" and res_omp != "SKIP":
                _, prd32 = res32
                _, prd_omp = res_omp
                if prd_omp > 0:
                    speedup = prd32 / prd_omp
                    print(f"   {label}: {speedup:.2f}x")
    
    print("\n" + "="*80)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*80)
    print("BENCHMARK SCALABILITÀ - QuantPivot")
    print("="*80)
    print(f"Parametri: D={D}, NQ={NQ}, h={H}, k={K}, x={X}")
    print("="*80)
    
    # Genera query una volta sola
    print("\nGenerazione query allineate...")
    query_32 = generate_dataset(NQ, D, 'float32', 16)
    query_64 = generate_dataset(NQ, D, 'float64', 32)
    print(f"  Query 32-bit: {query_32.shape}, aligned={query_32.ctypes.data % 16 == 0}")
    print(f"  Query 64-bit: {query_64.shape}, aligned={query_64.ctypes.data % 32 == 0}")
    
    total_tests = len(DATASET_SIZES) * 3
    current = 0
    
    for size in DATASET_SIZES:
        label = format_size(size)
        results[size] = {}
        
        print(f"\n{'='*60}")
        print(f"DATASET SIZE: {size:,} x {D}")
        print(f"{'='*60}")
        
        # Genera dataset
        print(f"  Creazione dataset {size:,} x {D}...")
        try:
            ds_32 = generate_dataset(size, D, 'float32', 16)
            print(f"    32-bit: {ds_32.shape}, aligned={ds_32.ctypes.data % 16 == 0}")
        except MemoryError:
            print(f"    32-bit: OOM!")
            ds_32 = None
            
        try:
            ds_64 = generate_dataset(size, D, 'float64', 32)
            print(f"    64-bit: {ds_64.shape}, aligned={ds_64.ctypes.data % 32 == 0}")
        except MemoryError:
            print(f"    64-bit: OOM!")
            ds_64 = None
        
        # Test 32-bit
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 32-bit SSE...")
        if ds_32 is not None:
            res = run_test_32(ds_32, query_32)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["32"] = res
        else:
            results[size]["32"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit AVX...")
        if ds_64 is not None:
            res = run_test_64(ds_64, query_64)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64"] = res
        else:
            results[size]["64"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Test 64-bit OpenMP
        current += 1
        print(f"\n  [{current}/{total_tests}] Testing 64-bit AVX + OpenMP...")
        if ds_64 is not None:
            res = run_test_64omp(ds_64, query_64)
            if res:
                print(f"    Fit: {res[0]:.4f}s, Predict: {res[1]:.4f}s, Total: {res[0]+res[1]:.4f}s")
            results[size]["64omp"] = res
        else:
            results[size]["64omp"] = "SKIP"
            print(f"    SKIP (OOM)")
        
        # Libera memoria
        del ds_32, ds_64
        import gc
        gc.collect()
    
    # Stampa tabella finale
    print_table()
    
    # Salva su file
    import io
    from contextlib import redirect_stdout
    
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_table()
    
    with open("benchmark_results.txt", "w") as f:
        f.write(buffer.getvalue())
    
    print(f"\nRisultati salvati in: benchmark_results.txt")

if __name__ == "__main__":
    main()
