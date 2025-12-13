#!/usr/bin/env python3
"""
Test completo per verificare robustezza con dataset di diverse dimensioni
"""

import numpy as np
import gruppo6.quantpivot32
import sys

def test_edge_case(N, D, h, k, x, name):
    """Test un singolo caso limite"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Parametri: N={N}, D={D}, h={h}, k={k}, x={x}")
    print(f"{'='*60}")
    
    try:
        # Crea dataset con numpy (automaticamente allineato)
        DS = np.random.randn(N, D).astype(np.float32)
        Q = np.random.randn(2, D).astype(np.float32)
        
        # Verifica allineamento
        ds_align = DS.ctypes.data % 16
        q_align = Q.ctypes.data % 16
        print(f"Allineamento DS: {ds_align} byte (0 = perfetto)")
        print(f"Allineamento Q:  {q_align} byte (0 = perfetto)")
        
        # Test fit
        qp = gruppo6.quantpivot32.QuantPivot()
        qp.fit(DS, h, x, silent=True)
        print(f"FIT completato")
        
        # Test predict
        ids, dists = qp.predict(Q, k, silent=True)
        print(f"PREDICT completato")
        
        # Verifica risultati
        assert ids.shape == (2, k), f"Shape IDs errata: {ids.shape}"
        assert dists.shape == (2, k), f"Shape DISTS errata: {dists.shape}"
        assert np.all(ids >= 0), "IDs negativi trovati!"
        assert np.all(ids < N), f"IDs fuori range [0, {N})!"
        assert np.all(dists >= 0), "Distanze negative trovate!"
        assert np.all(np.isfinite(dists)), "Distanze infinite o NaN trovate!"
        
        print(f"VALIDAZIONE passata")
        print(f"   IDs shape: {ids.shape}")
        print(f"   Dists shape: {dists.shape}")
        print(f"   Dist range: [{dists.min():.3f}, {dists.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"ERRORE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("TEST SUITE COMPLETA - CASI LIMITE")
    print("="*60)
    
    tests = [
        # (N, D, h, k, x, nome)
        
        # DIMENSIONI MINIME
        (100, 1, 2, 5, 1, "D=1 (minimo assoluto)"),
        (100, 2, 2, 5, 1, "D=2"),
        (100, 3, 2, 5, 2, "D=3 (dispari)"),
        
        # DIMENSIONI DISPARI/PRIME
        (100, 5, 2, 5, 3, "D=5 (primo)"),
        (100, 7, 2, 5, 4, "D=7 (primo)"),
        (100, 11, 2, 5, 5, "D=11 (primo)"),
        (100, 13, 2, 5, 6, "D=13 (primo)"),
        
        # NON MULTIPLI DI 4
        (100, 15, 2, 5, 8, "D=15 (non multiplo 4)"),
        (100, 17, 2, 5, 8, "D=17 (non multiplo 4)"),
        (100, 31, 2, 5, 16, "D=31 (non multiplo 4)"),
        (100, 63, 2, 5, 32, "D=63 (non multiplo 4)"),
        
        # MULTIPLI "STRANI"
        (100, 6, 2, 5, 3, "D=6 (multiplo 2, non 4)"),
        (100, 10, 2, 5, 5, "D=10 (multiplo 2, non 4)"),
        (100, 14, 2, 5, 7, "D=14 (multiplo 2, non 4)"),
        
        # MULTIPLI DI 4 MA NON 8
        (100, 12, 2, 5, 6, "D=12 (multiplo 4, non 8)"),
        (100, 20, 2, 5, 10, "D=20 (multiplo 4, non 8)"),
        (100, 28, 2, 5, 14, "D=28 (multiplo 4, non 8)"),
        
        # MULTIPLI DI 8 MA NON 16
        (100, 24, 2, 5, 12, "D=24 (multiplo 8, non 16)"),
        (100, 40, 2, 5, 20, "D=40 (multiplo 8, non 16)"),
        (100, 56, 2, 5, 28, "D=56 (multiplo 8, non 16)"),
        
        # NON MULTIPLI DI 16
        (100, 127, 2, 5, 64, "D=127 (non multiplo 16)"),
        (100, 255, 2, 5, 64, "D=255 (non multiplo 16)"),
        (100, 257, 2, 5, 64, "D=257 (non multiplo 16)"),
        
        # MULTIPLI PERFETTI DI 16
        (100, 16, 2, 5, 8, "D=16 (multiplo perfetto)"),
        (100, 32, 2, 5, 16, "D=32 (multiplo perfetto)"),
        (100, 64, 2, 5, 32, "D=64 (multiplo perfetto)"),
        (100, 128, 2, 5, 64, "D=128 (multiplo perfetto)"),
        (100, 256, 2, 5, 64, "D=256 (multiplo perfetto)"),
        
        # DATASET PICCOLI
        (10, 256, 2, 3, 64, "N=10 (molto piccolo)"),
        (25, 256, 5, 5, 64, "N=25"),
        (50, 256, 10, 10, 64, "N=50"),
        
        # DATASET GRANDI
        (1000, 256, 16, 8, 64, "N=1000"),
        (5000, 256, 32, 8, 64, "N=5000"),
        (10000, 128, 64, 8, 32, "N=10000"),
        
        # DIMENSIONI ALTE
        (500, 512, 16, 8, 128, "D=512 (alta dimensionalità)"),
        (200, 1000, 10, 5, 256, "D=1000 (molto alta)"),
        (100, 2048, 10, 5, 512, "D=2048 (estrema)"),
        
        # PARAMETRI X ESTREMI
        (100, 256, 10, 5, 1, "x=1 (minimo)"),
        (100, 256, 10, 5, 256, "x=D (massimo)"),
        (100, 256, 10, 5, 255, "x=D-1"),
        (100, 64, 10, 5, 63, "x=D-1 (dispari)"),
        
        # PARAMETRI K VARI
        (100, 256, 10, 1, 64, "k=1 (minimo)"),
        (100, 256, 10, 50, 64, "k=50 (grande)"),
        (100, 256, 10, 99, 64, "k=99 (quasi N)"),
        
        # PARAMETRI H VARI
        (100, 256, 1, 5, 64, "h=1 (minimo)"),
        (100, 256, 50, 5, 64, "h=50 (grande)"),
        (1000, 256, 100, 8, 64, "h=100 (molto grande)"),
        
        # COMBINAZIONI DIFFICILI
        (100, 7, 3, 5, 5, "Tutto dispari"),
        (99, 255, 11, 7, 127, "Tutti parametri dispari/primi"),
        (101, 257, 13, 11, 129, "Numeri primi ovunque"),
    ]
    
    passed = 0
    failed = 0
    
    for test_params in tests:
        if test_edge_case(*test_params):
            passed += 1
        else:
            failed += 1
    
    # SUMMARY
    print("\n" + "="*60)
    print("RIEPILOGO FINALE")
    print("="*60)
    print(f"Test passati: {passed}")
    print(f"Test falliti: {failed}")
    print(f"Totale: {passed + failed}")
    print(f"Success rate: {100*passed/(passed+failed):.1f}%")
    
    if failed == 0:
        print("\n TUTTI I TEST PASSATI! Codice production-ready! 🎉")
        return 0
    else:
        print(f"\n  {failed} test falliti - richiede debugging")
        return 1


if __name__ == "__main__":
    sys.exit(main())