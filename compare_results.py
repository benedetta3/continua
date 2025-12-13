import numpy as np
import struct

def load_ds2(filename, dtype):
    with open(filename, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        d = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, dtype=dtype).reshape(n, d)
    return data

# --- File del professore ---
id_prof = load_ds2("results_ids_2000x8_k8_x64_32.ds2", np.int32)
dist_prof = load_ds2("results_dst_2000x8_k8_x64_32.ds2", np.float32)

# --- I tuoi file ---
id_me = load_ds2("idNN_32_size-2000x256_nq-2000.ds2", np.int32)
dist_me = load_ds2("distNN_32_size-2000x256_nq-2000.ds2", np.float32)

# --- Confronto ID ---
print("Confronto ID...")
id_equal = np.array_equal(id_prof, id_me)
print("ID identici:", id_equal)

if not id_equal:
    diff = np.where(id_prof != id_me)
    print("Numero differenze:", len(diff[0]))
    print("Prime differenze (prof -> tuo):")
    for idx in range(min(10, len(diff[0]))):
        i, j = diff[0][idx], diff[1][idx]
        print(f"  posizione {i},{j}: prof={id_prof[i,j]}  tu={id_me[i,j]}")

# --- Confronto DISTANZE ---
print("\nConfronto distanze...")
diff = np.abs(dist_prof - dist_me)
max_diff = np.max(diff)

print("Massima differenza:", max_diff)

if max_diff <= 0.2:
    print("Distanze compatibili (errore â‰¤ 0.2)")
else:
    print("Distanze NON compatibili (errore > 0.2)")