import h5py
import numpy as np

def ler_tags_materiais(h5file="malha_dominio.h5"):
    with h5py.File(h5file, "r") as f:
        tags = np.array(f["data2"])
    return tags

if __name__ == "__main__":
    tags = ler_tags_materiais()
    print("Tags dos materiais por elemento:", tags)
