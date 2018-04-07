import numpy as np

def gen_basis_indices(F,H,W,dim):
    basis_indices = np.zeros((F,dim))

    if dim == H*W:
        basis_indices = np.matlib.repmat(np.arange(H*W),F,1)
    else:            
        for ii in range(F):
            indices = np.random.permutation(np.arange(dim//3,H*W - np.int(2*dim//3)))[:dim//3]
            basis_indices[ii,:dim//3] = (np.arange(dim//3)).astype(int)
            basis_indices[ii,dim//3:2*dim//3] = indices
            basis_indices[ii,2*dim//3:] = (np.arange(H*W - np.ceil(dim/3).astype(int),H*W)).astype(int)
    return basis_indices