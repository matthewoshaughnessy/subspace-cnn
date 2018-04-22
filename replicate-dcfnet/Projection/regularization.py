import numpy as np
import scipy.fftpack


def subspace_projection(k,w,basis=None, basis_indices=None):
    """
	n: Image height
	k: subspace dimension
	F: number of filters
	w: Filter coefficient: (F x C x HH x WW )
	"""
    F,C,HH,WW = w.shape
    if basis is None:
        basis = scipy.fftpack.dct(np.eye(HH*WW),norm='ortho')
    
    print( w )
    w_reshaped = np.reshape(w,(F,C,HH*WW),'F')
    w_projected = np.zeros(w_reshaped.shape)
    
    if basis_indices is None:
        basis_indices = np.zeros((F,k))
        for ii in range(F):
            #indices = np.random.permutation(HH*WW)[:k]
            basis_indices[ii,:] = np.arange(k)
            
    for ii in range(F):
        indices = basis_indices[ii,:].astype(int)
        for jj in range(C):
            B = np.dot(basis[:,indices],basis[:,indices].T)
            w_projected[ii,jj,:] = np.dot(B,w_reshaped[ii,jj,:])
    w_projected = np.reshape(w_projected,(F,C,HH,WW),'F')
    return w_projected
