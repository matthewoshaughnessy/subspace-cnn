import numpy as np
import scipy.fftpack
import torch


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

def subspace_projection_gpu(k,w,basis=None, basis_indices=None):
    """
    n: Image height
    k: subspace dimension
    F: number of filters
    w: Filter coefficient: (F x C x HH x WW )
    """
    F,C,HH,WW = w.shape
    if basis is None:
        basis = torch.from_numpy(scipy.fftpack.dct(np.eye(HH*WW),norm='ortho').as_type(float)).cuda()
    else:
        basis = basis.cuda()
    
    #w_reshaped = np.reshape(w,(F,C,HH*WW),'F')
    w_reshaped = w.view(F,C,HH*WW) # TODO -- 'F'?  transpose first?

    #w_projected = np.zeros(w_reshaped.shape)
    w_projected = torch.zeros(w_reshaped.shape)
    
    if basis_indices is None:
        #basis_indices = np.zeros((F,k))
        basis_indices = torch.zeros((F,k))
        for ii in range(F):
            basis_indices[ii,:] = np.arange(k)
            
    for ii in range(F):
        indices = basis_indices[ii,:].astype(int)
        for jj in range(C):
            basis_i = basis[:,indices]
            #B = np.dot(basis[:,indices],basis[:,indices].T)
            B = torch.mm(basis_i,torch.transpose(basis_i,0,1))
            #w_projected[ii,jj,:] = np.dot(B,w_reshaped[ii,jj,:])
            w_projected[ii,jj,:] = torch.mm(B,w_reshaped[ii,jj,:])
    #w_projected = np.reshape(w_projected,(F,C,HH,WW),'F')
    w_projected = w_projected.view(F,C,HH,WW)
    return w_projected

