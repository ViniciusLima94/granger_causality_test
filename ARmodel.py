'''
Autoregressive model example
X(t) = 0.9X(t-1) - 0.5X(t-2) + e(t)
Y(t) = 0.8Y(t-1) - 0.5Y(t-2) + 0.16X(t-1) - 0.2X(t-2) + n(t)
'''

import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as sig

def ARmodel(Tsim = 100):
	cov = np.array([ [1.00, 0.40], 
		             [0.40, 0.70] ])

	E = np.dot( cov, np.random.randn(2, Tsim) )
	e, n = E[0], E[1]
	
	X = np.random.rand(Tsim)
	Y = np.random.rand(Tsim)

	for i in range(2, Tsim):
		X[i] = 0.9 * X[i-1] - 0.5 * X[i-2] + e[i]
		Y[i] = 0.8 * Y[i-1] - 0.5 * Y[i-2] + 0.16 * X[i-1] - 0.2 * X[i-2] + n[i] 

	return X, Y

def transfer_function(coeff_mtx, cov_mtx, nfreq):
	w = np.arange(0,np.pi, np.pi/(nfreq//2+1))
	N = len(coeff_mtx)
	a2 = 1 - np.sum( np.squeeze([coeff_mtx[i-1][0,0] * np.exp(-1j*w*i) for i in range(1,N+1)]), axis = 0 )
	b2 = 0 - np.sum( np.squeeze([coeff_mtx[i-1][0,1] * np.exp(-1j*w*i) for i in range(1,N+1)]), axis = 0 )
	c2 = 0 - np.sum( np.squeeze([coeff_mtx[i-1][1,0] * np.exp(-1j*w*i) for i in range(1,N+1)]), axis = 0 )
	d2 = 1 - np.sum( np.squeeze([coeff_mtx[i-1][1,1] * np.exp(-1j*w*i) for i in range(1,N+1)]), axis = 0 )

	A = np.array([ [a2, b2],
		           [c2, d2] ])

	H = np.array([ [d2, -b2], 
		           [-c2, a2]]) / ((A[0,0]*A[1,1]-A[1,0]*A[0,1]))
	return w, H

def spectral_matrix(Hw, cov):
	Sw = (1+1j)*np.zeros([2, 2, Hw.shape[-1]])

	t00 = cov[0, 0] * Hw[0, 0].conj() + cov[0, 1] * Hw[0, 1].conj()
	t01 = cov[0, 0] * Hw[1, 0].conj() + cov[0, 1] * Hw[1, 1].conj()
	t10 = cov[1, 0] * Hw[0, 0].conj() + cov[1, 1] * Hw[0, 1].conj()
	t11 = cov[1, 0] * Hw[1, 0].conj() + cov[1, 1] * Hw[1, 1].conj()
	Sw[0, 0] = Hw[0, 0] * t00 + Hw[0, 1] * t10
	Sw[0, 1] = Hw[0, 0] * t01 + Hw[0, 1] * t11
	Sw[1, 0] = Hw[1, 0] * t00 + Hw[1, 1] * t10
	Sw[1, 1] = Hw[1, 0] * t01 + Hw[1, 1] * t11
	return Sw

def granger_causality(coeff_mtx, cov, nfreq):
	w, H = transfer_function(coeff_mtx, cov, nfreq)
	S    = spectral_matrix(H, cov)

	Hxx_tilda = H[0,0,:] + cov[1,0]/cov[0,0] * H[0,1,:]
	Hyy_circn = H[1,1,:] + cov[1,0]/cov[1,1] * H[1,0,:]

	fx2y = np.log( S[1,1,:] / (Hyy_circn*cov[1,1]*np.conj(Hyy_circn)) )
	fy2x = np.log( S[0,0,:] / (Hxx_tilda*cov[0,0]*np.conj(Hxx_tilda)) )
	fxy  = np.log( (Hxx_tilda*cov[0,0]*np.conj(Hxx_tilda)) * (Hyy_circn*cov[1,1]*np.conj(Hyy_circn)) / (S[0,0]*S[1,1]-S[1,0]*S[0,1]) )

	return fx2y, fy2x, fxy

a1 =  np.array([ [0.90, 0.00], 
	             [0.16, 0.80] ])
a2 = -np.array([ [0.50, 0.00], 
	             [0.20, 0.50] ])

cov = np.array([ [1.00, 0.40], 
	             [0.40, 0.70] ])




