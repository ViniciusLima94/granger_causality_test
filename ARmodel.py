'''
Autoregressive model example
X(t) = 0.9X(t-1) - 0.5X(t-2) + e(t)
Y(t) = 0.8Y(t-1) - 0.5Y(t-2) + 0.16X(t-1) - 0.2X(t-2) + n(t)
'''

import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as sig

np.random.seed(1981)

def psd(x, y=[], T=1000):
	'''
	Computes the power spectrum of x, if y is given then the cross spectral density is computed instead.
	x: x time series
	y: y time series
	T: Time length
	'''
	Xf = np.fft.fft(x)[:len(x)//2 + 1]

	if len(y) > 0:
		Yf = np.fft.fft(y)[:len(y)//2 + 1]
		Sxy = np.multiply(Yf, np.conjugate(Xf)) / float( T )
		return Sxy
	else:
		Sxx = np.multiply(Xf, np.conjugate(Xf)) / float( T )
		return Sxx

def ARmodel(nfreq = 1024, T = 100):
	N = nfreq * T

	cov = np.array([ [1.00, 0.40], 
		             [0.40, 0.70] ])

	E = nz = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=(N,))
	
	X = np.random.rand(N)
	Y = np.random.rand(N)

	for i in range(2, N):
		X[i] = 0.9 * X[i-1] - 0.5 * X[i-2] + E[:,0][i]
		Y[i] = 0.8 * Y[i-1] - 0.5 * Y[i-2] + 0.16 * X[i-1] - 0.2 * X[i-2] + E[:,1][i] 

	return X, Y

def cov_matrix(Z, lags):
	N = len(Z)
	L = len(Z[0])
	R = np.zeros([N, N, len(lags)])
	for i in range( len(lags) ):
		n = -lags[i]
		for r in range(N):
			for c in range(N):
				if i == 0:
					R[r, c, i] +=  np.dot(Z[r],Z[c].T)
				else:
					R[r, c, i] +=  np.dot(Z[r][lags[i]:],Z[c][:-lags[i]].T)
		R[:,:,i] /= (L-n)
	return R

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


'''
a1 =  np.array([ [0.90, 0.00], k = np.array([0, 1, 2])
Rxx = np.zeros([2, 2, len(k)])
for i in range(N):
	Rxx += cov_matrix(z[i], k) / float(N)
	             [0.16, 0.80] ])
a2 = -np.array([ [0.50, 0.00], 
	             [0.20, 0.50] ])

cov = np.array([ [1.00, 0.40], 
	             [0.40, 0.70] ])


sxx = (1+1j) * np.zeros(129)
syy = (1+1j) * np.zeros(129)
sxy = (1+1j) * np.zeros(129)

k = np.array([0, 1, 2])

Trials = 1000
R = np.zeros([2, 2, len(k)])

for i in range(Trials):
	X, Y = ARmodel()
	R += cov_matrix([X, Y], k) / float(Trials)
	
	_, sxx_aux = sig.welch(X, fs = 1024)
	_, syy_aux = sig.welch(Y, fs = 1024)
	_, sxy_aux = sig.csd(X, Y, fs = 1024)
	sxx+=sxx_aux/Trials
	syy+=syy_aux/Trials
	sxy+=sxy_aux/Trials
	'''

import numpy as np
import matplotlib.pyplot as plt

import nitime.algorithms as alg
import nitime.utils as utils
'''
a1 = np.array([[0.9, 0],
               [0.16, 0.8]])

a2 = np.array([[-0.5, 0],
               [-0.2, -0.5]])

am = np.array([-a1, -a2])

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([[x_var, xy_cov],
                [xy_cov, y_var]])
'''

#Number of realizations of the process
N = 500

k = np.array([0, 1, 2])
Rxx = np.zeros([2, 2, len(k)])
for i in range(N):
	#Rxx += cov_matrix(z[i], k) / float(N)
	X, Y = ARmodel(nfreq = 1024, T = 1)
	Rxx += cov_matrix([X,Y], k) / float(N)

R0 = Rxx[..., 0]
Rm = Rxx[..., 1:]

Rxx = Rxx.transpose(2, 0, 1)

a, ecov = alg.lwr_recursion(Rxx)

fx2y, fy2x, fxy = granger_causality(-a, ecov, nfreq=1024)

plt.plot(fx2y.real)
plt.plot(fy2x.real)
plt.plot(fxy.real)
plt.xlim([0,515])
plt.ylim([-0.01,0.7])

'''
import numpy as np
import matplotlib.pyplot as plt

import nitime.algorithms as alg
import nitime.utils as utils

np.random.seed(1981)

am = np.array([-a1, -a2])

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([[x_var, xy_cov],
                [xy_cov, y_var]])

n_freqs = 1024

w, Hw = alg.transfer_function_xy(am, n_freqs=n_freqs)
Sw_true = alg.spectral_matrix_xy(Hw, cov)

#Number of realizations of the process
N = 500
#Length of each realization:
L = 1024

order = am.shape[0]
n_lags = order + 1

n_process = am.shape[-1]

z = np.empty((N, n_process, L))
nz = np.empty((N, n_process, L))

for i in range(N):
    z[i], nz[i] = utils.generate_mar(am, cov, L)


Rxx = np.empty((N, n_process, n_process, n_lags))

for i in range(N):
    Rxx[i] = utils.autocov_vector(z[i], nlags=n_lags)

Rxx = Rxx.mean(axis=0)    

R0 = Rxx[..., 0]
Rm = Rxx[..., 1:]

Rxx = Rxx.transpose(2, 0, 1)

a, ecov = alg.lwr_recursion(Rxx)
'''
