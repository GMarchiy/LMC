import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import erfc
from numba import njit, prange
from scipy.optimize import fsolve
from scipy.stats import skewnorm, expon, linregress
import pickle
Na = 6e23
kB = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759

@njit(cache=True)
def prob(E, T):
    return np.exp(-E/(kB*T)) #min(1, exp) however p always in [0, 1) so if exp>1 prob = 1

@njit(cache=True)
def acc(E, T):
    p = np.random.random()
    if p<prob(E, T):
        return True
    else:
        return False
    
def init(Xtot, s, Fsize):
    cnt = 0
    for i in range(M):
        for j in range(Fsize):
            s[i][j] = 1
            cnt += 1
            if cnt == round(Xtot*M*Fsize):
                break
        if cnt == round(Xtot*M*Fsize):
            break
    return cnt
        
@njit(cache=True)
def trial(s, njobs):# a -> b, b -> a
    idx = np.arange(M*Fsize)
    idsa = idx[s.ravel()==-1]
    lsta = idsa[:]
    idsb = idx[s.ravel()==+1]
    lstb = idsb[:]
    # id1 = np.random.randint(0, M*Fsize-cnt)
    # id2 = np.random.randint(0, cnt)
    id1s = np.zeros(njobs, dtype=np.int64)
    id2s = np.zeros(njobs, dtype=np.int64)

    for i in range(njobs):
        id1s[i] = np.random.choice(lsta)
        I = np.where(lsta%Fsize==id1s[i]%Fsize)[0]
        lsta = np.delete(lsta, I)
        id2s[i] = np.random.choice(lstb)
        I = np.where(lstb%Fsize==id2s[i]%Fsize)[0]
        lstb = np.delete(lstb, I)

    return id1s, id2s

@njit()
def Etrial(s, ida, idb):
    ia = ida//Fsize
    ja = ida%Fsize
    ib = idb//Fsize
    jb = idb%Fsize
    # segregation energy
    dEseg = Eseg[ia]-Eseg[ib]
    # bonds
    dEint = np.sum(w[ia]*(s[:, ja]+1)/2 - w[ib]*(s[:, jb]+1)/2)
    dE = dEseg + dEint
    return dE

@njit(parallel=True)
def step(s, T, njobs):
    idas, idbs = trial(s, njobs)
    cnt = 0
    dEtot = 0
    dEtr = 0
    for i in prange(njobs):
        dE = Etrial(s, idas[i], idbs[i])
        if acc(dE, T):
            s.ravel()[idas[i]]=1
            s.ravel()[idbs[i]]=-1
            cnt += 1
            dEtot += dE
        if dE>0:
            dEtr += dE
    return cnt
    
@njit(cache=True)
def X(s):
    bn = (s+1)/2 
    x = np.sum(bn, axis=1)/Fsize
    return x

def rolling_mean(x, w):
    if w<1:
        w = 1
    return np.convolve(x, np.ones(w), 'valid') / w

def setup_w(M, i_c, scale_p, p_p, scale_n, p_n):
    w_p = expon.rvs(scale=scale_p, size=int(M*(M-1)/2))
    w_n = -expon.rvs(scale=scale_n, size=int(M*(M-1)/2))
    return _setup_w(M, i_c, w_p, p_p, w_n, p_n)
    
    
@njit(cache=True)
def _setup_w(M, i_c, w_p, p_p, w_n, p_n):
    w = np.zeros((M, M)) # shape = (M, M), symmetric -> w[i] = [w_i0, w_i1, ..., w_i(M-1)] 
    k, m = 0, 0
    for i in range(M-1):
        if i==i_c:
            continue # skip bond with grain 
        for j in range(i+1, M):
            if i==i_c:
                continue # skip bond with grain 
            p = np.random.random()
            if p < p_p:
                w[i,j]=w_p[k]
                k+=1
                w[j,i]=w[i,j]
            elif p < p_p + p_n:
                w[i,j]=w_n[m]
                m+=1
                w[j,i]=w[i,j]
    return w

def mult_along_axis(A, B, axis):
    A = np.array(A)
    B = np.array(B)
    # shape check
    if axis >= A.ndim:
        raise np.AxisError(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
            )
    shape = np.swapaxes(A, A.ndim-1, axis).shape
    B_brc = np.broadcast_to(B, shape)
    B_brc = np.swapaxes(B_brc, A.ndim-1, axis)
    return A * B_brc

def tot_energy(s, Eseg, w):
    M = s.shape[0]
    Fsize = s.shape[1]
    bn = (s+1)/2
    e_seg = mult_along_axis(bn, Eseg, axis=0).sum()/Fsize/M
    e_w = 0
    for i in range(Fsize):
        e_w += np.sum(np.tensordot(bn[:, i],bn[:, i],axes=0)*w)/(Fsize*M)
    return (e_seg + e_w)

# def energy
"""
INPUT
"""

alpha, epsilon, sigma = (-2.7315304755986145, 2.295156064847707, 19.878111272210262)
scale_p = 11.670335685247002
p_p = 0.002576511886860863 # probability to have positive bond
scale_n = 11.979711764972944
p_n = 0.005198557511100361 # probability to have negative bond

    
M = 1000    # number of site-types


#%%
"""
INIT E_SEG, W
"""

Eseg = skewnorm.rvs(alpha, loc=epsilon, scale=sigma, size=M) # shape = (M,)
# with open('Eseg.dump', 'rb') as f:
#     Eseg = pickle.load(f)

srt = np.argsort(Eseg)
#i_c = np.where(srt==M-1)[0][0] # index of grain
Eseg = Eseg[srt]

#w = setup_w(M, -1, scale_p, p_p, scale_n, p_n) # shape = (M, M), symmetric -> w[i] = [w_i0, w_i1, ..., w_i(M-1)] 
with open('w_matrix_1000.dump', 'rb') as f:
    w = pickle.load(f)
    
# w = np.zeros((M, M))
# w[:M-1, :M-1] = wgb   
w = w[srt]
w = w[:, srt]
plt.hist(w[w!=0], bins=100)
plt.axvline(0.025*eV2kJmole*2, ymin=0, ymax=plt.gca().get_ylim()[1], 
            color='red')
plt.axvline(-0.025*eV2kJmole*2, ymin=0, ymax=plt.gca().get_ylim()[1], 
            color='red')
plt.xlabel('$\omega$, kJ/mol')
plt.show()
plt.hist(w.sum(axis=1))
plt.xlabel('$\omega_{avg}$, kJ/mol')
plt.show()
#%%
"""
MAIN
"""
Fsize = 10 # number of sites in type
T0 = 10000
Tf = 0
Xtot = 10/100
Nsteps = int(1e9)
print_each = 1000
save_each = 1000
plot_each = 10000
dump_each = 50000
T_each = plot_each*10
k_f = 0.8
k_s = 0.95
cv_c = 0.005
njobs = 4


import time
start = time.time()
#print(start)

s = -np.ones((M, Fsize))
cnt = init(Xtot, s, Fsize)
# with open('s.dump', 'rb') as f:
#     s = pickle.load(f)

accepteds = 0
accepteds0 = 0
steps = 0
flag = True
es = np.zeros((Nsteps//save_each))
ts = np.zeros((Nsteps//plot_each))
cv = np.zeros((Nsteps//plot_each))
de = 0
det = 0
kA = 300
T = T0

while T>Tf and steps<Nsteps:
    accepteds += step(s, T, njobs)/njobs
    
    if steps%print_each==0:
        print('step: ', steps)
        da = accepteds - accepteds0
        print('acceptance ratio: ', round(da/print_each,4))
        accepteds0 = accepteds
    if steps%save_each==0:
        n = steps//save_each
        es[n] = tot_energy(s, Eseg, w)
        
    if steps%plot_each==0:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(es[:n+1], label='tot')
        plt.legend(loc='upper left')
        plt.subplot(132)
        bn = (s+1)/2
        xs = bn.sum(axis=1)/bn.shape[1]
        srt = np.argsort(xs)[::-1]
        plt.plot(xs[srt])
        plt.subplot(133)
        eslice = es[(steps-plot_each)//save_each:n+1]
        cv[steps//plot_each] = eslice.var()/(kB*T)**2
        plt.plot(cv[:steps//plot_each+1])
        plt.hlines(cv_c, xmin=0, xmax=steps//plot_each, 
                   color='grey', linestyle='--')
        plt.twinx()
        ts[steps//plot_each] = T
        plt.plot(ts[:steps//plot_each+1], color='red')
        plt.gcf().tight_layout()
        plt.show()
    if steps%T_each==0 and steps>0:
        cvmax = cv[(steps-T_each)//plot_each:steps//plot_each+1].max()
        if cvmax >= cv_c:
            k = k_s
        else:
            k = k_f
        T *= k
    if steps%dump_each==0:
        with open('s.dump', 'wb') as f:
            pickle.dump(s, f)
    steps += 1

end = time.time()
print('steps: ', steps*njobs, '; time: ', end-start)
