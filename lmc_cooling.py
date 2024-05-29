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
    for i in prange(njobs):
        dE = Etrial(s,idas[i], idbs[i])
        if acc(dE, T):
            s.ravel()[idas[i]]=1
            s.ravel()[idbs[i]]=-1
            cnt += 1
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
    return (e_seg + e_w), e_seg

def read_conv_conf(fname):
    with open(fname, 'r') as f :
        flag_w, flag_p, flag_s, flag_st = False, False, False, False
        for line in f:
            if 'width' in line:
                width = int(line.split()[-1])
                flag_w=True
            if 'points for convergence' in line:
                Npoints = int(line.split()[-1])
                flag_p=True
            if 'steps' in line:
                steps = int(line.split()[-1])
                flag_st=True
            if 'converged slope' in line:
                slope_conv = float(line.split()[-1])
                flag_s=True
            
    flag = (flag_w and flag_p and flag_s and flag_st)
    if not flag:
        print(flag_w, flag_p, flag_s, flag_st)
        raise ValueError(f'incorrect {fname}')
    return width, Npoints, steps, slope_conv

def slope(es, s1, w):
    sl = slice(s1, s1+w)
    x = np.arange(0, w)
    y = es[sl]
    res = linregress(x, y)
    return res.slope

def slopes(es, st, w):
    s1s = np.arange(0, es.shape[0]-w, st)
    slps = np.zeros(len(s1s))
    if len(s1s)<1:
        return np.zeros(1)
    for i in range(len(s1s)):
        slps[i] = slope(es, s1s[i], w)
    return slps
    

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

Eseg = np.concatenate((Eseg, [0]))
M = len(Eseg)
srt = np.argsort(Eseg)
#i_c = np.where(srt==M-1)[0][0] # index of grain
Eseg = Eseg[srt]

w = setup_w(M, -1, scale_p, p_p, scale_n, p_n)/3 # shape = (M, M), symmetric -> w[i] = [w_i0, w_i1, ..., w_i(M-1)] 
# with open('w_matrix_1000.dump', 'rb') as f:
#     wgb = pickle.load(f)
    
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
Fsize = 200 # number of sites in type
T = 600
dT = 10
Nsteps = int(1e8)
Xtot = 10/100
print_each = 50000
save_each = 10000
plot_each = 10000
dump_each = 50000
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
es = np.zeros((Nsteps//plot_each))
es0 = np.zeros((Nsteps//plot_each))

N_conv = 0
N_conv_tot = 0
while T>0:
    accepteds += step(s, T, njobs)/njobs
    
    if steps%print_each==0:
        print('step: ', steps)
        da = accepteds - accepteds0
        print('acceptance ratio: ', round(da/print_each,4))
        accepteds0 = accepteds
    if steps%plot_each==0:
        plt.figure(figsize=(15, 5))
        n = steps//plot_each
        es[n], es0[n] = tot_energy(s, Eseg, w)
        plt.subplot(131)
        plt.plot(es[:n+1], label='tot')
        plt.legend(loc='upper left')
        plt.twinx()
        plt.plot(es0[:n+1], color='red', label='non-int')
        plt.legend(loc='upper right')
        plt.subplot(132)
        width, Npoints, st, slope_conv = read_conv_conf('convergence_conf.txt')
        slps = slopes(es[:n+1], st, width)
        if np.any(slps):
            plt.plot(slps, '.')
            _N_conv_tot = np.sum(np.abs(slps)<=slope_conv)
            if _N_conv_tot > N_conv_tot:
                N_conv += (_N_conv_tot-N_conv_tot)
            N_conv_tot = _N_conv_tot
            if len(slps)>0:
                if slps[-1]>slope_conv:
                    N_conv = 0
        print('N convergence: ', N_conv)
        if N_conv>=Npoints:
            T-=dT
            accepteds = 0
            accepteds0 = 0
            steps = -1
            es = np.zeros((Nsteps//plot_each))
            es0 = np.zeros((Nsteps//plot_each))
            N_conv = 0
            N_conv_tot = 0
            print('New temperature: ', T)
        plt.hlines(slope_conv, 0, max(1, len(slps)), color='grey', linestyle='--')
        plt.hlines(-slope_conv, 0, max(1, len(slps)), color='grey', linestyle='--')
        plt.subplot(133)
        bn = (s+1)/2
        xs = bn.sum(axis=1)/bn.shape[1]
        srt = np.argsort(xs)[::-1]
        plt.plot(xs[srt])
        plt.gcf().tight_layout()
        plt.show()
    if steps%dump_each==0:
        with open('s.dump', 'wb') as f:
            pickle.dump(s, f)
    steps += 1
    
print('')
print('avg acceptance ratio: ', round(accepteds/steps,4))
print('')
end = time.time()
print('steps: ', steps*njobs, '; time: ', end-start)

#%%
# with open('s.dump', 'rb') as f:
#     s = pickle.load(f)
    
from copy import deepcopy as dc
from itertools import chain, repeat


#s = ss[66712]
#s_selected = ss[30000:67712:1000]
s_selected = [ss[67712]]
for s in s_selected:
    bn = (1+s)/2
    xs = bn.sum(axis=1)/Fsize
    srt = np.argsort(xs)[::-1]
    #xs = xs[srt]
    # plt.plot(xs[srt])
    # plt.show()
    

    
    lenght = len(xs[xs!=0])
    xs = xs[srt]#[:lenght]
    plt.plot(xs)
    
    # kT = 0.025*eV2kJmole*600/300
    # wsorted = w[srt][:lenght]
    # wsorted = wsorted[:, srt][:, :lenght]
    # wsorted[np.abs(wsorted)<2*kT] = 0
    
    
    
    
    # Er = Eseg[srt][:lenght] + np.sum(np.tril(wsorted, -1), axis=1)
    
    
    # Xs = list(set(xs))
    # Erc = np.zeros(len(Xs))
    # Fs = np.ones(len(Xs)).astype(int)
   
    # Xs = list(set(xs))
    # for i in range(len(Xs)):
    #     msk = (xs==Xs[i])
    #     Erc[i] = Er[msk].mean()
    #     Fs[i] = np.sum(msk)
        
    # plt.plot(Er)
    # plt.show()
    
    # Fs = np.ones(len(Er)).astype(int)
    # Erc = dc(Er)
    # flag = True
    # while flag:
    #     flag = False
    #     i = 1
    #     while Fs[i] != 0:
    #         if Erc[i]<=Erc[i-1]:
    #             flag = True
    #             Erc[i-1] = (Erc[i-1]*Fs[i-1]+Erc[i]*Fs[i])/(Fs[i-1]+Fs[i])
    #             Erc[i:-1] = Erc[i+1:]
    #             Erc[-1] = 0
    #             Fs[i-1] += Fs[i]
    #             Fs[i:-1] = Fs[i+1:]
    #             Fs[-1] = 0
                
    #         else:
    #             i+=1
                
    # plt.plot(Erc)
    # plt.show()
    # Ehist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(Fs, Erc)))
    # #Ehist = np.sort(Er)
    # plt.plot(Ehist)
    

plt.show()
# plt.hist(Ehist, density=True)
# params = skewnorm.fit(Ehist)
# Es = np.linspace(np.min(Ehist), np.max(Ehist))
# plt.plot(Es, skewnorm.pdf(Es, *params))





