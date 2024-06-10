import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import erfc
from numba import njit, prange
from scipy.optimize import fsolve
from scipy.stats import skewnorm, expon, linregress
import pickle
import numba_mpi as comm

Ncpu = comm.size()
rank = comm.rank()

Na = 6e23
kB = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759

@njit(cache=True)
def prob_loc(dE, dN, mu, T):
    return np.exp(-(dE+dN*mu)/(kB*T)) #min(1, exp) however p always in [0, 1) so if exp>1 prob = 1

@njit(cache=True)
def prob_glob(N, kappa, dc, c, c0, T):
    return np.exp(-kappa*N*N*dc*(dc-2*(c-c0))/(kB*T)) #min(1, exp) however p always in [0, 1) so if exp>1 prob = 1

def init(Xtot, M, Fp):
    s = -np.ones((M, Fp), dtype=np.int8)
    cnt = 0
    for i in range(M):
        for j in range(Fp):
            sij = np.random.choice([-1,1], p=[1-Xtot, Xtot])
            s[i][j] = sij
            if sij == 1:
                cnt += 1
            if cnt == round(Xtot*M*Fp):
                break
        if cnt == round(Xtot*M*Fp):
            break
    return s, cnt

@njit(cache=True)
def Etrial(s, i, j, Eseg, w):
    bn = (s[:,j]+1)/2
    return -(Eseg[i] + np.sum(w[i]*bn))*s[i,j]

@njit(cache=True)
def acc_loc(dE, dN, mu, T):
    p = np.random.random()
    if p<prob_loc(dE, dN, mu, T):
        return True
    else:
        return False
    
@njit(cache=True)
def acc_glob(N, kappa, dc, c, c0, T):
    p = np.random.random()
    if p<prob_glob(N, kappa, dc, c, c0, T):
        return True
    else:
        return False
    
@njit#(cache=True)
def step(s, T, mu, kappa, c0, Eseg, w, Fsize, Ntot):
    M, Fp = s.shape
    i = np.random.randint(0, M)
    j = np.random.randint(0, Fp)
    dN = -s[i,j]
    dE = Etrial(s, i, j, Eseg, w)
    if acc_loc(dE, dN, mu, T):
        accepted = 1
    else:
        accepted = 0
        dN = 0
        dE = 0
    N = Fsize*M
    dNtot = np.empty(1, dtype=np.int32)
    comm.allreduce(np.array([dN], dtype=np.int32), dNtot, comm.Operator.SUM)
    dc = dNtot/N
    c = Ntot/N
    
    if acc_glob(N, kappa, dc, c, c0, T):
        s[i,j] += dN
        Ntot += dNtot
    else: 
        accepted = 0
    return accepted, Ntot, dE/N
    
Fsize = 16
M = 1000
Xtot = 10/100

Nsites = Fsize*M

T = 5000
mu = 100
kappa = 10

Nsteps = 1e8
print_each = 1000
save_each = 1000
plot_each = 10*save_each

c0 = Xtot

alpha, epsilon, sigma = (-2.7315304755986145, 2.295156064847707, 19.878111272210262)
Eseg = skewnorm.rvs(alpha, loc=epsilon, scale=sigma, size=M) # shape = (M,)
# with open('Eseg.dump', 'rb') as f:
#     Eseg = pickle.load(f)

srt = np.argsort(Eseg)
#i_c = np.where(srt==M-1)[0][0] # index of grain
Eseg = Eseg[srt]

with open('w_matrix_1000.dump', 'rb') as f:
    w = pickle.load(f)
w = w[srt]
w = w[:, srt]
    
partitions = np.arange(Fsize)%Ncpu
mask = (partitions==rank)
partition = np.arange(Fsize)[mask]
Fp = partition.shape[0]
print(f'rank {rank}; partition {partition}')

s, cnt = init(Xtot, M, Fp)
Ntot = np.empty(1, dtype=np.int32)
comm.allreduce(np.array([cnt], dtype=np.int32), Ntot, comm.Operator.SUM)

accepteds = 0
steps = 0
E = 0
Etot = 0
if rank == 0:
    es = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    cs = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    acs = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    
while steps<Nsteps:
    accepted, Ntot, dE = step(s, T, mu, kappa, c0, Eseg, w, Fsize, Ntot)
    accepteds += accepted
    steps += 1
    E += dE
    # if steps%print_each==0:
    #     print(f'rank {rank}: {round(accepteds/steps, 4)}')
    if steps%save_each==0:
        dEtot = np.empty(1, dtype=np.float64)
        comm.allreduce(np.array([E], dtype=np.float64), dEtot, comm.Operator.SUM)
        E = 0
        accepteds_tot = np.empty(1, dtype=np.int32)
        comm.allreduce(accepteds, accepteds_tot, comm.Operator.SUM)
        accepteds = 0
        if rank == 0:
            Etot += dEtot
            n = steps//save_each
            es[n] = Etot
            cs[n] = 100*Ntot/Nsites
            acs[n] = accepteds_tot/save_each/Ncpu
            
    if steps%plot_each==0:
        xs = np.sum((s+1)/2, axis=1)
        comm.allreduce(xs, xs, comm.Operator.SUM)
        xs = xs/Fsize

        if rank == 0:
            scale = 2
            plt.figure(figsize=(15/scale, 5/scale))
            plt.subplot(131)
            plt.plot(es[:n+1])
            plt.subplot(132)
            plt.plot(np.sort(xs)[::-1])
            plt.subplot(133)
            plt.plot(cs[:n+1])
            plt.twinx()
            plt.plot(acs[:n+1], color='red')
            plt.gcf().tight_layout()
            plt.savefig(f'plots/step.png')
            plt.close()
        
        
        
        
        
        
        
        
        
