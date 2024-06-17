import numpy as np 
from matplotlib import pyplot as plt
from scipy.special import erfc
from numba import njit, prange
from scipy.optimize import fsolve
from scipy.stats import skewnorm, expon, linregress
import pickle
import numba_mpi as comm
from mpi4py import MPI
import time

Ncpu = comm.size()
rank = comm.rank()
rng = np.random.default_rng(1234)

Na = 6e23
kB = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759

@njit(cache=True)
def prob_loc(dE, dN, mu, T):
    return np.exp(-(dE+dN*mu)/(kB*T)) #min(1, exp) however p always in [0, 1) so if exp>1 prob = 1

@njit(cache=True)
def prob_glob(N, kappa, dc, c, c0):
    return np.exp(-kappa*dc*(dc+2*(c-c0))) #min(1, exp) however p always in [0, 1) so if exp>1 prob = 1

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
def acc_glob(N, kappa, dc, c, c0, rng):
    p = rng.random()
    if p<prob_glob(N, kappa, dc, c, c0):
        return True
    else:
        return False
    
@njit#(cache=True)
def step(s, T, mu, kappa, c0, Eseg, w, Fsize, Ntot, rng):
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
    comm.allreduce(np.array([dN], dtype=np.int32), dNtot, 
    comm.Operator.SUM)
    dc = dNtot/N
    c = Ntot/N
    
    if acc_glob(N, kappa, dc, c, c0, rng):
        s[i,j] += dN*2
        Ntot += dNtot[0]
    else: 
        dE = 0
        accepted = 0
    return accepted, Ntot, dE
    
Fsize = 4
M = 1000
Xtot = 10/100

Nsites = Fsize*M

T = 2000
Tf = 100
mu = 79
n_mu = 10 #adjust mu every plot_each*n_mu
dmu = 10 #step for adjusting mu (mu = mu + (c-c0)*dmu/100)

kappa = 10000
n_kappa = 100 #adjust kappa every plot_each*n_kappa
dkappa = np.sqrt(10) # kappa = kappa *dkappa
ac_treshold = 0.05 # decrease kappa if acc_ratio < ac_treshold
cv_treshold = 5 # increase kappa if std > cv_treshold

Neq = int(2e6)
Ncool = int(2e6)
Nstd = int(2e6)
T_each = Neq+Ncool+Nstd
k_f = 2e-1/Ncool
k_s = 1e-1/Ncool
estd_c = 0.4

Nsteps = 1e8
print_each = 1000
save_each = 1000
plot_each = 10*save_each
dump_each = 100000

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

continue_from_dump = False
continue_from_dump = True

if continue_from_dump:
    with open('s.dump', 'rb') as f:
        stot = pickle.load(f) 
    s = stot[:,partition]
    Ntot = np.int32(np.sum((stot+1)/2))
    with open('params.dump', 'rb') as f:
         (T, mu, kappa) = pickle.load(f) 

else:
    s, cnt = init(Xtot, M, Fp)
    Ntot = np.empty(1, dtype=np.int32)
    comm.allreduce(np.array([cnt], dtype=np.int32), Ntot, comm.Operator.SUM)

accepteds = 0
steps = 0
E = 0
Etot = 0
if rank == 0:
    es = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    ts = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    cs = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    acs = np.zeros(int(Nsteps//save_each+1), dtype=np.float64)
    
cooling = False
while T>Tf:
    accepted, Ntot, dE = step(s, T, mu, kappa, c0, 
    Eseg, w, Fsize, Ntot, rng)
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
        comm.allreduce(accepteds, accepteds_tot, 
        comm.Operator.SUM)
        accepteds = 0
        if rank == 0:
            Etot += dEtot
            n = steps//save_each
            es[n] = Etot/Ntot
            cs[n] = 100*Ntot/Nsites
            acs[n] = accepteds_tot/save_each/Ncpu
            ts[n] = T
            
    if steps%plot_each==0:
        xs = np.sum((s+1)/2, axis=1).astype(np.float32)
        xstot = np.empty(M, dtype=np.float32)
        MPI.COMM_WORLD.Reduce([xs, M], [xstot, M],
        op=MPI.SUM, root=0)

        if rank == 0 and steps>0:
            n = steps//save_each
            n0 = (steps-plot_each*n_mu)//save_each
            n1 = (steps-plot_each*n_kappa)//save_each
            cm = cs[n0:n+1].mean()
            cv = cs[n1:n+1].std()
            acm = acs[n1:n+1].mean()
            if steps%int(plot_each*n_mu)==0:
                mu = mu+(cm/100-c0)*dmu
            if steps%int(plot_each*n_kappa)==0:
                if cv>cv_treshold:
                    kappa *= dkappa
                elif acm<ac_treshold:
                    kappa /= dkappa

            scale = 2
            plt.figure(figsize=(15/scale, 5/scale))
            plt.subplot(131)
            plt.plot(ts[:n+1], color='red')
            plt.twinx()
            plt.plot(es[:n+1])
            plt.subplot(132)
            xstot = xstot/Fsize
            plt.plot(np.sort(xstot)[::-1])
            plt.subplot(133)
            plt.plot(acs[:n+1], color='red')
            plt.twinx()
            plt.plot(cs[:n+1])
            plt.text(n*0.5, cs.max()*0.5, f'std {round(cv, 3)}')
            plt.text(n*0.2, cs.max()*0.1, f'kappa {round(kappa, 0)}')
            plt.text(n*0.5, cs.max()*0.3, f'mu {round(mu, 2)}')
            plt.gcf().tight_layout()
            plt.savefig(f'plots/step.png')
            plt.close()
        if steps%int(plot_each*n_mu)==0:
            mu = MPI.COMM_WORLD.bcast(mu, root=0)
        if steps%int(plot_each*n_kappa)==0:
            kappa = MPI.COMM_WORLD.bcast(kappa, root=0)

    if steps%T_each==0 and steps>0:
        cooling = True
        cool_step = 0
        if rank == 0:
            n0 = (steps-Nstd)//save_each
            n = steps//save_each
            estd = es[n0:n+1].std()
            with open('estd.txt', 'a') as f:
                f.write(f'{steps} {T} {estd}\n')
        else:
            estd = None
        estd = MPI.COMM_WORLD.bcast(estd, root=0)
        if estd>= estd_c:
            k = k_s
        else:
            k = k_f
    if cooling:
        if cool_step < Ncool:
            T *= (1-k)
            cool_step += 1
        else:
            cooling = False

    if steps%dump_each==0:
        stot = np.empty((M,Fsize), dtype=np.int8)
        MPI.COMM_WORLD.Gather(s, stot, root=0)
        if rank==0:
            with open('s.dump', 'wb') as f:
                pickle.dump(stot, f)
            with open('params.dump', 'wb') as f:
                pickle.dump((T, mu, kappa), f)
        
        
        
        
        
        
        
        
        
