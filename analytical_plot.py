import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc
from scipy.stats import skewnorm
from copy import deepcopy as dc
from itertools import chain, repeat
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal
Na = 6e23
kB = 1.380649e-23*Na*1e-3 #kJK-1/mol
eV2kJmole = 96.4915666370759


Wf = -117.4934383990136
Wwk = -123.37552565015118

Xgb = 10/100
W = 0#Wwk
T = 600

def X(E, wavg, T, m, xgb):
    #print(E)
    return 1/(1+np.exp((E-m+wavg*xgb)/(kB*T)))


def func(m, Fs, Es, w, T, xgb):
    xs = X(Es, w, T, m, xgb)
    return np.sum(xs*Fs) - xgb

def Xs(name):
    data = np.loadtxt(name)
    E_s = data[1:, 0]
    F_s = data[1:, 1].astype(int)
    Fs = F_s/np.sum(F_s)
    Ehist = np.array(list(chain.from_iterable(
        repeat(j, times = i) for i, j in zip(F_s, E_s))))
    m = fsolve(func, -100, args=(Fs, E_s, 0, T, Xgb))
    print(m)
    xs = X(Ehist, 0, T, m, Xgb)
    return xs

def Xs_hybr(name, ids_c_name, wavg_name):
    data = np.loadtxt(name)[1:]
    E_s = data[:, 0]
    F_s = data[:, 1].astype(int)
    Fs = F_s/np.sum(F_s)
    Ehist = np.array(list(chain.from_iterable(
        repeat(j, times = i) for i, j in zip(F_s, E_s))))
    data = np.loadtxt(ids_c_name)[1:]
    ids = np.sort(list(set(data[data!=-1])))
    wavg = np.loadtxt(wavg_name)[1:]
    _wavg = np.zeros(len(data))
    for i in range(len(data)):
        lst = data[i]
        lst = lst[lst!=-1]
        for e in lst:
            _wavg[i] += wavg[np.where(ids==e)]
    
    m = fsolve(func, -85, args=(Fs, E_s, _wavg, T, Xgb))
    print(m)
    #m = -200
    Whist = np.array(list(chain.from_iterable(
        repeat(j, times = i) for i, j in zip(F_s, _wavg))))
    xs = X(Ehist, Whist, T, m, Xgb)
    return xs, Whist

#names = ['E_F_rep_0K.txt', 'E_F_300K.txt', 'E_F_600K.txt']
#labels = ['analytical 0K full', 'analytical 300K attr', 'analytical 600K attr']
names = ['E_F_rep_0K.txt', 'E_F_600K.txt']
#ids_c_names = ['', 'ids_c_300K.txt']
ids_c_names = ['', '', '', '']
wavg_names = ['', 'wavg_300K.txt', '', '']
labels = ['$0\,\mathrm{K}$ spectrum', r'$600\,\mathrm{K}$ spectrum']
plt.figure(dpi=800)
for name, label, ids_c_name, wavg_name in zip(names, labels,
                                              ids_c_names,
                                              wavg_names):
    if ids_c_name == '':
        xs = Xs(name)
    else:
        xs, whist = Xs_hybr(name, ids_c_name, wavg_name)
        x600 = dc(xs)
        xs = np.sort(xs)[::-1]
    plt.plot(np.arange(len(xs))/len(xs), xs, label=label)
    
"""
weak approach
"""
mean = (-12.604490845213158, -148.22447733945143) # (Eseg, wavg)
cov = np.array([[  173.14793409,  -512.91940965],
                [ -512.91940965, 11227.81251111]])

es = np.linspace(-45, 20, num=101)
ws = np.linspace(-400, 100, num=100)

Es, Ws = np.meshgrid(es, ws)

r = multivariate_normal(mean=mean, cov=cov)

xs = np.dstack((Es, Ws))
de = (es[1]-es[0])
dw = (ws[1]-ws[0])
Fs = r.pdf(xs)*dw*de
print('accounted fraction of distribution:', Fs.sum())

def Xwk(E, w, T, m, xgb):
    return 1/(1+np.exp((E+2*xgb*w-m)/(kB*T)))

def func_wk(m, Fs, Es, Ws, T, xgb):
    xs = Xwk(Es, Ws, T, m, xgb)
    return np.sum(xs*Fs) - xgb


m = fsolve(func_wk, 1, args=(Fs, Es, Ws, T, Xgb))
ys = Xwk(Es, Ws, T, m, Xgb).ravel()
srt = np.argsort(ys)[::-1]
ys = ys[srt]
xs = np.cumsum(Fs.ravel()[srt])
plt.plot(xs, ys, label=r'random mixing $600\,\mathrm{K}$')

"""
MC data
"""


# xs_mc = np.loadtxt('xs_10.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 30')
# xs_mc = np.loadtxt('xs_10_fs1avg.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 1 avg')
# xs_mc = np.loadtxt('xs_10_fs100.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 100')
# xs_mc = np.loadtxt('xs_10_fs200.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 200')
xs_mc = np.loadtxt('xs_10_fs200avg.txt')
plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC')
# xs_mc = np.loadtxt('xs_10_fs400.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 400')
# xs_mc = np.loadtxt('xs_10_fs400_avg.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 600K 400 avg')
# xs_mc = np.loadtxt('xs_10_300K.txt')
# plt.plot(np.arange(len(xs_mc))/len(xs_mc), xs_mc, label='MC 300K 200')

plt.ylabel('concentration')
plt.xlabel('CDF($\mathrm{\Delta E^{seg}}$)')
plt.xticks([0, 1])
plt.xlim((0,1))
plt.legend()
plt.show()




