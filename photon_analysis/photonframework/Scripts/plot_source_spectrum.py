import numpy as np
import matplotlib.pyplot as plt

from tools.fermi import pl_super_exp_cutoff4
from tools.fits_tools import read_fits_file
# N, gamma, E0, d, b
Vela = [2.638452015e-10*1.782054164*1000, -2.226355939, 1.860980729, 5.625108103*0.1, 0.4922263025]
Geminga = [8.757223793e-11*3.957524627*1000, -1.986491957, 1.605283438, 6.856774801*0.1, 0.6806318702]

def test(E, N, gamma, E0, d, b):#
    return N*(E/E0)**(gamma+d/b) * np.exp(d/b**2 * (1-(E/E0)**b))
    return (gamma-d/2*np.log(E/E0) - d*b/2*np.log(E/E0)**2 - (d*b**2)/ 24 * np.log(E/E0)**3)

sources_hdu = read_fits_file('data/gll_psc_v22.fit')
sources = sources_hdu.data
Vela_fit = sources[sources.Source_Name == '4FGL J0835.3-4510']
Geminga_fit = sources[sources.Source_Name == '4FGL J0633.9+1746']
Vela_fit_para = [Vela_fit['PLEC_Flux_Density']*1000, -Vela_fit['PLEC_Index'], Vela_fit['Pivot_Energy']/1000, Vela_fit['PLEC_Expfactor']*1000**Vela_fit['PLEC_Exp_index'], Vela_fit['PLEC_Exp_index']]
Geminga_fit_para = [Geminga_fit['PLEC_Flux_Density']*1000, -Geminga_fit['PLEC_Index'], Geminga_fit['Pivot_Energy']/1000, Geminga_fit['PLEC_Expfactor']*1000**Geminga_fit['PLEC_Exp_index'], Geminga_fit['PLEC_Exp_index']]

print(Vela_fit['PLEC_Index'])

fig, ax = plt.subplots()

E = np.linspace(0.1,1000, 10000)
ax.plot(E, test(E, *Vela), label = 'Vela')
ax.plot(E, test(E, *Geminga), label = 'geminga')
ax.plot(E, test(E, *Vela_fit_para), label = 'Vela_fit')
ax.plot(E, test(E, *Geminga_fit_para), label = 'geminga_fit')
ax.legend()

ax.semilogx()
ax.semilogy()


plt.show()