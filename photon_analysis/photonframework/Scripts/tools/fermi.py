#!/usr/bin/env python3

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, gammaincc
from scipy.special import gamma as scipyGamma
import re

def power_law(E, N, E0, index):   
    return N * (E/E0)**(index) 

def integrated_power_law(E,N,E0,index):
    return N * E*(E/E0)**index /(1+index)

def broken_power_law(E, gamma1, gamma2, Eb, N):
    return np.where(E < Eb, N*(E/Eb)**gamma1, N*(E/Eb)**gamma2)

def power_law2(E, N, gamma, Emin, Emax):
    return (N*(gamma+1)*E**gamma)/(Emax**(gamma+1) - Emin**(gamma+1))

def broken_power_law2(E, N, gamma1, gamma2, E_min, E_max, E_b):
    
    def integral_gamma1(E):
        return (E / E_b) ** gamma1

    def integral_gamma2(E):
        return (E / E_b) ** gamma2

    if E_max < E_b:
        norm_factor, _ = quad(integral_gamma1, E_min, E_max)
    elif E_min > E_b:
        norm_factor, _ = quad(integral_gamma2, E_min, E_max)
    else:
        norm1, _ = quad(integral_gamma1, E_min, E_b)
        norm2, _ = quad(integral_gamma2, E_b, E_max)
        norm_factor = norm1 + norm2

    N_0 = N / norm_factor  

    if np.isscalar(E):
        if E < E_b:
            return N_0 * (E / E_b) ** gamma1
        else:
            return N_0 * (E / E_b) ** gamma2
    else:
        return np.where(E < E_b, 
                        N_0 * (E / E_b) ** gamma1, 
                        N_0 * (E / E_b) ** gamma2)

def smooth_broken_power_law(E, N, E0, gamma1, gamma2, beta):
    return N*(E/E0)**gamma1 * (1+(E/E0)**((gamma1-gamma2)/beta))**(-beta)

def log_parabola(E, N, E0, alpha, beta):   
    return N * (E / E0)**(-alpha - beta*np.log(E/E0)) 

def integrated_log_parabola(E, N, E0, alpha, beta):
    return np.sqrt(np.pi) * E0 * N * np.exp((alpha-1)**2 / 4*beta) * erf((2*beta*np.log(E/E0)+alpha-1)/(2*np.sqrt(beta))) / (2*np.sqrt(beta))

def exp_cutoff(E, N, E0, gamma, Eb, p1, p2, p3):
    return np.where(E < Eb, (E/E0)**gamma, (E/E0)**gamma * np.exp(-((E-Eb)/p1+p2*np.log(E/Eb) + p3*np.log(E/Eb)**2)))

def b_pl_exp_cutoff(E, N, Eb, gamma1, gamma2, Eabs, p1):
    if E < Eb and E < Eabs:
        return N * (E/Eb)**gamma1
    elif E > Eb and E < Eabs:
        return N * (E/Eb)**gamma2
    elif E < Eb and E > Eabs:
        return N * (E/Eb)**gamma1 * np.exp(-(E-Eabs)/p1)
    elif E > Eb and E > Eabs:
        return N * (E/Eb)**gamma2 * np.exp(-(E-Eabs)/p1)
    
def pl_super_exp_cutoff(E, N, gamma, E0, Ec, b):
    return N*(E/E0)**gamma * np.exp(-(E/Ec)**b)

def integrated_pl_super_exp_cutoff(E, N, gamma, E0, Ec, b):
    return -(N*Ec**(gamma+1)*gammaincc((gamma+1)/b, E**b/Ec**b)*scipyGamma((gamma+1)/b))/(b*E0**gamma)

def pl_super_exp_cutoff2(E, N, E0, gamma, a, b):
    N*(E/E0)**gamma * np.exp(-a*E**b)

def pl_super_exp_cutoff3(E, N, E0, gamma, c, b):
    N*(E/E0)**(gamma+b*c) * np.exp(c*(1-(E/E0)**b))

def pl_super_exp_cutoff4(E, N, gamma, E0, d, b):
    a  =np.where(np.abs(b*np.log(E/E0)) < 1e-2, 
                    N*(E/E0)**(gamma-d/2*np.log(E/E0) - d*b/2*np.log(E/E0)**2 - (d*b**2)/ 24 * np.log(E/E0)**3),
                    N*(E/E0)**(gamma+d/b) * np.exp(d/b**2 * (1-(E/E0)**b)))
    return a

def integrated_super_exp_cutoff_4(e_low, e_up, N, E0, gamma, d, b):
    if hasattr(N, '__iter__'):
        fluxes = []
        for i in range(len(N)):
            fluxes.append(quad(pl_super_exp_cutoff4, e_low, e_up, args = (N[i],gamma[i],E0[i],d[i],b[i]))[0])
        return np.array(fluxes)
    else:
        return quad(pl_super_exp_cutoff4, e_low, e_up, args = (N,gamma,E0,d,b))[0]

def constant_value(E, N):
    return np.ones(len(E))*N

def gaussian(E, N, E_mean, sigma):
    return N/(sigma*np.sqrt(2*np.pi)) * np.exp((-(E-E_mean)**2)/(2*sigma**2))

def band_function(E, N, alpha, beta, Ep):
    return np.where(E<Ep*(alpha-beta)/(alpha+2), 
                    N*(E/0.1)**alpha * np.exp(-(E/Ep)*(alpha+2)),
                    N*(E/0.1)**beta * ((Ep/0.1)* (alpha-beta)/(alpha+2))**(alpha-beta) * np.exp(beta-alpha))

def pl_exp_cutoff(E, N, E0, Gamma, Ec, b):   
    return N * (E/E0)**(-Gamma) * np.exp((E0/Ec)**b - (E/Ec)**b) 

def pl_exp_cutoff_4FGL(E, N, E0, Gamma, a, b):   
    return N * (E/E0)**(-Gamma) * np.exp(a*(E0**b - E**b))

def source_functor(source):
    spectrum_type = source['SpectrumType']

    name = source['Source_Name']

    if 'FGL' in name:
        if spectrum_type == "PowerLaw":
            N = source["PL_Flux_Density"]
        elif spectrum_type == "LogParabola":
            N = source["LP_Flux_Density"]
        elif "ExpCutoff" in spectrum_type:
            N = source["PLEC_Flux_Density"]
    
    elif "FL14Y" in name:
        N = source['Flux_Density']
    else:
        raise NotImplementedError(f'Unexpected source name: {name}')
    
    N *= 1000.0                             # MeV to GeV
    E0 = source["Pivot_Energy"] / 1000.0    # MeV to GeV

    if spectrum_type == 'PowerLaw':
        index = source['PL_Index']
        return power_law, (N, E0, index)
    elif spectrum_type == 'LogParabola':
        alpha = source['LP_Index']
        beta = source['LP_beta']
        return log_parabola, (N, E0, alpha, beta)
    elif spectrum_type == 'PLSuperExpCutoff':
        gamma = source['PLEC_IndexS']
        E_c = source['PLEC_ExpfactorS']
        b = source['PLEC_Exp_Index']
        E_c *= (1000.0)
        return pl_super_exp_cutoff, (N, gamma, E0, E_c, b)
    
def source_flux(source, E):
    f, args = source_functor(source)
    return f(E, *args)

def integrated_source_flux(source, e_low, e_up):
    f, args = source_functor(source)
    return quad(f, e_low, e_up, args=args)[0]

def integrated_source_flux_no_quad(source, e_low, e_up):
    spectrum_type = source['SpectrumType']

    name = source['Source_Name']

    if 'FGL' in name:
        if spectrum_type == "PowerLaw":
            N = source["PL_Flux_Density"]
        elif spectrum_type == "LogParabola":
            N = source["LP_Flux_Density"]
        elif "ExpCutoff" in spectrum_type:
            N = source["PLEC_Flux_Density"]
    
    elif "FL14Y" in name:
        N = source['Flux_Density']
    else:
        raise NotImplementedError(f'Unexpected source name: {name}')
    
    N *= 1000.0                             # MeV to GeV
    E0 = source["Pivot_Energy"] / 1000.0    # MeV to GeV

    if spectrum_type == 'PowerLaw':
        index = -source['PL_Index']
        return integrated_power_law(e_up, N, E0, index) - integrated_power_law(e_low, N, E0, index)
    elif spectrum_type == 'LogParabola':
        alpha = source['LP_Index']
        beta = source['LP_beta']
        return integrated_log_parabola(e_up, N, E0, alpha, beta) - integrated_log_parabola(e_low, N, E0, alpha, beta)
    elif spectrum_type == 'PLSuperExpCutoff':
        gamma = source['PLEC_IndexS']
        E_c = source['PLEC_ExpfactorS']
        b = source['PLEC_Exp_Index']
        E_c *= (1000.0)
        return integrated_pl_super_exp_cutoff(e_up, N, gamma, E0, E_c, b) - integrated_pl_super_exp_cutoff(e_low, N, gamma, E0, E_c, b)

def integrated_source_flux_power_law(sources, e_low, e_up):
    e_low_array = np.ones(len(sources))*e_low
    e_up_array = np.ones(len(sources))*e_up
    N = sources["PL_Flux_Density"]
    N *= 1000.0                             # MeV to GeV
    E0 = sources["Pivot_Energy"] / 1000.0    # MeV to GeV
    index = sources['PL_Index']
    return integrated_power_law(e_up, N, E0, index) - integrated_power_law(e_low, N, E0, index)

def integrated_source_flux_power_law_xml(Parameters, e_low, e_up):
    e_low_array = np.ones(len(Parameters))*e_low
    e_up_array = np.ones(len(Parameters))*e_up
    N = Parameters["Prefactor"]
    N *= 1000.0                             # MeV to GeV
    E0 = Parameters["Scale"] / 1000.0    # MeV to GeV
    index = Parameters['Index']
    return integrated_power_law(e_up, N, E0, index) - integrated_power_law(e_low, N, E0, index)

def integrated_source_flux_log_parabola(sources, e_low, e_up):
    e_low_array = np.ones(len(sources))*e_low
    e_up_array = np.ones(len(sources))*e_up
    N = sources["PL_Flux_Density"]
    N *= 1000.0                             # MeV to GeV
    E0 = sources["Pivot_Energy"] / 1000.0    # MeV to GeV
    alpha = sources['LP_Index']
    beta = sources['LP_beta']
    return integrated_log_parabola(e_up, N, E0, alpha, beta) - integrated_log_parabola(e_low, N, E0, alpha, beta)

def integrated_source_flux_log_parabola_xml(Parameters, e_low, e_up):
    e_low_array = np.ones(len(Parameters))*e_low
    e_up_array = np.ones(len(Parameters))*e_up
    N = Parameters["norm"]
    N *= 1000.0                             # MeV to GeV
    E0 = Parameters["Eb"] / 1000.0    # MeV to GeV
    alpha = Parameters['alpha']
    beta = Parameters['beta']
    return integrated_log_parabola(e_up, N, E0, alpha, beta) - integrated_log_parabola(e_low, N, E0, alpha, beta)


def integrated_source_flux_pl_super_exp_cutoff(sources, e_low, e_up):
    e_low_array = np.ones(len(sources))*e_low
    e_up_array = np.ones(len(sources))*e_up
    N = sources["PL_Flux_Density"]
    N *= 1000.0                             # MeV to GeV
    E0 = sources["Pivot_Energy"] / 1000.0    # MeV to GeV
    gamma = -sources['PLEC_Index']
    E_c = sources['PLEC_Expfactor']
    b = sources['PLEC_Exp_index']
    E_c *= (1000**b)
    return integrated_pl_super_exp_cutoff(e_up, N, gamma, E0, E_c, b) - integrated_pl_super_exp_cutoff(e_low, N, gamma, E0, E_c, b)

def integrated_source_flux_pl_super_exp_cutoff_xml(Paramters, e_low, e_up):
    e_low_array = np.ones(len(Paramters))*e_low
    e_up_array = np.ones(len(Paramters))*e_up
    N = Paramters["Prefactor"]
    N *= 1000.0                             # MeV to GeV
    E0 = Paramters["Scale"] / 1000.0    # MeV to GeV
    gamma = Paramters['IndexS']
    E_c = Paramters['ExpfactorS']
    b = Paramters['Index2']
    E_c *= 1/(1000.0)
    return integrated_pl_super_exp_cutoff(e_up, N, gamma, E0, E_c, b) - integrated_pl_super_exp_cutoff(e_low, N, gamma, E0, E_c, b)

def integrated_source_flux_pl_super_exp_cutoff_4_xml(Parameters, e_low, e_up):
    e_low_array = np.ones(len(Parameters))*e_low
    e_up_array = np.ones(len(Parameters))*e_up
    N = Parameters["Prefactor"]
    N *= 1000.0                             # MeV to GeV
    E0 = Parameters["Scale"] / 1000.0    # MeV to GeV
    gamma = Parameters['IndexS']
    d = Parameters['ExpfactorS']
    b = Parameters['Index2']

    return integrated_super_exp_cutoff_4(e_low, e_up, N, E0, gamma, d, b)

def averaged_source_flux(source, e_low, e_up):
    return integrated_source_flux(source, e_low, e_up) / (e_up - e_low)

def find_source(catalog, source_name):
    matches = []
    print(source_name)
    for s in catalog:
        full_name = s['Source_Name']
        assoc1 = s['ASSOC1']
        assoc2 = s['ASSOC2']
        if source_name in full_name or source_name in assoc1 or source_name in assoc2:
            print(source_name, full_name, assoc1, assoc2)
            matches.append(s)
    if len(matches) > 1:
        raise NotImplementedError('Multiple matches')
    if len(matches) == 0:
        raise NotImplementedError('No matches')
    return matches[0]

def find_source_reg_exp(catalog, regexp):
    matches = []
    regexp = re.compile(regexp)
    for s in catalog:
        full_name = s['Source_Name']
        assoc1 = s['ASSOC1']
        assoc2 = s['ASSOC2']
        if regexp.search(full_name) or regexp.search(assoc1) or regexp.search(assoc2):
            matches.append(s)
    if len(matches) != 1:
        raise NotImplementedError('Multiple matches')
    return matches[0]
        


