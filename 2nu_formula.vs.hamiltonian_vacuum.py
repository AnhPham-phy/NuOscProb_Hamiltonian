"""Compute the oscillation probability P(ν_mu -> ν_tau) in normal mass ordeing, delta CP = 0.

Flavour: 0=e, 1=mu, 2=tau
Parameters:
  L : float or array
    Baseline in km
  E : float
    Neutrino energy in GeV
  delta_m2 : float
    Mass difference squared in eV^2
  theta : float
    Mixing angle in radians

  Returns:
    P : float or array
      Oscillation/Survival probability
"""

import numpy as np
from scipy.linalg import expm   

# Experiment's parameters 
exp = "MINOS"
L = 735  # km
E = 3  # GeV
delta_m2 = 2.51e-3  # eV^2
theta23 = np.radians(42.1) #(NuFIT 5.1, NH)

# I.1.1 2nu oscillation in vacuum using analytical formula [1]
def P_2nu_vacuum_analysis(L, E, delta_m2, theta):
  hbarc = 197.3269804e-9     # eV·m
  L_natural_unit = (L * 1e3) / hbarc
  E_eV = E * 1e9
  phase = delta_m2 * L_natural_unit / (4*E_eV)
  probability = np.sin(2*theta)**2 * np.sin(phase)**2
  return probability

#I.1.2 2nu oscillation in vacuum using time-evolutor
def P_2nu_vacuum_evolutor(L, E, delta_m2, theta):
    # Constants (natural unit)
    hbarc = 1.973269804e-7     # eV·m
    L_natural_unit = (L * 1e3) / hbarc   # eV^-1 = hbar.c / 1eV = 1.97x10^-7
    E_eV = E * 1e9
    # Hamiltonian in mass basis (only depends on mass difference squared => set m1 = 0)
    H_mass = np.diag([0, delta_m2/(2*E_eV)])
    # Mixing matrix
    U = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    # Hamiltonian in flavor basis
    H_flavor = U @ H_mass @ U.conj().T
    # Time-evolution operator
    U_t = expm(-1j * H_flavor * L_natural_unit)
    # Initial and final states
    psi_mu = np.array([1, 0])
    psi_tau = np.array([0, 1])
    # Evolve state
    psi_t = U_t @ psi_mu
    amplitude = np.vdot(psi_tau, psi_t)
    probability = np.abs(amplitude)**2

    return probability

P_oscillation_anal = np.round(P_2nu_vacuum_analysis(L, E, delta_m2, theta23), 10)
P_survival = 1 - P_oscillation_anal
print(f"P(ν_mu -> ν_tau) in vacuum for {exp} experiment (analytical formula):", P_oscillation_anal)
print("Survival probability:", P_survival)

P_oscillation_evol = np.round(P_2nu_vacuum_evolutor(L, E, delta_m2, theta23), 10)
P_survival = 1 - P_oscillation_evol
print(f"P(ν_mu -> ν_tau) in vacuum for {exp} experiment (time-evolution):", P_oscillation_evol)
print("Survival probability:", P_survival)

# Absolute difference between analytical and exact results
diff = P_oscillation_anal - P_oscillation_evol
print(f"Difference of analytical formula vs Hamiltonian_2nu_{exp}: ", diff)