"""
Parameters:
    L : float
        Baseline in km
    E : float
        Neutrino energy in GeV
    U : 3x3 complex ndarray
        PMNS mixing matrix
    delta_m2 : list or array of float
        Mass-squared differences in eV^2
    alpha : int
        Initial flavor (0=e, 1=mu, 2=tau)
    beta : int
        Final flavor (0=e, 1=mu, 2=tau)

Returns:
    P : float
        Oscillation probability
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Experiment's parameters
exp = "DUNE"
L = 1285                               # km
E_range = np.linspace(0.1, 10, 500)     # GeV
# Additional matter potential term
G_F = 1.1663787e-5     # GeV^-2
rho = 2.825            # g/cm^3
Ye = 0.5
Na = 6.02214076e23     # mol^-1
# PMNS parameters (NuFIT 5.1, NH)
theta12 = np.radians(33.45)
theta23 = np.radians(42.1)
theta13 = np.radians(8.62)
delta_cp = 0.7 * np.pi
s12, c12 = np.sin(theta12), np.cos(theta12)
s23, c23 = np.sin(theta23), np.cos(theta23)
s13, c13 = np.sin(theta13), np.cos(theta13)
# Mass-squared differences (NuFIT 5.1, NH)
Dmsq21 = 7.42e-5  # eV^2
Dmsq31 = 2.515e-3 # eV^2

# PMNS matrix
U = np.array([
    [ c12*c13,               s12*c13,            s13*np.exp(-1j*delta_cp)],
    [-s12*c23 - c12*s23*s13*np.exp(1j*delta_cp),  c12*c23 - s12*s23*s13*np.exp(1j*delta_cp), s23*c13 ],
    [ s12*s23 - c12*c23*s13*np.exp(1j*delta_cp), -c12*s23 - s12*c23*s13*np.exp(1j*delta_cp), c23*c13 ]
], dtype=complex)

#I.2.2 3nu_matter_evolutor_NH
def P_3nu_evolutor(L, E, U, delta_m2, rho, alpha, beta): 
	hbarc = 197.3269804e-9     # eV.m
	L_natural_unit = (L * 1e3) / hbarc   # eV^-1
	E_eV = E * 1e9
	dm21, dm31 = delta_m2
	dm32 = dm31 - dm21
	dm2 = [0, dm21, dm31]  # dm11^2=0 reference
	H_mass = np.diag([0, dm21/(2*E_eV), dm31/(2*E_eV)])
	H_flavor = U @ H_mass @ U.conj().T
    # Additional potential term
	V = np.sqrt(2) * G_F * rho * Ye * Na * 7.645373e-33 # eV
	#V = np.sqrt(2) * G_F * 5.966551932e-9   # for approx using proton mass
	V_matter = np.diag([V, 0, 0])
	H_flavor += V_matter
    # Time evolution
	U_t = expm(-1j * H_flavor * L_natural_unit)
    # Flavor states
	psi_alpha = np.zeros(3, dtype=complex)
	psi_alpha[alpha] = 1
	psi_beta = np.zeros(3, dtype=complex)
	psi_beta[beta] = 1
    # Evolve and calculate probability
	psi_t = U_t @ psi_alpha
	amplitude = np.vdot(psi_beta, psi_t)
	probability = np.abs(amplitude)**2
	
	return probability

# NuFast
#eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
#YerhoE2a = 1.52e-4
eVsqkm_to_GeV_over4 = (1e3 / 197.3269804e-9) / (4 * 1e9)
YerhoE2a = 1.51891739e-4 
# --------------------------------------------------------------------- #
# Set the number of Newton-Raphson iterations which sets the precision. #
# 0 is close to the single precision limit and is better than DUNE/HK   #
# in the high statistics regime. Increasing N_Newton to 1,2,... rapidly #
# improves the precision at a modest computational cost                 #
# --------------------------------------------------------------------- #
N_Newton = 0

#s12sq = 0.31
#s13sq = 0.02
#s23sq = 0.55
s12sq = np.sin(theta12)**2
s13sq = np.sin(theta13)**2
s23sq = np.sin(theta23)**2

### NUFAST
def Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton):
	# --------------------------------------------------------------------- #
	# First calculate useful simple functions of the oscillation parameters #
	# --------------------------------------------------------------------- #
	c13sq = 1 - s13sq

	# Ueisq's
	Ue2sq = c13sq * s12sq
	Ue3sq = s13sq

	# Umisq's, Utisq's and Jvac
	Um3sq = c13sq * s23sq
	# Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
	Ut2sq = s13sq * s12sq * s23sq
	Um2sq = (1 - s12sq) * (1 - s23sq)

	Jrr = np.sqrt(Um2sq * Ut2sq)
	sind = np.sin(delta_cp)
	cosd = np.cos(delta_cp)

	Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
	Jmatter = 8 * Jrr * c13sq * sind
	Amatter = Ye * rho * E * YerhoE2a
	Dmsqee = Dmsq31 - s12sq * Dmsq21

	# calculate A, B, C, See, Tee, and part of Tmm
	A = Dmsq21 + Dmsq31 # temporary variable
	See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq
	Tmm = Dmsq21 * Dmsq31 # using Tmm as a temporary variable
	Tee = Tmm * (1 - Ue3sq - Ue2sq)
	C = Amatter * Tee
	A = A + Amatter

	# ---------------------------------- #
	# Get lambda3 from lambda+ of MP/DMP #
	# ---------------------------------- #
	xmat = Amatter / Dmsqee
	tmp = 1 - xmat
	lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat))

	# ---------------------------------------------------------------------------- #
	# Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
	# ---------------------------------------------------------------------------- #
	B = Tmm + Amatter * See # B is only needed for N_Newton >= 1
	for i in range(N_Newton):
		lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B) # this strange form prefers additions to multiplications

	# ------------------- #
	# Get  Delta lambda's #
	# ------------------- #
	tmp = A - lambda3
	Dlambda21 = np.sqrt(tmp * tmp - 4 * C / lambda3)
	lambda2 = 0.5 * (A - lambda3 + Dlambda21)
	Dlambda32 = lambda3 - lambda2
	Dlambda31 = Dlambda32 + Dlambda21

	# ----------------------- #
	# Use Rosetta for Veisq's #
	# ----------------------- #
	# denominators
	PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21)
	Xp3 = PiDlambdaInv * Dlambda21
	Xp2 = -PiDlambdaInv * Dlambda31

	# numerators
	Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
	Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

	Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq
	Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

	Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
	Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

	# ------------- #
	# Use NHS for J #
	# ------------- #
	Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv

	# ----------------------- #
	# Get all elements of Usq #
	# ----------------------- #
	Ue1sq = 1 - Ue3sq - Ue2sq
	Um1sq = 1 - Um3sq - Um2sq

	Ut3sq = 1 - Um3sq - Ue3sq
	Ut2sq = 1 - Um2sq - Ue2sq
	Ut1sq = 1 - Um1sq - Ue1sq

	# ----------------------- #
	# Get the kinematic terms #
	# ----------------------- #
	Lover4E = eVsqkm_to_GeV_over4 * L / E

	D21 = Dlambda21 * Lover4E
	D32 = Dlambda32 * Lover4E

	sinD21 = np.sin(D21)
	sinD31 = np.sin(D32 + D21)
	sinD32 = np.sin(D32)

	triple_sin = sinD21 * sinD31 * sinD32

	sinsqD21_2 = 2 * sinD21 * sinD21
	sinsqD31_2 = 2 * sinD31 * sinD31
	sinsqD32_2 = 2 * sinD32 * sinD32

	# ------------------------------------------------------------------- #
	# Calculate the three necessary probabilities, separating CPC and CPV #
	# ------------------------------------------------------------------- #
	Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
				+ (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
				+ (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
	Pme_CPV = -Jmatter * triple_sin

	Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
				 + Um3sq * Um1sq * sinsqD31_2 \
				 + Um3sq * Um2sq * sinsqD32_2)

	Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
				 + Ue3sq * Ue1sq * sinsqD31_2 \
				 + Ue3sq * Ue2sq * sinsqD32_2)

	# ---------------------------- #
	# Assign all the probabilities #
	# ---------------------------- #
	#probs_returned = np.empty((3, 3))
	#probs_returned[0][0] = Pee												# Pee
	#probs_returned[0][1] = Pme_CPC - Pme_CPV								# Pem
	#probs_returned[0][2] = 1 - Pee - probs_returned[0][1]  			    # Pet

	#probs_returned[1][0] = Pme_CPC + Pme_CPV								# Pme
	#probs_returned[1][1] = Pmm												# Pmm
	#probs_returned[1][2] = 1 - probs_returned[1][0] - Pmm					# Pmt

	#probs_returned[2][0] = 1 - Pee - probs_returned[1][0]					# Pte
	#probs_returned[2][1] = 1 - probs_returned[0][1] - Pmm					# Ptm
	#probs_returned[2][2] = 1 - probs_returned[0][2] - probs_returned[1][2]	# Ptt

	probs_ee = Pee
	probs_em = Pme_CPC - Pme_CPV
	probs_et = 1 - Pee - probs_em

	probs_me = Pme_CPC + Pme_CPV
	probs_mm = Pmm
	probs_mt = 1 - probs_me - Pmm

	return probs_me    

def Probability_Vacuum_LBL(s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E):
	# --------------------------------------------------------------------- #
	# First calculate useful simple functions of the oscillation parameters #
	# --------------------------------------------------------------------- #
	c13sq = 1 - s13sq

	# Ueisq's
	Ue3sq = s13sq
	Ue2sq = c13sq * s12sq

	# Umisq's, Utisq's and Jvac
	Um3sq = c13sq * s23sq
	# Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
	Ut2sq = s13sq * s12sq * s23sq
	Um2sq = (1 - s12sq) * (1 - s23sq)

	Jrr = np.sqrt(Um2sq * Ut2sq)
	sind = np.sin(delta)
	cosd = np.cos(delta)
	Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
	Jvac = 8 * Jrr * c13sq * sind

	# ----------------------- #
	# Get all elements of Usq #
	# ----------------------- #
	Ue1sq = 1 - Ue3sq - Ue2sq
	Um1sq = 1 - Um3sq - Um2sq

	Ut3sq = 1 - Um3sq - Ue3sq
	Ut2sq = 1 - Um2sq - Ue2sq
	Ut1sq = 1 - Um1sq - Ue1sq

	# ----------------------- #
	# Get the kinematic terms #
	# ----------------------- #
	Lover4E = eVsqkm_to_GeV_over4 * L / E

	D21 = Dmsq21 * Lover4E
	D31 = Dmsq31 * Lover4E

	sinD21 = np.sin(D21)
	sinD31 = np.sin(D31)
	sinD32 = np.sin(D31-D21)

	triple_sin = sinD21 * sinD31 * sinD32

	sinsqD21_2 = 2 * sinD21 * sinD21
	sinsqD31_2 = 2 * sinD31 * sinD31
	sinsqD32_2 = 2 * sinD32 * sinD32

	# ------------------------------------------------------------------- #
	# Calculate the three necessary probabilities, separating CPC and CPV #
	# ------------------------------------------------------------------- #
	Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
			+ (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
			+ (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2

	Pme_CPV = -Jvac * triple_sin

	Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
				 + Um3sq * Um1sq * sinsqD31_2 \
				 + Um3sq * Um2sq * sinsqD32_2)

	Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
				 + Ue3sq * Ue1sq * sinsqD31_2 \
				 + Ue3sq * Ue2sq * sinsqD32_2)

	# ---------------------------- #
	# Assign all the probabilities #
	# ---------------------------- #
	#probs_returned = np.empty((3, 3))
	#probs_returned[0][0] = Pee												# Pee
	#probs_returned[0][1] = Pme_CPC - Pme_CPV								# Pem
	#probs_returned[0][2] = 1 - Pee - probs_returned[0][1]  	 			# Pet

	#probs_returned[1][0] = Pme_CPC + Pme_CPV								# Pme
	#probs_returned[1][1] = Pmm												# Pmm
	#probs_returned[1][2] = 1 - probs_returned[1][0] - Pmm					# Pmt

	#probs_returned[2][0] = 1 - Pee - probs_returned[1][0]					# Pte
	#probs_returned[2][1] = 1 - probs_returned[0][1] - Pmm					# Ptm
	#probs_returned[2][2] = 1 - probs_returned[0][2] - probs_returned[1][2]	# Ptt

	probs_ee = Pee
	probs_em = Pme_CPC - Pme_CPV
	probs_et = 1 - Pee - probs_em

	probs_me = Pme_CPC + Pme_CPV
	probs_mm = Pmm
	probs_mt = 1 - probs_me - Pmm

	return probs_me

### NuExact
import sys
sys.path.append('D:/Documents/USTH/USTH/Internships/IFIRSE - Neutrino Physics/Code/NuOscProbExact-master/NuOscProbExact-master/src')
import oscprob3nu
import hamiltonians3nu
from globaldefs import *

def oscprob_nuexact_vacuum(L, E, delta_m21_sq, delta_m31_sq, U):
	h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_NO_BF,
                                                                S23_NO_BF,
                                                                S13_NO_BF,
                                                                DCP_NO_BF,
                                                                D21_NO_BF,
                                                                D31_NO_BF)
	h_vacuum = np.multiply(1./(E*1e9), h_vacuum_energy_indep)

		# CONV_KM_TO_INV_EV is pulled from globaldefs; it converts km to eV^{-1}
	Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
			oscprob3nu.probabilities_3nu( h_vacuum, L*CONV_KM_TO_INV_EV)

	return Pme

def oscprob_nuexact_matter(L, E, delta_m21_sq, delta_m31_sq, U):
	h_vacuum_energy_indep = \
    hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(  S12_NO_BF,
                                                                S23_NO_BF,
                                                                S13_NO_BF,
                                                                DCP_NO_BF,
                                                                D21_NO_BF,
                                                                D31_NO_BF)

	# Units of VCC_EARTH_CRUST: [eV]
	h_matter = hamiltonians3nu.hamiltonian_3nu_matter(  h_vacuum_energy_indep,
														E*1e9,
														VCC_EARTH_CRUST)

	Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = \
		oscprob3nu.probabilities_3nu(h_matter, L*CONV_KM_TO_INV_EV)

	return Pme

# Analytical formula for Pee_JUNO
def Pee_analytical(L, E, theta12, theta13, Dmsq21, Dmsq31):
	hbarc = 197.3269804e-9     # eV.m
	L_natural_unit = (L * 1e3) / hbarc
	E_eV = E * 1e9
	Dmsq32 = Dmsq31 - Dmsq21	
	s12, c12 = np.sin(theta12), np.cos(theta12)
	s13, c13 = np.sin(theta13), np.cos(theta13)   
    # oscillation phases
	phi21 = Dmsq21 * L_natural_unit / (4*E_eV) 
	phi31 = Dmsq31 * L_natural_unit / (4*E_eV) 
	phi32 = Dmsq32 * L_natural_unit / (4*E_eV)	
	term21 = (c13**4) * (np.sin(2*theta12)**2) * (np.sin(phi21)**2)
	term31 = (np.sin(2*theta13)**2) * (c12**2) * (np.sin(phi31)**2)
	term32 = (np.sin(2*theta13)**2) * (s12**2) * (np.sin(phi32)**2)
	
	return 1 - term21 - term31 - term32

Pee_ana = [Pee_analytical(L, E, theta12, theta13, Dmsq21, Dmsq31) for E in E_range]

P_nuexact_matter = np.array([oscprob_nuexact_matter(L, E, Dmsq21, Dmsq31, U) for E in E_range])
P_nuexact_vacuum = np.array([oscprob_nuexact_vacuum(L, E, Dmsq21, Dmsq31, U) for E in E_range])

P_nufast_matter = np.array([Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton) for E in E_range])
P_nufast_vacuum = np.array([Probability_Vacuum_LBL(s12sq, s13sq, s23sq, delta_cp, Dmsq21, Dmsq31, L, E) for E in E_range])

P_hamiltonian = np.array([P_3nu_evolutor(L, E, U, [Dmsq21, Dmsq31], rho, 1, 0) for E in E_range])

# Plotting
exp = "DUNE"
plt.figure(figsize=(12,5))
#plt.plot(E_range, Pee_ana, label=r'$P(\nu_e \to \nu_e) (Analytical)$', color='blue', linestyle='-', linewidth=2)
#plt.plot(E_range, P_nuexact_matter, label=r'$P(\nu_\mu \to \nu_e) (NuExact-matter)$', color='red', linestyle='-', linewidth=2)
#plt.plot(E_range, P_nuexact_vacuum, label=r'$P(\nu_\mu \to \nu_e) (NuExact-vacuum)$', color='red', linestyle=':', linewidth=2)
plt.plot(E_range, P_nufast_matter, label=r'$P(\nu_\mu \to \nu_e) (NuFast-matter)$', color='yellow', linestyle='-', linewidth=2)
#plt.plot(E_range, P_nufast_vacuum, label=r'$P(\nu_\mu \to \nu_e) (NuFast-vacuum)$', color='yellow', linestyle=':', linewidth=2)
plt.plot(E_range, P_hamiltonian, label=r'$P(\nu_\mu \to \nu_e) (hamiltonian)$', color='green', linestyle='-', linewidth=2)
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('Probability')
plt.title(f'P(numu2nue) at L={L} km ({exp}) in vacuum')
plt.grid(True)
plt.ylim(0, 1.1)

# Annotate the maximum (minimum survival)
#max_osc_idx = np.argmax(Pmu_tau)
#E_max_osc = E_range[max_osc_idx]
#plt.axvline(x=E_max_osc, color='gray', linestyle='--', label=f'E ~ {E_max_osc:.2f} GeV')

plt.legend()
plt.tight_layout()
plt.show()

#E0 = 2.5
#Pmu_e_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 0), 5)
#Pmu_tau_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 2), 5)
#Pmu_mu_3 = np.round(P_3nu_matter_evolutor(L, E0, U, [delta_m21, delta_m31], rho, 1, 1), 5)

#print(f"At L={L} km, E={E0} GeV, rho={rho} g/cm^3:")
#print(f"P(\nu_\mu -> \nu_e)  = {Pmu_e_3}")
#print(f"P(\nu_\mu -> \nu_\tau)  = {Pmu_tau_3}")
#print(f"P(\nu_\mu -> \nu_\mu)  = {Pmu_mu_3}")
#print("Total Probability: ", Pmu_e_3 + Pmu_tau_3 + Pmu_mu_3)
#print(f"Total probability deviation: {Pmu_e_3 + Pmu_tau_3 + Pmu_mu_3 - 1:.5e}") 

# Absolute difference
y1 = P_hamiltonian   
y2 = P_nufast_matter   
# Difference plot
diff = y1 - y2
plt.figure(figsize=(12,5))
plt.title(f'Hamiltonian vs NuFast at L={L} km ({exp}) in matter')
plt.xlabel(r'Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel(r'$\Delta P(\nu_\mu \to \nu_e)$')
plt.plot(E_range, diff, label='Difference')
plt.grid(True)
plt.show()

# Unitarity test
I = np.eye(U.shape[0])
norm = np.linalg.norm(U.conj().T @ U - I, "fro")

print(r'$U^\dagger U - I$: ', norm)
