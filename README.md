# NuOscProb_Hamiltonian

This package computes neutrino oscillation probability by the most fundamental approach, explicitly diagonalising the Hamiltonian. We aim to design a general program that can be applied to a wide range of problems with the highest precision, for the purpose of serving future experiments such as JUNO, DUNE, Hyper-K, etc. We examined our program vs analytical formulas and other approximate/exact packages such as *NuFast* and *NuOscProbsExact*. Both vacuum and matter effects are included.

Requirements
- Python 3.8 or newer (tested with Python 3.11.4)
- Install dependencies:
  ```bash
  pip install numpy scipy 

## Structure
The parameters can be changed at the start of the program.

Different Hamiltonians and PMNS matrices are written in a separate function. One could call the function on needed.

---

## Notes on Neutrino Oscillations

* **Oscillation**: the process of changing flavour identity of neutrinos ($e, \mu, \tau$) after propagation through a distance and time. Detectors measure different mixing–mass states, indicating oscillation in the mass eigenstates.
* **Explanation**: caused by mass differences ($\Delta m^2 \neq 0 \ \Rightarrow\ m_i \neq m_j \\Rightarrow\ \text{neutrinos are massive}$
).

### Quantum Mechanics 

* Start from the initial state $|\nu_\alpha\rangle \ \(\Psi(x, 0)\)$;  $(\alpha = e, \mu, \tau)$ to the final state $|\nu_\beta\rangle \ \(\Psi(x, t)\)$; $(\beta = e, \mu, \tau)$.
* Apply time evolution: $\Psi(x, t) = \Psi(x, 0) \. e^{-i H t}  $.
* Oscillation probability is given by overlap $\| \Psi(x, t) \|^2$.

### Approaches

1. **Mass-basis phase evolution (analytical formula)**: The neutrino state then exists as a superposition of the three mass eigenstates until it interacts and the wave function describes it collapses into weak eigenstates. Directly expand the initial state in mass basis = evolve the phase of each eigenstate. $\ \Rightarrow$ Oscillations caused by the phase difference of the eigenstates
2. **Evolution operator (Hamiltonian)**: apply the time-evolution operator exp(-iHt) directly. $\ \Rightarrow$ Oscillations are rotations in Hilbert space caused by energy differences of the flavour states.

### In Matter

* Neutrinos propagating in matter interact with electrons and nucleons.
* **CC interactions** ($\nu_e + e^- \to \nu_e + e^-$) affect only electron neutrinos.
* **NC interactions** ($\nu_i + n,p \to \nu_i + n,p$) affect all flavours equally and thus do not change oscillations.

**$\Rightarrow$ Matter effect**: adds an extra potential `V_CC` in the Hamiltonian, modifying oscillation behaviour (MSW effect).

---

## References

\[1] Mark Thomson, *Modern Particle Physics*

\[2] G. Fantini, A. Gallo Rosso, and F. Vissani, *The Formalism of Neutrino Oscillations: An Introduction*, \*\*arXiv:\*\*hep-ph/1802.05781v2 (2020)

\[3] Son Van Cao, *Study of Antineutrino Oscillation Using Accelerator and Atmospheric Data in MINOS*, DOI: 10.2172/1151746

\[4] M.C. Gonzalez-Garcia, Michele Maltoni, and Thomas Schwetz, *NuFIT: Three-Flavour Global Analyses of Neutrino Oscillation Experiments*, \*\*arXiv:\*\*hep-ph/2111.03086v2 (2021)

\[5] Peter J. Mohr, David B. Newell, Barry N. Taylor, and Eite Tiesinga, *CODATA Recommended Values of the Fundamental Physical Constants: 2022*, \*\*arXiv:\*\*hep-ph/2409.03787v2 (2024)

\[6] Ara Ioannisian and Stefan Pokorski, *Three Neutrino Oscillations in Matter*, \*\*arXiv:\*\*hep-ph/1801.10488v4 (2018)

\[7] Peter B. Denton and Stephen J. Parke, *Fast and Accurate Algorithm for Calculating Long-Baseline Neutrino Oscillation Probabilities with Matter Effects: NuFast*, \*\*arXiv:\*\*hep-ph/2405.02400v1 (2024)

\[8] Mauricio Bustamante, *NuOscProbExact: a general-purpose code to compute exact two-flavor and three-flavor neutrino oscillation probabilities*, \*\*arXiv:\*\*hep-ph/1904.12391v2 (2019)

\[9] JUNO Collaboration, *JUNO Physics and Detector*, \*\*arXiv:\*\*hep-ph/2104.02562v2 (2021)

---
