import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
h = 6.62607015e-34     # Planck constant (J·s)
c = 2.99792458e8       # Speed of light (m/s)
kB = 1.380649e-23      # Boltzmann constant (J/K)

# --- Planck's law in terms of wavelength ---
def planck_lambda(wavelength, T):
    """
    wavelength: m
    T: Kelvin
    returns: spectral energy density ρλ(T)
    """
    numerator = 8 * np.pi * h * c / wavelength**5
    denominator = np.exp(h * c / (wavelength * kB * T)) - 1
    return numerator / denominator

# --- Wavelength range (0 - 2000 nm) ---
wavelength_nm = np.linspace(1, 2000, 1000)   # avoid λ=0
wavelength_m = wavelength_nm * 1e-9

# --- Temperatures to plot ---
temperatures = [3000, 4000, 5000, 6000]

# --- Plot ---
plt.figure(figsize=(8, 6))

for T in temperatures:
    rho = planck_lambda(wavelength_m, T)
    plt.plot(wavelength_nm, rho, label=f"T = {T} K")

plt.title("Planck Radiation Law (Spectral Energy Density vs Wavelength)")
plt.xlabel("Wavelength (nm)")
plt.ylabel(r"$\rho_\lambda(T)$  (J·m$^{-4}$)")
plt.legend()
plt.grid(True)
plt.xlim(0, 2000)
plt.tight_layout()
plt.show()
