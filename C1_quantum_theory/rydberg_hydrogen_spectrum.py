import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Rydberg constant (1/m)
# -----------------------------
R_H = 1.097e7

# -----------------------------
# 2. Function to calculate wavelength
#    using the Rydberg formula
# -----------------------------
def hydrogen_wavelength(n_i, n_f):
    """
    Calculate the emission wavelength (in nm)
    for hydrogen transition n_i -> n_f
    """
    if n_i <= n_f:
        return None
    inv_lambda = R_H * (1 / n_f**2 - 1 / n_i**2)
    wavelength_m = 1 / inv_lambda
    wavelength_nm = wavelength_m * 1e9
    return wavelength_nm

# -----------------------------
# 3. Generate spectral lines
# -----------------------------
series = {
    "Lyman (n_f=1)": 1,
    "Balmer (n_f=2)": 2,
    "Paschen (n_f=3)": 3
}

spectral_lines = {}

for name, n_f in series.items():
    lines = []
    for n_i in range(n_f + 1, 11):   # transitions from n_i to n_f
        wl = hydrogen_wavelength(n_i, n_f)
        lines.append((n_i, wl))
    spectral_lines[name] = lines

# -----------------------------
# 4. Print wavelengths
# -----------------------------
for name, lines in spectral_lines.items():
    print(f"\n{name}")
    for n_i, wl in lines:
        print(f"n = {n_i} -> {series[name]} : {wl:.2f} nm")

# -----------------------------
# 5. Plot emission spectrum
# -----------------------------
plt.figure(figsize=(12, 6))

y_positions = {
    "Lyman (n_f=1)": 3,
    "Balmer (n_f=2)": 2,
    "Paschen (n_f=3)": 1
}

for name, lines in spectral_lines.items():
    y = y_positions[name]
    for n_i, wl in lines:
        plt.vlines(wl, y - 0.3, y + 0.3, linewidth=2, label=name if n_i == lines[0][0] else "")

        # annotate transition
        # plt.text(wl, y + 0.35, f"{n_i}→{series[name]}", rotation=90,
        #          ha='center', va='bottom', fontsize=8)

plt.yticks([1, 2, 3], ["Paschen", "Balmer", "Lyman"])
plt.xlabel("Wavelength (nm)", fontsize=12)
plt.ylabel("Series", fontsize=12)
plt.title("Emission Spectrum of the Hydrogen Atom", fontsize=14)
plt.xlim(0, 2000)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()