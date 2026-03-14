import numpy as np
import matplotlib.pyplot as plt


def get_hkl(default=(2, 2, 2)):
    s = input(f"Enter [h,k,l] as a row vector {list(default)} -> ").strip()
    if s == "":
        return np.array(default, dtype=float)
    s = s.replace("[", "").replace("]", "").replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 3:
        raise ValueError("Please provide exactly three numbers for [h,k,l].")
    return np.array(list(map(float, parts)), dtype=float)


def safe_sinN_over_sin(beta, N, tol=1e-6):
    """
      Rh(i)=N; if abs(sin(beta))>=1e-6, Rh(i)=sin(N*beta)/sin(beta)
    """
    denom = np.sin(beta)
    out = np.full_like(beta, float(N), dtype=float)
    mask = np.abs(denom) >= tol
    out[mask] = np.sin(N * beta[mask]) / denom[mask]
    return out


def main():
    # -----------------------------
    # BCC cubic crystal vectors
    # -----------------------------
    a = 2.87  # Angstroms, Fe
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.0, 1.0, 0.0])
    a3 = a * np.array([0.0, 0.0, 1.0])

    # Miller indices
    A = get_hkl(default=(2, 2, 2))
    h, k, l = A
    print(f"[h,k,l]=[{h:6.3f},{k:6.3f},{l:6.3f}]")

    # -----------------------------
    # Reciprocal lattice vectors
    # -----------------------------
    Vt = np.dot(a1, np.cross(a2, a3))  # unit cell volume
    b1 = 2.0 * np.pi * np.cross(a2, a3) / Vt
    b2 = 2.0 * np.pi * np.cross(a3, a1) / Vt
    b3 = 2.0 * np.pi * np.cross(a1, a2) / Vt

    G_v = h * b1 + k * b2 + l * b3
    G_m = np.linalg.norm(G_v)
    G_hat = G_v / G_m

    # plane spacing and wavelength choice
    d = 2.0 * np.pi / G_m
    lam = d / 3.0

    # -----------------------------
    # Angle range
    # -----------------------------
    themin, themax = 0.0, 80.0
    thes = (themax - themin) / 400.0
    thet = np.arange(themin, themax + thes, thes)  # degrees
    theta = np.deg2rad(thet)  # radians

    # -----------------------------
    # Atom params for Fe and BCC geometry
    # -----------------------------
    Z = 26  # Fe
    dnn = np.sqrt(3.0) * a / 2.0
    R = dnn / 2.0

    N = 50  # number of cells used in lattice sum

    # dot products of G_hat with the basis vectors
    Ga1 = np.dot(G_hat, a1)
    Ga2 = np.dot(G_hat, a2)
    Ga3 = np.dot(G_hat, a3)

    # basis atoms (BCC): two-atom basis
    nb = 2
    rb = np.zeros((nb, 3), dtype=float)
    rb[0, :] = a * np.array([0.0, 0.0, 0.0])
    rb[1, :] = a * np.array([1.0, 1.0, 1.0]) / 2.0

    # dot products of G_hat with basis positions
    Gb = np.array([np.dot(G_hat, rbj) for rbj in rb], dtype=float)

    # -----------------------------
    # Loop over theta: compute fj, Rh/Rk/Rl, structure factor SG
    # -----------------------------
    fj = np.zeros_like(theta, dtype=float)
    SG = np.zeros_like(theta, dtype=complex)

    for i, thv in enumerate(theta):
        if thv == 0.0:
            thv = 1e-6  # prevent theta=0 problems

        # G = 2*k*sin(theta), with k = 2*pi/lambda => G = (4*pi/lambda)*sin(theta)
        G = (4.0 * np.pi / lam) * np.sin(thv)

        # constant-n approximation for atomic form factor
        GR = G * R
        fj[i] = 3.0 * Z * (np.sin(GR) - GR * np.cos(GR)) / (GR**3)

        # Laue condition factors (divide by h,k,l)
        betA = G * Ga1 / 2.0 / h
        betB = G * Ga2 / 2.0 / k
        betC = G * Ga3 / 2.0 / l

        Rh = (np.sin(N * betA) / np.sin(betA)) if abs(np.sin(betA)) >= 1e-6 else float(N)
        Rk = (np.sin(N * betB) / np.sin(betB)) if abs(np.sin(betB)) >= 1e-6 else float(N)
        Rl = (np.sin(N * betC) / np.sin(betC)) if abs(np.sin(betC)) >= 1e-6 else float(N)

        # structure factor for BCC basis
        # SG(i) = sum_j fj(i) * exp(-i * G * Gb(j))
        SG[i] = 0.0 + 0.0j
        for j in range(nb):
            SG[i] += fj[i] * np.exp(-1j * G * Gb[j])

        # Note: Rh, Rk, Rl are used later via T0 (lattice sum intensity)
        if i == 0:
            Rh_arr = np.zeros_like(theta, dtype=float)
            Rk_arr = np.zeros_like(theta, dtype=float)
            Rl_arr = np.zeros_like(theta, dtype=float)
        Rh_arr[i] = Rh
        Rk_arr[i] = Rk
        Rl_arr[i] = Rl

    # lattice sum intensity
    T0 = np.abs(Rh_arr * Rk_arr * Rl_arr) ** 2
    SG_int = np.abs(SG) ** 2

    # -----------------------------
    # Figure 1: form factor
    # -----------------------------
    str_f = f"Form factor for F_e (Z=26), λ={lam:6.3f} Å"
    plt.figure(figsize=(7, 5))
    plt.plot(thet, fj, 'k')
    plt.title(str_f)
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel(r'$f_j$')
    plt.grid(True)
    plt.tight_layout()

    # -----------------------------
    # Figure 2: structure factor magnitude squared
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(thet, SG_int, 'k')
    plt.axis([0, np.max(thet), 0, 1])
    str_sg = (f"BCC F_e (Z=26), [a, d, λ]=[{a:6.3f}, {d:6.3f}, {lam:6.3f}] Å, "
              f"(hkl)=({h:4.0f},{k:4.0f},{l:4.0f})")
    plt.title(str_sg)
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel('Structure Factor')
    plt.grid(True)
    plt.tight_layout()

    # -----------------------------
    # Figure 3: intensity at detector
    # -----------------------------
    I = SG_int * T0
    plt.figure(figsize=(7, 5))
    plt.plot(thet, I, 'k')
    plt.axis([0, np.max(thet), 0, 1e9])
    plt.title(str_sg)
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel('Calculated Bragg Peak Intensity')
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
