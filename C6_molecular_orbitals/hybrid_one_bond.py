import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def plot_sp3_hybrids():
    # -----------------------------
    # Grid parameters
    # -----------------------------
    N = 128
    ul = 1.0
    us = 2.0 * ul / N

    grid = np.linspace(-ul, ul, N + 1)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")

    # -----------------------------
    # Common radial part
    # -----------------------------
    r = np.sqrt(x**2 + y**2 + z**2)
    fr = 0.5 * np.exp(-r / 2.0)

    # -----------------------------
    # sp3 hybrid orbitals (optimized)
    # -----------------------------
    ff = [
        (2.0 - r + x + y + z) * fr,  # ff1
        (2.0 - r - x - y + z) * fr,  # ff2
        (2.0 - r + x - y - z) * fr,  # ff3
        (2.0 - r - x + y - z) * fr   # ff4
    ]

    colors = [
        (0.80, 0.10, 0.10, 0.65),  # red
        (0.10, 0.60, 0.10, 0.65),  # green
        (0.10, 0.20, 0.80, 0.65),  # blue
        (0.60, 0.10, 0.60, 0.65)   # purple
    ]

    isovalue = 0.95

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, (orb, col) in enumerate(zip(ff, colors)):
        verts, faces, normals, _ = marching_cubes(
            orb, level=isovalue, spacing=(us, us, us)
        )

        # shift vertices to correct coordinates
        verts += np.array([grid[0], grid[0], grid[0]])

        mesh = Poly3DCollection(verts[faces], linewidths=0.05)
        mesh.set_facecolor(col)
        mesh.set_edgecolor((0, 0, 0, 0.15))
        ax.add_collection3d(mesh)

    # -----------------------------
    # View & axes (MATLAB-like)
    # -----------------------------
    ax.view_init(elev=14, azim=110)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(-0.25, 0.25)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"sp$^3$ Hybrid Orbitals (ff1–ff4)")

    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sp3_hybrids()
