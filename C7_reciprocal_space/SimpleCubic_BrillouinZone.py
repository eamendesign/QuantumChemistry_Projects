import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def liner(ax, v1, v2, linestyle='-', color='k', linewidth=1.0):
    """Draw a line from v1 to v2."""
    ax.plot([v1[0], v2[0]],
            [v1[1], v2[1]],
            [v1[2], v2[2]],
            linestyle=linestyle, color=color, linewidth=linewidth)


def setter(ax, x, y, z, face_color=(0.70, 0.60, 0.70), edge_alpha=0.3, face_alpha=0.5):
    """
    Fill a polygon face given x,y,z coordinate arrays (like MATLAB fill3),
    with controllable edge/face transparency.
    """
    verts = [list(zip(x, y, z))]
    poly = Poly3DCollection(verts, facecolors=[face_color], edgecolors='k', linewidths=1)
    poly.set_alpha(face_alpha)          # face alpha
   
    poly.set_edgecolor((0, 0, 0, edge_alpha))
    ax.add_collection3d(poly)


def scbz():
    # -----------------------------
    # High symmetry points (units of 2*pi/a)
    # -----------------------------
    Z = np.array([0.0, 0.0, 0.0])          # Gamma
    X = np.array([0.5, 0.0, 0.0])
    R = np.array([0.5, 0.5, 0.5])
    M = np.array([0.5, 0.5, 0.0])

    # Tetrahedron volume
    VSC = abs(np.dot(X, np.cross(R, M))) / 6.0

    print("========== Simple Cubic ==========")
    print(f"SC: irreducible V={VSC} of BZ volume")
    print("SC symmetry points (in units of 2*pi/a):")
    print("X=[1/2,0,0], R=[1/2,1/2,1/2], M=[1/2,1/2,0]")
    print("Known SC BZ volume: (2*pi/a)^3")

    # -----------------------------
    # Plot setup
    # -----------------------------
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=150)  
    ax.grid(True)

    # -----------------------------
    # Irreducible BZ tetrahedron edges (black)
    # -----------------------------
    # lines to symmetry points
    liner(ax, Z, X, linestyle='--', color='k', linewidth=1.0)
    liner(ax, Z, R, linestyle='--', color='k', linewidth=1.0)
    liner(ax, Z, M, linestyle='--', color='k', linewidth=1.0)
    # edges among X,R,M
    liner(ax, X, R, linestyle='--', color='k', linewidth=1.0)
    liner(ax, X, M, linestyle='--', color='k', linewidth=1.0)
    liner(ax, R, M, linestyle='--', color='k', linewidth=1.0)

    # -----------------------------
    # Fill tetrahedron faces
    # -----------------------------
    # face (Z, X, R)
    x = np.array([0.0, X[0], R[0]])
    y = np.array([0.0, X[1], R[1]])
    z = np.array([0.0, X[2], R[2]])
    setter(ax, x, y, z)

    # face (Z, X, M)
    x = np.array([0.0, X[0], M[0]])
    y = np.array([0.0, X[1], M[1]])
    z = np.array([0.0, X[2], M[2]])
    setter(ax, x, y, z)

    # face (Z, R, M)
    x = np.array([0.0, R[0], M[0]])
    y = np.array([0.0, R[1], M[1]])
    z = np.array([0.0, R[2], M[2]])
    setter(ax, x, y, z)

    # face (X, R, M)
    x = np.array([X[0], R[0], M[0]])
    y = np.array([X[1], R[1], M[1]])
    z = np.array([X[2], R[2], M[2]])
    setter(ax, x, y, z)

    # Labels
    ax.text(Z[0], Z[1], Z[2], r'$\Gamma$', fontsize=14)
    ax.text(X[0], X[1], X[2], 'X', fontsize=14)
    ax.text(R[0], R[1], R[2], 'R', fontsize=14)
    ax.text(M[0], M[1], M[2], 'M', fontsize=14)

    ax.set_title(r"SC BZ & irreducible Brillouin Zone in units of $2\pi/a$")

    # -----------------------------
    # SC BZ cube edges (blue)
    # corners to which lines will be drawn
    # -----------------------------
    c1 = np.array([ 0.5,  0.5,  0.5])
    c2 = np.array([-0.5,  0.5,  0.5])
    c3 = np.array([-0.5, -0.5,  0.5])
    c4 = np.array([ 0.5, -0.5,  0.5])
    c5 = np.array([ 0.5,  0.5, -0.5])
    c6 = np.array([-0.5,  0.5, -0.5])
    c7 = np.array([-0.5, -0.5, -0.5])
    c8 = np.array([ 0.5, -0.5, -0.5])

    # top face connectors
    liner(ax, c1, c2, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c1, c4, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c1, c5, linestyle='-', color='b', linewidth=1.0)

    liner(ax, c2, c6, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c2, c3, linestyle='-', color='b', linewidth=1.0)

    liner(ax, c3, c4, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c3, c7, linestyle='-', color='b', linewidth=1.0)

    # bottom/top connectors
    liner(ax, c5, c6, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c5, c8, linestyle='-', color='b', linewidth=1.0)

    liner(ax, c7, c6, linestyle='-', color='b', linewidth=1.0)
    liner(ax, c7, c8, linestyle='-', color='b', linewidth=1.0)

    liner(ax, c8, c4, linestyle='-', color='b', linewidth=1.0)

    # -----------------------------
    # Axis formatting
    # -----------------------------
    ax.set_box_aspect([1, 1, 1])  # axis equal
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel(r'$k_z$')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scbz()
