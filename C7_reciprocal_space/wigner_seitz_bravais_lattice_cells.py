#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import Voronoi

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from ase.lattice import all_variants


def get_wigner_seitz_3d(cell, grid_range=(-1, 2)):
    """
    Wigner–Seitz cell via Voronoi around the origin lattice point.
    Returns:
      ws_vertices: (Nv,3)
      ws_edges: list of (M,3) polylines (closed)
      ws_facets: list of (K,3) polygon vertices (unordered but co-planar)
    """
    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    gx, gy, gz = np.mgrid[grid_range[0]:grid_range[1],
                         grid_range[0]:grid_range[1],
                         grid_range[0]:grid_range[1]]
    px, py, pz = np.tensordot(cell, np.array([gx, gy, gz]), axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    origin_idx = np.where(np.all(np.isclose(points, 0.0), axis=1))[0]
    if len(origin_idx) != 1:
        raise RuntimeError(f"Origin point not uniquely found: {origin_idx}")
    origin_idx = int(origin_idx[0])

    vor = Voronoi(points)

    ws_edges = []
    ws_facets = []
    ws_vids = set()

    for (p0, p1), ridge_vids in zip(vor.ridge_points, vor.ridge_vertices):
        if p0 != origin_idx and p1 != origin_idx:
            continue
        if np.any(np.array(ridge_vids) < 0):  # infinite ridge
            continue

        ridge_vids = np.asarray(ridge_vids, dtype=int)
        poly = vor.vertices[ridge_vids]
        ws_facets.append(poly)
        ws_edges.append(vor.vertices[np.r_[ridge_vids, ridge_vids[0]]])  # closed polyline
        ws_vids.update(ridge_vids.tolist())

    ws_vertices = vor.vertices[sorted(ws_vids)]
    return ws_vertices, ws_edges, ws_facets


def get_cell_poly_3d(cell):
    """
    Parallelepiped: vertices, edges, quad faces.
    """
    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    dx, dy, dz = np.mgrid[0:2, 0:2, 0:2]
    dxyz = np.c_[dx.ravel(), dy.ravel(), dz.ravel()]
    px, py, pz = np.tensordot(cell, np.array([dx, dy, dz]), axes=[0, 0])
    vertices = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    edges = []
    n = len(vertices)
    for i in range(n):
        for j in range(i):
            if np.abs(dxyz[i] - dxyz[j]).sum() == 1:
                edges.append(np.vstack([vertices[i], vertices[j]]))

    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    facets = [vertices[f] for f in faces]
    return vertices, edges, facets


def _unique_rows(points, tol=1e-10):
    # 去重（防止极少数情况下重复点导致排序/三角化异常）
    pts = np.asarray(points, float)
    key = np.round(pts / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def order_polygon_vertices_3d(points):
    """
    将共面多边形顶点按平面内极角排序（用于正确绘制面片/扇形三角化）。
    """
    pts = _unique_rows(points)
    if len(pts) < 3:
        return pts

    c = pts.mean(axis=0)

    # 选两条不共线的边来构建法向
    v0 = pts[0] - c
    n = None
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - c
        v2 = pts[i + 1] - c
        nn = np.cross(v1, v2)
        if np.linalg.norm(nn) > 1e-12:
            n = nn / np.linalg.norm(nn)
            break

    # 若几乎共线（退化），直接返回
    if n is None:
        return pts

    # 平面内正交基 u, v
    u = v0
    if np.linalg.norm(u) < 1e-12:
        # 若第一个点恰好接近中心，找个别的
        for p in pts:
            if np.linalg.norm(p - c) > 1e-12:
                u = p - c
                break
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    # 计算每个点在 (u,v) 基的极角并排序
    rel = pts - c
    ang = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(ang)
    return pts[order]


def polygon_fan_triangulation(points):
    """
    对已排序的凸多边形进行扇形三角化：
      (p0, pi, p(i+1))
    """
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return []
    tris = []
    p0 = pts[0]
    for i in range(1, len(pts) - 1):
        tris.append(np.vstack([p0, pts[i], pts[i + 1]]))
    return tris


def add_vectors(ax, cell, color, prefix="a", lw=1.6):
    for i, v in enumerate(cell):
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=color, linewidth=lw)
        ax.text(v[0], v[1], v[2], f"{prefix}{i+1}", color=color, fontsize=7)


def add_edges(ax, edges, color, lw=1.8, alpha=0.9):
    segs = [e for e in edges]
    lc = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)


def add_vertices(ax, verts, size=8, alpha=0.9):
    verts = np.asarray(verts)
    if len(verts) == 0:
        return
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='k', s=size, alpha=alpha)


def add_faces(ax, facets, color, alpha=0.10):
    """
    facets: list of polygon vertices (unordered or ordered)
    做法：先按平面内极角排序，再用 fan triangulation。
    完全避免 3D Delaunay/Qhull 的维度错误。
    """
    all_tris = []
    for f in facets:
        f_ord = order_polygon_vertices_3d(f)
        tris = polygon_fan_triangulation(f_ord)
        all_tris.extend(tris)

    if not all_tris:
        return

    pc = Poly3DCollection(all_tris, facecolors=color, edgecolors='none', alpha=alpha)
    ax.add_collection3d(pc)


def set_axes_equal(ax, pts):
    pts = np.asarray(pts)
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    half = span / 2.0

    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def main():
    bravais = {b.variant: (b.longname, b) for b in all_variants()}

    keys = [
        'CUB', 'BCC', 'FCC', 'TET', 'BCT1',
        'ORC', 'ORCI', 'ORCC', 'ORCF1', 'RHL1',
        'HEX', 'MCL', 'MCLC1', 'TRI1a'
    ]
    titles = [
        "Primitive Cubic",
        "Body-centered Cubic (bcc)",
        "Face-centered Cubic (fcc)",
        "Primitive Tetragonal",
        "Body-centered Tetragonal",
        "Primitive Orthorhombic",
        "Body-centered Orthorhombic",
        "Base-centered Orthorhombic",
        "Face-centered Orthorhombic",
        "Rhombohedral",
        "Hexagonal",
        "Simple Monoclinic",
        "Base-centered Monoclinic",
        "Triclinic",
    ]

    nrows, ncols = 3, 5
    fig = plt.figure(figsize=(10, 7.5))

    colors = dict(p='red', c='blue', w='green')

    for idx, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.set_title(titles[idx], fontsize=9, pad=2)

        latt = bravais[key][1]
        pcell = latt.tocell().array
        ccell = latt.conventional().tocell().array

        all_pts = []

        # Primitive
        pv, pe, pf = get_cell_poly_3d(pcell)
        add_vectors(ax, pcell, colors['p'], prefix="a", lw=1.4)
        add_vertices(ax, pv, size=6)
        add_edges(ax, pe, colors['p'], lw=1.8, alpha=0.9)
        add_faces(ax, pf, colors['p'], alpha=0.07)
        all_pts.append(pv)

        # Conventional (optional)
        if not np.allclose(pcell, ccell):
            cv, ce, cf = get_cell_poly_3d(ccell)
            add_vectors(ax, ccell, colors['c'], prefix="b", lw=1.2)
            add_vertices(ax, cv, size=6)
            add_edges(ax, ce, colors['c'], lw=1.6, alpha=0.9)
            add_faces(ax, cf, colors['c'], alpha=0.05)
            all_pts.append(cv)

        # Wigner–Seitz
        wv, we, wf = get_wigner_seitz_3d(pcell, grid_range=(-1, 2))
        add_vertices(ax, wv, size=6)

        # WS edges: polyline -> segments
        ws_segs = []
        for polyline in we:
            polyline = np.asarray(polyline)
            for i in range(len(polyline) - 1):
                ws_segs.append(np.vstack([polyline[i], polyline[i + 1]]))
        add_edges(ax, ws_segs, colors['w'], lw=1.8, alpha=0.9)

        # WS faces (robust triangulation)
        add_faces(ax, wf, colors['w'], alpha=0.07)
        all_pts.append(wv)

        pts = np.vstack(all_pts)
        set_axes_equal(ax, pts)

        ax.set_axis_off()
        ax.view_init(elev=18, azim=-50)

    # 最后一个空格子（3x5=15）
    ax_last = fig.add_subplot(nrows, ncols, nrows * ncols, projection='3d')
    ax_last.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
