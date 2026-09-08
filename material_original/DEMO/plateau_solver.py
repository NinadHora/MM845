"""Discrete Plateau problem: soap films spanning two coaxial rings.

We minimize the total area of a triangle mesh with fixed boundary (two rings of
radius R at z = +-h/2) by explicit gradient descent. Since the vertex-wise area
gradient is the (integrated) mean-curvature vector, gradient descent on area IS
a discrete mean curvature flow -- the mesh evolves exactly like a soap film.

Smooth theory, used for validation:
  * Minimal surfaces of revolution are catenoids  r(z) = c cosh(z/c),
    with c solving  c cosh(h/(2c)) = R  (0, 1 or 2 roots; the larger root is
    the stable, area-minimizing catenoid).
  * Catenoid area:  A = pi c (h + c sinh(h/c)).
  * Past the critical separation  h* = 2 mu R / cosh(mu),  mu tanh(mu) = 1
    (h* ~ 1.3255 R, a saddle-node/fold bifurcation), NO spanning minimal
    surface exists: the film pinches off toward the Goldschmidt solution
    (two flat disks). The flow reproduces this pinch-off.

Run:  python plateau_solver.py        (writes figures/ next to this file)
Pure numpy/scipy/matplotlib, a few seconds of CPU, ~1300 vertices per mesh.
"""

import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# ---------------------------------------------------------------- palette ----
# Validated reference palette (dataviz): categorical slots 1-2, sequential blue.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ_BLUES = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
             "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef"]
CMAP_R = LinearSegmentedColormap.from_list("seq_blue_dark_low", SEQ_BLUES)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


# ------------------------------------------------------------------- mesh ----
def cylinder_mesh(R, h, n_theta=48, n_z=None):
    """Triangulated cylinder r=R between z=-h/2 and z=+h/2.

    Returns vertices V (N,3), faces F (M,3) and a boolean mask of the fixed
    (boundary-ring) vertices. n_z is kept odd so a vertex row sits at z=0.
    """
    if n_z is None:
        n_z = int(round(h / 0.04)) + 1
        n_z = max(15, n_z + (n_z % 2 == 0))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-h / 2.0, h / 2.0, n_z)
    T, Z = np.meshgrid(theta, z)
    V = np.column_stack([R * np.cos(T).ravel(), R * np.sin(T).ravel(), Z.ravel()])
    idx = np.arange(n_z * n_theta).reshape(n_z, n_theta)
    jp = np.r_[1:n_theta, 0]                      # theta-neighbour with wrap
    v00, v01 = idx[:-1, :], idx[:-1, jp]
    v10, v11 = idx[1:, :], idx[1:, jp]
    F = np.concatenate([np.stack([v00, v01, v11], -1).reshape(-1, 3),
                        np.stack([v00, v11, v10], -1).reshape(-1, 3)])
    fixed = np.zeros(n_z * n_theta, dtype=bool)
    fixed[idx[0]] = fixed[idx[-1]] = True
    return V, F, fixed


# -------------------------------------------------- area, gradient, masses ---
def area_and_grad(V, F):
    """Total area, its gradient wrt vertices, and per-triangle areas.

    For a triangle (a,b,c) with unit normal n:  grad_a Area = n x (c-b) / 2
    (perpendicular to the opposite edge, in the triangle plane).
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = _cross(b - a, c - a)
    nn = np.sqrt(np.einsum("ij,ij->i", n, n))
    n_hat = n / np.maximum(nn, 1e-300)[:, None]
    G = (_scatter(F[:, 0], 0.5 * _cross(n_hat, c - b), len(V))
         + _scatter(F[:, 1], 0.5 * _cross(n_hat, a - c), len(V))
         + _scatter(F[:, 2], 0.5 * _cross(n_hat, b - a), len(V)))
    tri_area = 0.5 * nn
    return tri_area.sum(), G, tri_area


def _cross(u, v):
    """Row-wise cross product, faster than np.cross for (M,3) arrays."""
    w = np.empty_like(u)
    w[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    w[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    w[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return w


def _scatter(idx, vals, n):
    """Sum rows of vals (M,3) into an (n,3) array at positions idx."""
    out = np.empty((n, 3))
    for d in range(3):
        out[:, d] = np.bincount(idx, weights=vals[:, d], minlength=n)
    return out


def vertex_masses(F, tri_area, n_verts):
    """Barycentric lumped mass: one third of each incident triangle's area."""
    return np.bincount(F.ravel(), weights=np.repeat(tri_area / 3.0, 3),
                       minlength=n_verts)


# ------------------------------------------------------------ the descent ----
def relax(V, F, fixed, max_iter=20000, tol=1e-11, collapse_frac=0.06,
          record_every=None):
    """Mean curvature flow  dV/dt = -grad(Area)/mass  with fixed boundary.

    Explicit Euler with heat-equation step dt ~ 0.2 * (min edge)^2, refreshed
    as the mesh deforms, plus a per-step displacement cap so the pinch-off
    stays tame. Stops on: converged (relative area change < tol for 30
    straight iterations), collapsed (neck radius < collapse_frac * R,
    i.e. past the critical separation), or max_iter.
    """
    V = V.copy()
    R = np.hypot(V[fixed, 0], V[fixed, 1]).max()
    free = ~fixed
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])

    def cfl():
        e = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1).min()
        return 0.2 * e * e, 0.5 * e

    dt, cap = cfl()
    trace = {"area": [], "neck": []}
    snap_it, snap_V = [], []
    A_prev, calm, status, it = np.inf, 0, "max_iter", 0

    for it in range(max_iter + 1):
        A, G, tri_area = area_and_grad(V, F)
        neck = np.hypot(V[:, 0], V[:, 1]).min()
        trace["area"].append(A)
        trace["neck"].append(neck)
        if record_every and it % record_every == 0:
            snap_it.append(it)
            snap_V.append(V.copy())
        if neck < collapse_frac * R:
            status = "collapsed"
            break
        calm = calm + 1 if abs(A_prev - A) < tol * A else 0
        if calm >= 30:
            status = "converged"
            break
        A_prev = A
        if it % 25 == 0:
            dt, cap = cfl()
        m = vertex_masses(F, tri_area, len(V))
        step = dt * G / np.maximum(m, 1e-3 * m.mean())[:, None]
        ln = np.sqrt(np.einsum("ij,ij->i", step, step))
        over = ln > cap
        if over.any():
            step[over] *= (cap / ln[over])[:, None]
        V[free] -= step[free]

    if record_every and snap_it[-1] != it:
        snap_it.append(it)
        snap_V.append(V.copy())
    trace = {k: np.asarray(v) for k, v in trace.items()}
    return V, status, trace, (snap_it, snap_V)


# --------------------------------------------------------- catenoid theory ---
def _f(c, R, h):
    return c * np.cosh(h / (2.0 * c)) - R


def catenoid_c(R, h, branch="stable"):
    """Catenoid parameter c with  c cosh(h/2c) = R, or None past h*.

    Two roots below the critical separation; the larger is the stable
    (area-minimizing) catenoid, the smaller the unstable saddle.
    """
    c_lo = h / 1400.0                      # keeps cosh's argument overflow-free
    res = minimize_scalar(_f, bounds=(c_lo, R), method="bounded", args=(R, h))
    if _f(res.x, R, h) > 0:
        return None
    lo, hi = ((res.x, R) if branch == "stable" else (c_lo, res.x))
    return brentq(_f, lo, hi, args=(R, h))


def catenoid_area(c, h):
    return np.pi * c * (h + c * np.sinh(h / c))


def critical_ratio():
    """h*/R for existence of the catenoid:  mu tanh mu = 1,  h* = 2 mu R/cosh mu."""
    mu = brentq(lambda m: m * np.tanh(m) - 1.0, 0.5, 3.0)
    return 2.0 * mu / np.cosh(mu), 1.0 / np.cosh(mu)   # (h*/R, neck c*/R)


# ---------------------------------------------------------------- drawing ----
def draw_film(ax, V, F, R, h, zlim, title=""):
    """Render the film into a 3d axis, faces coloured by distance from axis
    (sequential blue ramp, dark = small radius = the neck) + Lambert shading."""
    tri = V[F]
    r_cen = np.hypot(tri[..., 0].mean(1), tri[..., 1].mean(1))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_hat = n / np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    lam = 0.66 + 0.34 * np.abs(n_hat @ light)
    face = CMAP_R(np.clip(r_cen / R, 0, 1))
    face[:, :3] *= lam[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=face, edgecolors=face,
                                         linewidths=0.25, zsort="average"))
    th = np.linspace(0, 2 * np.pi, 240)
    for zz in (-h / 2.0, h / 2.0):                 # the fixed wire boundary
        ax.plot(R * np.cos(th), R * np.sin(th), zz, color=INK, lw=1.5)
    ax.set_xlim(-1.05 * R, 1.05 * R)
    ax.set_ylim(-1.05 * R, 1.05 * R)
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect((2.1 * R, 2.1 * R, 2.0 * zlim))
    ax.view_init(elev=16, azim=-58)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, pad=0)


def radius_colorbar(fig, axes):
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_R)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.04, shrink=0.5, aspect=40)
    cb.set_label("distance from axis  r / R", color=MUTED, fontsize=9)
    cb.ax.tick_params(color=BASELINE, labelcolor=MUTED, labelsize=8.5)
    cb.outline.set_edgecolor(BASELINE)


# ------------------------------------------------------------------ main -----
def main():
    t0 = time.time()
    R = 1.0
    here = Path(__file__).resolve().parent
    outdir = here / "figures"
    outdir.mkdir(exist_ok=True)

    hcrit, neck_crit = critical_ratio()
    print(f"critical separation  h*/R = {hcrit:.6f}   (fold at neck c*/R = {neck_crit:.4f})")

    # -- two hero runs, with snapshots for the animation ----------------------
    runs = {}
    for tag, hr in (("sub", 1.00), ("super", 1.50)):
        V0, F, fixed = cylinder_mesh(R, hr * R)
        Vf, status, trace, snaps = relax(V0, F, fixed, record_every=20)
        runs[tag] = dict(hr=hr, F=F, V=Vf, status=status, trace=trace, snaps=snaps)
        print(f"h/R = {hr:.2f}:  {status:9s} after {len(trace['area']) - 1:5d} iters, "
              f"{len(V0)} vertices, final neck {trace['neck'][-1]:.4f}")

    sub, sup = runs["sub"], runs["super"]
    c_th = catenoid_c(R, sub["hr"] * R)
    A_th = catenoid_area(c_th, sub["hr"] * R)
    A_num, neck_num = sub["trace"]["area"][-1], sub["trace"]["neck"][-1]
    err_A = abs(A_num - A_th) / A_th
    err_neck = abs(neck_num - c_th) / c_th
    print(f"\nvalidation at h/R = 1.00  (theory: stable catenoid c = {c_th:.6f})")
    print(f"  area  : mesh {A_num:.6f}  vs  pi*c*(h + c sinh(h/c)) = {A_th:.6f}   "
          f"rel err {err_A:.2%}")
    print(f"  neck  : mesh {neck_num:.6f}  vs  c = {c_th:.6f}   rel err {err_neck:.2%}")

    # -- sweep across the fold ------------------------------------------------
    ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.25, 1.30, 1.35, 1.40, 1.50, 1.60]
    print("\nsweep over ring separation:")
    sweep = []
    for hr in ratios:
        V0, F, fixed = cylinder_mesh(R, hr * R)
        Vf, status, trace, _ = relax(V0, F, fixed, max_iter=12000)
        if status == "max_iter" and trace["neck"][-1] < trace["neck"][-500] - 1e-3 * R:
            # neck still visibly moving at the cap (slow pinch near the fold):
            # continue the flow from where it stopped
            Vf, status, trace, _ = relax(Vf, F, fixed, max_iter=30000)
        neck = 0.0 if status == "collapsed" else trace["neck"][-1]
        c_here = catenoid_c(R, hr * R)
        th = f"{c_here:.4f}" if c_here else "  --  "
        print(f"  h/R = {hr:4.2f}   {status:9s}   neck = {neck:.4f}   theory c = {th}")
        sweep.append((hr, neck, status))

    # ================================================================ fig 1 ==
    zlim = 1.05 * max(sub["hr"], sup["hr"]) * R / 2.0
    fig = plt.figure(figsize=(9.0, 4.8), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_film(ax1, sub["V"], sub["F"], R, sub["hr"] * R, zlim,
              f"h/R = {sub['hr']:.2f}  $\\rightarrow$  catenoid")
    draw_film(ax2, sup["V"], sup["F"], R, sup["hr"] * R, zlim,
              f"h/R = {sup['hr']:.2f}  $\\rightarrow$  pinch-off")
    ax1.text2D(0.5, -0.02, f"area matches theory to {err_A:.2%}",
               transform=ax1.transAxes, ha="center", color=INK2, fontsize=9)
    ax2.text2D(0.5, -0.02, "no spanning minimal surface exists  (h > h*)",
               transform=ax2.transAxes, ha="center", color=INK2, fontsize=9)
    fig.suptitle("Soap films between two rings — area gradient descent", y=0.98)
    radius_colorbar(fig, [ax1, ax2])
    fig.savefig(outdir / "fig1_films.png", bbox_inches="tight")
    plt.close(fig)

    # ================================================================ fig 2 ==
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=150)

    # (a) area convergence
    tr = sub["trace"]
    axA.plot(tr["area"], color=BLUE, lw=1.8)
    axA.axhline(A_th, color=INK2, lw=1.2, ls="--")
    axA.text(len(tr["area"]) * 0.98, A_th, "catenoid area (theory)  ",
             ha="right", va="bottom", color=INK2, fontsize=9)
    axA.set_xlabel("iteration")
    axA.set_ylabel("mesh area")
    axA.set_title(f"(a)  descent at h/R = 1.00 — final gap {err_A:.2%}", fontsize=11)

    # (b) relaxed profile vs the catenary
    r_i = np.hypot(sub["V"][:, 0], sub["V"][:, 1])
    axB.scatter(sub["V"][:, 2], r_i, s=6, color=BLUE, alpha=0.35, lw=0,
                label="mesh vertices")
    zz = np.linspace(-sub["hr"] / 2, sub["hr"] / 2, 300)
    axB.plot(zz, c_th * np.cosh(zz / c_th), color=INK2, lw=1.4, ls="--",
             label=r"$r = c\,\cosh(z/c)$")
    axB.set_xlabel("z")
    axB.set_ylabel("r")
    axB.set_ylim(0, 1.05)
    axB.set_title(f"(b)  profile — neck {neck_num:.4f} vs c = {c_th:.4f}", fontsize=11)
    axB.legend(loc="lower right", fontsize=9)

    # (c) neck dynamics: existence vs pinch-off
    axC.plot(sub["trace"]["neck"], color=BLUE, lw=1.8, label="h/R = 1.00 (subcritical)")
    axC.plot(sup["trace"]["neck"], color=ORANGE, lw=1.8, label="h/R = 1.50 (supercritical)")
    axC.scatter([len(sup["trace"]["neck"]) - 1], [sup["trace"]["neck"][-1]],
                s=36, color=ORANGE, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    axC.axhline(c_th, color=INK2, lw=1.2, ls="--")
    axC.text(len(sub["trace"]["neck"]) * 0.30, c_th - 0.014, "stable catenoid neck",
             ha="left", va="top", color=INK2, fontsize=9)
    axC.set_xlabel("iteration")
    axC.set_ylabel("neck radius  min r")
    axC.set_ylim(0, 1.02)
    axC.set_title("(c)  neck radius under the flow", fontsize=11)
    axC.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "fig2_validation.png", bbox_inches="tight")
    plt.close(fig)

    # ================================================================ fig 3 ==
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    hh = np.linspace(0.02, 1.62, 400)
    stab = np.array([(catenoid_c(R, h) or np.nan) for h in hh])
    unst = np.array([(catenoid_c(R, h, "unstable") or np.nan) for h in hh])
    ax.plot(hh, stab, color=BLUE, lw=2.0, label="stable catenoid (theory)")
    ax.plot(hh, unst, color=MUTED, lw=1.4, ls=":", label="unstable catenoid (saddle)")
    box = dict(facecolor=SURFACE, edgecolor="none", pad=1.6)
    ax.plot([0, 1.65], [0, 0], color=INK2, lw=2.0, solid_capstyle="butt")
    ax.text(0.06, 0.028, "Goldschmidt solution (two flat disks)", color=INK2,
            fontsize=9, bbox=box)
    ax.axvline(hcrit, color=BASELINE, lw=1.2, ls="--")
    ax.text(hcrit + 0.02, 0.30, f"$h^* \\approx {hcrit:.4f}\\,R$", color=INK2,
            fontsize=10, bbox=box)
    ax.scatter([hcrit], [neck_crit], s=30, color=INK, zorder=4)
    ax.text(hcrit - 0.025, neck_crit - 0.015, "fold", ha="right", va="top",
            color=INK2, fontsize=9, bbox=box)
    ok = [(h, n) for h, n, s in sweep if s != "collapsed"]
    ko = [(h, n) for h, n, s in sweep if s == "collapsed"]
    ax.scatter(*zip(*ok), s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.4,
               zorder=3, label="spanning film (gradient descent)")
    if ko:
        ax.scatter(*zip(*ko), s=52, color=ORANGE, marker="X",
                   edgecolors=SURFACE, linewidths=0.8, zorder=3,
                   label="pinch-off (gradient descent)")
    ax.set_xlim(0, 1.65)
    ax.set_ylim(-0.04, 1.02)
    ax.set_xlabel("ring separation  h / R")
    ax.set_ylabel("neck radius  c / R")
    ax.set_title("Existence of the soap film: a saddle-node bifurcation", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.74), fontsize=9,
              frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_bifurcation.png", bbox_inches="tight")
    plt.close(fig)

    # ============================================================= animation =
    n_move, n_hold, fps = 46, 8, 12
    fig = plt.figure(figsize=(8.4, 4.2), dpi=100)
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2, projection="3d")
    fig.suptitle("dA/dt < 0 : gradient descent on area  =  mean curvature flow",
                 fontsize=11, y=0.97)
    frames = []
    for f in range(n_move + n_hold):
        u = min(f, n_move - 1) / (n_move - 1)
        for ax, run in ((axL, sub), (axR, sup)):
            it_list, V_list = run["snaps"]
            si = int(round((len(it_list) - 1) * u ** 1.6))
            it = it_list[si]
            ax.cla()
            note = ("pinch-off!" if run["status"] == "collapsed" and si == len(it_list) - 1
                    else f"area {run['trace']['area'][it]:.3f}")
            draw_film(ax, V_list[si], run["F"], R, run["hr"] * R, zlim,
                      f"h/R = {run['hr']:.2f}   iter {it}   {note}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf.copy()))
    plt.close(fig)
    durations = [int(1000 / fps)] * (len(frames) - 1) + [2500]   # hold the ending
    frames[0].save(outdir / "film_evolution.gif", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0)

    print(f"\nwrote {outdir}/fig1_films.png, fig2_validation.png, "
          f"fig3_bifurcation.png, film_evolution.gif")
    print(f"total wall time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
