# Soap films between two rings — a discrete Plateau problem

A self-contained numerical experiment in geometric analysis: find the surface of
least area spanning two coaxial rings of radius $R$ at separation $h$, by
gradient descent on the area of a triangle mesh. Pure `numpy` / `scipy` /
`matplotlib`, ~1300 vertices, a couple of minutes on one CPU core.

The punchline is that the vertex-wise gradient of total mesh area is the
(integrated) **mean-curvature vector**, so gradient descent on area *is* a
discrete **mean curvature flow** — the mesh evolves exactly the way a soap film
drains area, and it converges to a discrete minimal surface whenever one exists.

## The mathematics being verified

- Minimal surfaces of revolution are **catenoids** $r(z) = c\cosh(z/c)$, where
  $c$ solves $c\cosh\!\big(\tfrac{h}{2c}\big) = R$. Below a critical separation
  this has two roots: the larger is the stable, area-minimizing catenoid, the
  smaller an unstable saddle of the area functional.
- Catenoid area: $A = \pi c\,(h + c\sinh(h/c))$.
- At $h^\* = 2\mu R/\cosh\mu$ with $\mu\tanh\mu = 1$ (so $h^\* \approx 1.3255\,R$,
  neck $c^\* \approx 0.5524\,R$) the two branches merge and annihilate — a
  **saddle-node (fold) bifurcation**. For $h > h^\*$ *no* spanning minimal
  surface exists: the film pinches off toward the **Goldschmidt solution**
  (two flat disks).

The flow reproduces all of it: at $h/R = 1$ the relaxed mesh matches the
catenoid's area and neck radius to **0.09 %**, the sweep of final neck radii
lands on the stable branch $c(h)$ to 3–4 decimals, and every supercritical run
pinches off — including the delicate $h/R = 1.35$, just past the fold, where the
pinch suffers critical slowing-down.

## Run it

```bash
python plateau_solver.py        # any Python with numpy, scipy, matplotlib
```

Runtime ≈ 2–3 min (the hero runs, a 12-value sweep across the fold, three PNGs
and one GIF). Everything is deterministic — no seeds, no downloads, no GPU.

## Outputs (written to `figures/`)

| File | What it shows |
|---|---|
| `fig1_films.png` | Converged catenoid at $h/R=1.0$ vs pinch-off at $h/R=1.5$ |
| `fig2_validation.png` | (a) area descent vs theory, (b) relaxed profile on the catenary $r=c\cosh(z/c)$, (c) neck dynamics: settle vs collapse |
| `fig3_bifurcation.png` | Final neck radius across the fold: stable/unstable branches, Goldschmidt line, $h^\*$, and the numerical points |
| `film_evolution.gif` | The flow itself, side by side: existence vs non-existence |

## How it works (`plateau_solver.py`)

- `cylinder_mesh` — triangulated cylinder between the rings; boundary rows are
  Dirichlet (the wire).
- `area_and_grad` — for a triangle $(a,b,c)$ with unit normal $\hat n$,
  $\nabla_a\,\mathrm{Area} = \tfrac12\,\hat n \times (c-b)$; scatter-summed
  with `bincount`.
- `relax` — explicit Euler on $\dot V = -\nabla A / m$ (lumped barycentric
  masses), heat-equation step $dt \sim 0.2\,\ell_{\min}^2$ refreshed as the
  mesh deforms, displacement cap, and three exits: converged / collapsed
  (neck $< 0.06R$) / iteration cap.
- `catenoid_c`, `catenoid_area`, `critical_ratio` — the closed-form theory the
  numerics is checked against.

## Where to take it (course project seeds)

- Replace the ring boundary by a **disk-topology mesh over an Enneper-type
  wire** and watch the flow find the Enneper surface.
- Swap explicit descent for the **Pinkall–Polthier cotangent-Laplacian
  iteration** (solve $\Delta_S x = 0$ per step) and compare convergence rates.
- Push past area: **Willmore flow** ($\int H^2$), or learn the film with a
  PINN and compare against this mesh ground truth.

*MM845 — AI for Geometry (IMECC/UNICAMP, 2S 2026). Demo built live with an AI
coding agent.*
