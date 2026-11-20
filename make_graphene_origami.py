#!/usr/bin/env python3
"""
Hydrogenation-assisted graphene origami (HAGO) -> LAMMPS data file.

A rectangular single-layer graphene sheet is built with the armchair
direction along x and the zigzag direction along y. Crease lines are made
by putting H atoms on ONE side (+z) of the sheet, on every carbon inside a
narrow strip. Those carbons turn sp3, the strip buckles, and the sheet
folds itself along the strip at 300 K.

Units: metal (Angstrom).  atom_style atomic.  Type 1 = C, type 2 = H.

Patterns
  single : one crease across the middle (sheet folds in half)
  tube   : --ncrease parallel creases, evenly spaced (sheet rolls into a
           polygonal tube / box)
  cross  : one crease along x and one along y (four-flap fold)

Also writes creases.json, which the analysis script uses to measure the
fold angle of each crease.
"""
import argparse, json, math
import numpy as np

A_CC = 1.42     # C-C bond length (A)
D_CH = 1.09     # C-H bond length (A)

p = argparse.ArgumentParser()
p.add_argument("--lx", type=float, default=80.0, help="sheet length along x, armchair (A)")
p.add_argument("--ly", type=float, default=40.0, help="sheet width along y, zigzag (A)")
p.add_argument("--pattern", choices=["single", "tube", "cross"], default="single")
p.add_argument("--ncrease", type=int, default=3, help="number of creases for 'tube'")
p.add_argument("--width", type=float, default=2.6,
               help="width of each hydrogenated strip (A); wider strip = larger fold angle")
p.add_argument("--seed", type=int, default=1)
p.add_argument("-o", "--out", default="origami.data")
a = p.parse_args()

# ---- graphene, rectangular 4-atom cell --------------------------------------
cx, cy = 3 * A_CC, math.sqrt(3) * A_CC
basis = np.array([[0.0, 0.0], [0.5 * A_CC, 0.5 * cy], [1.5 * A_CC, 0.5 * cy], [2 * A_CC, 0.0]])
nx, ny = int(round(a.lx / cx)), int(round(a.ly / cy))
C = np.array([[i * cx + b[0], j * cy + b[1]] for i in range(nx) for j in range(ny) for b in basis])
C -= C.min(axis=0)
Lx, Ly = C[:, 0].max(), C[:, 1].max()

# ---- crease lines -----------------------------------------------------------
# each crease: axis ('x' = line parallel to y at fixed x, 'y' = line parallel to x)
creases = []
if a.pattern == "single":
    creases.append(("x", 0.5 * Lx))
elif a.pattern == "tube":
    for k in range(a.ncrease):
        creases.append(("x", Lx * (k + 1) / (a.ncrease + 1)))
elif a.pattern == "cross":
    creases += [("x", 0.5 * Lx), ("y", 0.5 * Ly)]

def in_strip(xy, ax, pos):
    c = xy[:, 0] if ax == "x" else xy[:, 1]
    return np.abs(c - pos) <= 0.5 * a.width + 1e-6

hyd = np.zeros(len(C), bool)
for ax, pos in creases:
    hyd |= in_strip(C, ax, pos)

# tiny random z so the flat sheet has a way out of plane
rng = np.random.default_rng(a.seed)
z = rng.normal(0.0, 0.05, len(C))
Cxyz = np.column_stack([C, z])
Hxyz = Cxyz[hyd] + np.array([0, 0, D_CH])     # all H on the +z side

atoms = [(1, *r) for r in Cxyz] + [(2, *r) for r in Hxyz]
pad = 60.0
lo = np.array([-pad, -pad, -pad]); hi = np.array([Lx + pad, Ly + pad, pad])

with open(a.out, "w") as f:
    f.write(f"HAGO graphene origami, pattern={a.pattern}, creases={len(creases)}, strip={a.width} A\n\n")
    f.write(f"{len(atoms)} atoms\n2 atom types\n\n")
    for d, (l, h) in zip("xyz", zip(lo, hi)):
        f.write(f"{l:.4f} {h:.4f} {d}lo {d}hi\n")
    f.write("\nMasses\n\n1 12.011\n2 1.008\n\nAtoms # atomic\n\n")
    for i, (t, x, y, zz) in enumerate(atoms, 1):
        f.write(f"{i} {t} {x:.5f} {y:.5f} {zz:.5f}\n")

# metadata for analysis: crease positions and which C ids sit on each flap
meta = {"Lx": Lx, "Ly": Ly, "width": a.width, "pattern": a.pattern,
        "nC": int(len(C)), "nH": int(hyd.sum()),
        "creases": [{"axis": ax, "pos": pos} for ax, pos in creases]}
json.dump(meta, open("creases.json", "w"), indent=1)
print(f"sheet {Lx/10:.1f} x {Ly/10:.1f} nm | C {len(C)} | H {hyd.sum()} | "
      f"creases {len(creases)} ({a.pattern}) | strip {a.width} A -> {a.out}")
