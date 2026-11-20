#!/usr/bin/env python3
"""
Miura-ori graphene origami (HAGO) -> LAMMPS data file.

Crease pattern (in the flat sheet, x = armchair, y = zigzag):
  - straight creases : vertical lines x = i*a,           i = 1 .. nx-1
  - zig-zag creases  : y = j*b + z(x),                   j = 1 .. ny-1
                       z(x) is a triangle wave, +h/2 at even i, -h/2 at odd i,
                       h = a / tan(alpha)   (alpha = Miura sector angle)
  The creases cut the sheet into nx x ny parallelogram panels.

Mountain / valley assignment (standard Miura, flat-foldable, Maekawa 3:1
at every vertex):
  zig-zag crease j          : M if j even, V if j odd
  straight segment (i, row r): M if (i + r) even, V if odd

How a crease type is written chemically:
  the sheet bends AWAY from the hydrogenated face (H ends up on the convex
  side), so  mountain = H on +z,  valley = H on -z.
  A carbon that falls inside both an M and a V strip (at a vertex) gets no H.

Units metal, atom_style atomic, type 1 = C, type 2 = H.
Writes the data file and miura.json (panels, creases, expected M/V) for
analyze_miura.py.
"""
import argparse, json, math
import numpy as np

A_CC, D_CH = 1.42, 1.09

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=4, help="panel columns")
p.add_argument("--ny", type=int, default=4, help="panel rows")
p.add_argument("--a", type=float, default=25.0, help="panel width along x (A)")
p.add_argument("--b", type=float, default=25.0, help="panel height along y (A)")
p.add_argument("--alpha", type=float, default=60.0, help="Miura sector angle (deg)")
p.add_argument("--width", type=float, default=2.8, help="hydrogenated strip width (A)")
p.add_argument("--seed", type=int, default=1)
p.add_argument("-o", "--out", default="miura.data")
a = p.parse_args()

h = a.a / math.tan(math.radians(a.alpha))
LX, LY = a.nx * a.a, a.ny * a.b
assert a.b > h, "rows overlap: increase --b or --alpha"

# ---- graphene sheet ---------------------------------------------------------
cx, cy = 3 * A_CC, math.sqrt(3) * A_CC
basis = np.array([[0, 0], [0.5 * A_CC, 0.5 * cy], [1.5 * A_CC, 0.5 * cy], [2 * A_CC, 0]])
C = np.array([[i * cx + bb[0], j * cy + bb[1]]
              for i in range(int(LX / cx) + 1) for j in range(int(LY / cy) + 1) for bb in basis])
C = C[(C[:, 0] <= LX) & (C[:, 1] <= LY)]
x, y = C[:, 0], C[:, 1]

def zig(xx):                       # triangle wave, +h/2 at even multiples of a
    t = xx / a.a
    i = np.floor(t); f = t - i
    s = np.where(i % 2 == 0, 1.0, -1.0)
    return s * h / 2 * (1 - 2 * f)

# panel index of every atom
col = np.clip((x // a.a).astype(int), 0, a.nx - 1)
row = np.clip(np.floor((y - zig(x)) / a.b).astype(int), 0, a.ny - 1)

def seg_dist(P, A, B):
    AB = B - A; t = np.clip(((P - A) @ AB) / (AB @ AB), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)

M = np.zeros(len(C), bool); V = np.zeros(len(C), bool)
half = 0.5 * a.width
creases = []                       # (kind, panelA, panelB, type +1 M / -1 V)

# zig-zag creases
for j in range(1, a.ny):
    typ = 1 if j % 2 == 0 else -1
    d = np.full(len(C), 1e9)
    for i in range(a.nx):
        A = np.array([i * a.a, j * a.b + (h / 2 if i % 2 == 0 else -h / 2)])
        B = np.array([(i + 1) * a.a, j * a.b + (h / 2 if (i + 1) % 2 == 0 else -h / 2)])
        d = np.minimum(d, seg_dist(C, A, B))
        creases.append(("zig", [i, j - 1], [i, j], typ))
    (M if typ > 0 else V)[d <= half] = True

# straight creases, segment by segment
for i in range(1, a.nx):
    for r in range(a.ny):
        typ = 1 if (i + r) % 2 == 0 else -1
        rr = np.clip(np.floor((y - zig(np.full_like(x, i * a.a))) / a.b), 0, a.ny - 1)
        sel = (np.abs(x - i * a.a) <= half) & (rr == r)
        (M if typ > 0 else V)[sel] = True
        creases.append(("str", [i - 1, r], [i, r], typ))

both = M & V
M &= ~both; V &= ~both
rng = np.random.default_rng(a.seed)
z = rng.normal(0, 0.05, len(C))
Cxyz = np.column_stack([C, z])
H = np.vstack([Cxyz[M] + [0, 0, D_CH], Cxyz[V] - [0, 0, D_CH]])

# panel cores for analysis: carbons at least 3 A away from any crease strip
from scipy.spatial import cKDTree
strip = M | V | both
dcrease, _ = cKDTree(C[strip]).query(C)
core = (~strip) & (dcrease > 3.0)

atoms = [(1, *r) for r in Cxyz] + [(2, *r) for r in H]
pad = 60.0
with open(a.out, "w") as f:
    f.write(f"Miura-ori HAGO graphene, {a.nx}x{a.ny} panels, a={a.a} b={a.b} alpha={a.alpha} strip={a.width}\n\n")
    f.write(f"{len(atoms)} atoms\n2 atom types\n\n")
    f.write(f"{-pad:.3f} {LX+pad:.3f} xlo xhi\n{-pad:.3f} {LY+pad:.3f} ylo yhi\n{-pad:.3f} {pad:.3f} zlo zhi\n")
    f.write("\nMasses\n\n1 12.011\n2 1.008\n\nAtoms # atomic\n\n")
    for k, (t, xx, yy, zz) in enumerate(atoms, 1):
        f.write(f"{k} {t} {xx:.5f} {yy:.5f} {zz:.5f}\n")

ids = np.arange(1, len(C) + 1)
panels = {f"{i},{r}": ids[core & (col == i) & (row == r)].tolist() for i in range(a.nx) for r in range(a.ny)}
json.dump({"nx": a.nx, "ny": a.ny, "a": a.a, "b": a.b, "alpha": a.alpha, "h": h, "LX": LX, "LY": LY,
           "panels": panels,
           "creases": [{"kind": k, "A": A_, "B": B_, "type": t} for k, A_, B_, t in creases]},
          open("miura.json", "w"))
print(f"Miura {a.nx}x{a.ny} panels | sheet {LX/10:.1f} x {LY/10:.1f} nm | C {len(C)} | "
      f"H top (M) {M.sum()} | H bottom (V) {V.sum()} | vertex C without H {both.sum()} | "
      f"creases {len(creases)} -> {a.out}")
