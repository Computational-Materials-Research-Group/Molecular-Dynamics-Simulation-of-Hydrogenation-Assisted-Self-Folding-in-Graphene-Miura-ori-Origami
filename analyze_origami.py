#!/usr/bin/env python3
"""
Post-processing for the HAGO run.

Fold angle of each crease, from the dump:
  - crease centre  = centroid of the hydrogenated C atoms in that strip
  - flap vectors   = crease centre -> centroid of C atoms 6-14 A from the
                     crease on each side (picked once, in the flat sheet)
  - fold angle     = 180 - angle(u1, u2)   (0 = flat, 180 = folded flat)

Usage:  python3 analyze_origami.py origami.data origami.dump creases.json
Writes fold_angle.dat, origami_results.png, snapshots.png
"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

data, dump, meta = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
DT_PS = 0.0005   # timestep in ps

def read_data(fn):
    L = open(fn).read().split("Atoms")[1].split("\n")[2:]
    a = np.array([l.split()[:5] for l in L if l.strip()], float)
    return a[np.argsort(a[:, 0])]

def read_dump(fn):
    frames, f = [], open(fn)
    while True:
        l = f.readline()
        if not l: break
        if l.startswith("ITEM: TIMESTEP"):
            step = int(f.readline())
            f.readline(); n = int(f.readline())
            for _ in range(4): f.readline()
            f.readline()
            a = np.array([f.readline().split() for _ in range(n)], float)
            frames.append((step, a[np.argsort(a[:, 0])]))
    return frames

a0 = read_data(data)
isC = a0[:, 1] == 1
xy0 = a0[:, 2:4]
# hydrogenated carbons = C with an H directly above in the as-built sheet
Hxy = a0[~isC, 2:4]
hydC = np.zeros(len(a0), bool)
for h in Hxy:
    d = np.linalg.norm(xy0 - h, axis=1)
    hydC[np.argmin(np.where(isC, d, 1e9))] = True

groups = []
for c in meta["creases"]:
    k = 0 if c["axis"] == "x" else 1
    s = xy0[:, k] - c["pos"]
    strip = hydC & (np.abs(s) < meta["width"])
    left = isC & (s < -6) & (s > -14)
    right = isC & (s > 6) & (s < 14)
    groups.append((strip, left, right))

frames = read_dump(dump)
t, ang = [], []
for step, a in frames:
    r = a[:, 2:5]
    row = []
    for strip, L, R in groups:
        c = r[strip].mean(0)
        u1, u2 = r[L].mean(0) - c, r[R].mean(0) - c
        cosv = u1 @ u2 / np.linalg.norm(u1) / np.linalg.norm(u2)
        row.append(180 - np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    t.append(step * DT_PS); ang.append(row)
t, ang = np.array(t), np.array(ang)
np.savetxt("fold_angle.dat", np.column_stack([t, ang]),
           header="time_ps " + " ".join(f"fold{i+1}_deg" for i in range(ang.shape[1])), fmt="%.3f")

# energies from fix ave/time
e = np.loadtxt("energy.dat")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(ang.shape[1]):
    ax[0].plot(t, ang[:, i], lw=1.6, label=f"crease {i+1}")
ax[0].set(xlabel="time (ps)", ylabel="fold angle (deg)", title="Self-folding at 300 K", ylim=(0, 185))
ax[0].axhline(180, color="0.6", ls=":", lw=1); ax[0].legend(frameon=False)
ax[1].plot(e[:, 0] * DT_PS, e[:, 2] - e[0, 2], lw=1.6, color="#c0392b")
ax[1].set(xlabel="time (ps)", ylabel="ΔE_LJ (eV)", title="van der Waals energy (adhesion when negative)")
for x in ax: x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("origami_results.png", dpi=160)

# four snapshots, side and oblique views
pick = np.linspace(0, len(frames) - 1, 4).astype(int)
fig = plt.figure(figsize=(14, 3.8))
for j, k in enumerate(pick):
    step, a = frames[k]; r = a[:, 2:5]; ty = a[:, 1]
    axx = fig.add_subplot(1, 4, j + 1, projection="3d")
    axx.scatter(*r[ty == 1].T, s=2, c=r[ty == 1, 2], cmap="viridis", depthshade=False)
    axx.scatter(*r[ty == 2].T, s=4, c="crimson", depthshade=False)
    span = np.ptp(r, 0).max() / 2; m = r.mean(0)
    axx.set(xlim=(m[0]-span, m[0]+span), ylim=(m[1]-span, m[1]+span), zlim=(m[2]-span, m[2]+span))
    axx.view_init(elev=15, azim=-70); axx.set_axis_off()
    axx.set_title(f"{step*DT_PS:.0f} ps, fold {ang[k].mean():.0f}°", fontsize=10)
fig.tight_layout(); fig.savefig("snapshots.png", dpi=150)
print("final fold angles (deg):", np.round(ang[-1], 1))
