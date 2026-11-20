#!/usr/bin/env python3
"""
Analysis of the Miura-ori HAGO run.

For every crease between two panels A and B:
  - panel normal  : plane fit (SVD) to the panel-core carbons, kept pointing
                    the same way as in the previous frame (starts at +z)
  - fold angle    : angle between the two panel normals (0 = flat)
  - fold sense    : s = -( nA.(cB-cA) + nB.(cA-cB) ),  s > 0 -> mountain
  - signed angle  : +angle if the crease folds the way the pattern asks
                    (M as mountain, V as valley), -angle if it folds wrong

Footprint: x and y extent of the panel centroids relative to the flat sheet.
A Miura sheet contracts in BOTH x and y as it folds.

Usage:  python3 analyze_miura.py miura.data origami.dump miura.json
Writes  miura_folds.dat, miura_results.png, miura_snapshots.png
"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dump, meta = sys.argv[2], json.load(open(sys.argv[3]))
DT_PS = 0.0005

def frames_of(fn):
    f = open(fn)
    while True:
        l = f.readline()
        if not l: return
        if l.startswith("ITEM: TIMESTEP"):
            step = int(f.readline()); f.readline(); n = int(f.readline())
            for _ in range(4): f.readline()
            cols = f.readline().split()[2:]
            ix, it = cols.index("x"), cols.index("type")
            rows = [f.readline().split() for _ in range(n)]
            ids = np.array([int(r[0]) for r in rows])
            o = np.argsort(ids)
            typ = np.array([int(r[it]) for r in rows])[o]
            xyz = np.array([[float(v) for v in r[ix:ix + 3]] for r in rows])[o]
            yield step, typ, xyz

panels = {k: np.array(v) - 1 for k, v in meta["panels"].items()}
keys = list(panels)
cr = meta["creases"]

T, ang, sgn_ok, foot, snaps = [], [], [], [], []
prevN = {k: np.array([0, 0, 1.0]) for k in keys}
allframes = list(frames_of(dump))
pick = set(np.linspace(0, len(allframes) - 1, 4).astype(int))
for fi, (step, typ, r) in enumerate(allframes):
    cen, nrm = {}, {}
    for k in keys:
        P = r[panels[k]]; c = P.mean(0)
        n = np.linalg.svd(P - c)[2][2]
        if n @ prevN[k] < 0: n = -n
        cen[k], nrm[k], prevN[k] = c, n, n
    row = []
    for c in cr:
        A, B = f"{c['A'][0]},{c['A'][1]}", f"{c['B'][0]},{c['B'][1]}"
        th = np.degrees(np.arccos(np.clip(nrm[A] @ nrm[B], -1, 1)))
        s = -(nrm[A] @ (cen[B] - cen[A]) + nrm[B] @ (cen[A] - cen[B]))
        row.append(th if np.sign(s) == c["type"] else -th)
    C = np.array([cen[k] for k in keys])
    T.append(step * DT_PS); ang.append(row); foot.append(np.ptp(C, 0))
    if fi in pick: snaps.append((step, typ, r))
T, ang, foot = np.array(T), np.array(ang), np.array(foot)
foot = foot / foot[0]
kind = np.array([c["kind"] for c in cr])
np.savetxt("miura_folds.dat", np.column_stack([T, ang]),
           header="time_ps " + " ".join(f"{c['kind']}_{c['A'][0]}{c['A'][1]}-{c['B'][0]}{c['B'][1]}_{'M' if c['type']>0 else 'V'}"
                                        for c in cr), fmt="%.2f")

e = np.loadtxt("energy.dat")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for kd, col, lab in [("zig", "#1f77b4", "zig-zag creases"), ("str", "#ff7f0e", "straight creases")]:
    m = ang[:, kind == kd]
    ax[0].plot(T, m.mean(1), color=col, lw=1.8, label=f"{lab} (mean)")
    ax[0].fill_between(T, m.min(1), m.max(1), color=col, alpha=0.15)
ax[0].axhline(0, color="0.5", lw=0.8)
ax[0].set(xlabel="time (ps)", ylabel="signed fold angle (deg)",
          title="Fold angle (+ = folds as M/V pattern asks)")
ax[0].legend(frameon=False, fontsize=8)
ok = (ang > 10).sum(1) / ang.shape[1] * 100
ax[1].plot(T, foot[:, 0], lw=1.8, label="x extent / flat")
ax[1].plot(T, foot[:, 1], lw=1.8, label="y extent / flat")
ax[1].plot(T, ok / 100, lw=1.4, color="0.3", ls="--", label="fraction of creases folded correctly")
ax[1].set(xlabel="time (ps)", ylim=(0, 1.05), title="Footprint contraction")
ax[1].legend(frameon=False, fontsize=8)
ax[2].plot(e[:, 0] * DT_PS, e[:, 2] - e[0, 2], color="#c0392b", lw=1.6)
ax[2].set(xlabel="time (ps)", ylabel="ΔE_LJ (eV)", title="van der Waals energy")
for x in ax: x.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("miura_results.png", dpi=160)

fig = plt.figure(figsize=(16, 4.2))
for j, (step, typ, r) in enumerate(snaps):
    axx = fig.add_subplot(1, 4, j + 1, projection="3d")
    c = r[typ == 1]
    axx.scatter(*c.T, s=1.2, c=c[:, 2], cmap="viridis", depthshade=False)
    hh = r[typ == 2]; axx.scatter(*hh.T, s=1.5, c="crimson", depthshade=False)
    span = np.ptp(r, 0).max() / 2; m = r.mean(0)
    axx.set(xlim=(m[0]-span, m[0]+span), ylim=(m[1]-span, m[1]+span), zlim=(m[2]-span, m[2]+span))
    axx.view_init(elev=35, azim=-60); axx.set_axis_off()
    k = int(np.argmin(np.abs(T - step * DT_PS)))
    axx.set_title(f"{step*DT_PS:.0f} ps | {ok[k]:.0f}% creases correct", fontsize=10)
fig.tight_layout(); fig.savefig("miura_snapshots.png", dpi=150)
print(f"final: mean |fold| zig {np.abs(ang[-1, kind=='zig']).mean():.0f} deg, "
      f"str {np.abs(ang[-1, kind=='str']).mean():.0f} deg | correct {ok[-1]:.0f}% | "
      f"footprint x {foot[-1,0]:.2f}, y {foot[-1,1]:.2f}")
