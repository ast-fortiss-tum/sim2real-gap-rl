#!/usr/bin/env python3
"""
Plot the first three laps (≈1080°) of every run in *dataset_final*
and print the mean L-2 lateral error (cm) w.r.t. the centre-line.

Run:
    python plot_all_circuits.py
"""

import os, json, ast, numpy as np, matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATASET_DIR    = "/home/cubos98/catkin_ws/src/Vehicle/results_reality/dataset_final"
OUT_FIG        = "all_runs_circuit.png"
SCALE          = 25          # multiply every (x, y) by this factor
TRIM_LIMIT_DEG = 1080        # analyse first 3 laps

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_json(p):                   # tiny IO helper
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def scale_pts(seq,f=SCALE):         # multiply all xy by f
    return [[x*f,y*f] for x,y in seq]

def collect_runs(root):             # (run_name, poses_json_path)
    runs=[]
    for entry in sorted(os.listdir(root)):
        full=os.path.join(root,entry)
        if os.path.isdir(full):
            poses=[f for f in os.listdir(full) if f.endswith("_poses.json")]
            if poses:
                runs.append((entry, os.path.join(full,poses[0])))
    return runs

def centre_line(l0,l1):
    n=min(len(l0),len(l1))
    return [( (l0[i][0]+l1[i][0])/2, (l0[i][1]+l1[i][1])/2 ) for i in range(n)]

def mean_l2_error(poses, centre):
    errs=[]
    for x,y in poses:
        d=[np.hypot(x-cx, y-cy) for cx,cy in centre]
        errs.append(min(d))           # closest centre sample
    return np.mean(errs) if errs else 0.0

def unwrap_angle(poses,centre):
    cx,cy=centre
    ang=np.unwrap(np.arctan2([y-cy for _,y in poses],
                             [x-cx for x,_ in poses]))
    return ang-ang[0]

def trim_laps(poses, centre, limit=TRIM_LIMIT_DEG):
    if len(poses)<2: return poses
    deg=np.degrees(unwrap_angle(poses,centre))
    idx=np.argmax(np.abs(deg)>limit)
    return poses if idx==0 else poses[:idx]

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

runs=collect_runs(DATASET_DIR)
if not runs:
    raise SystemExit("No *_poses.json found")

fig,axes=plt.subplots(2,3,figsize=(15,8)); axes=axes.ravel()

print("\nrun-name                mean-L2-error [cm]")
print("-------------------------------------------")

for ax,(run, pf) in zip(axes, runs):
    data=load_json(pf)
    lanes_raw, poses_raw = data["lanes"], data["pose"]
    if isinstance(poses_raw,str): poses_raw=ast.literal_eval(poses_raw)
    if isinstance(lanes_raw[0],str): lanes_raw=[ast.literal_eval(l) for l in lanes_raw]

    poses=scale_pts(poses_raw)
    lanes=[scale_pts(l) for l in lanes_raw]

    # geometry
    centre_geom=(np.mean([p[0] for p in lanes[0]+lanes[1]]),
                 np.mean([p[1] for p in lanes[0]+lanes[1]]))
    poses_trim = trim_laps(poses, centre_geom)
    centreline = centre_line(lanes[0], lanes[1])
    l2 = (1.0 - mean_l2_error(poses_trim, centreline)/100.0) *100.0 # ACOMPLISHON MEASUREMENT!!!

    print(f"{run:<24} {l2:10.2f}")

    # ---------- plot ----------
    ax.set_facecolor("dimgray"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    lane_c=["white","white","yellow"]+["gray"]*max(0,len(lanes)-3)
    lane_s=["-","-","--"]+["-"]*max(0,len(lanes)-3)
    for ln,c,s in zip(lanes,lane_c,lane_s):
        xs,ys=zip(*ln); ax.plot(xs,ys,color=c,ls=s,lw=1)
    xs,ys=zip(*poses_trim)
    ax.plot(xs,ys,color="red",lw=1.2)
    m=100
    ax.set_xlim(min(xs)-m,max(xs)+m); ax.set_ylim(min(ys)-m,max(ys)+m)
    ax.set_title(f"{run}\nSuccess measure = {l2:.1f}%",fontsize=9)

for extra in axes[len(runs):]:
    extra.set_visible(False)

fig.suptitle("First 3 laps – mean L-2 distance to centre-line", fontsize=14)
fig.tight_layout(rect=[0,0.03,1,0.92])
fig.savefig(OUT_FIG,dpi=150)
print("\nfigure saved →",OUT_FIG)
