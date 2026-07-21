#!/usr/bin/env python3
"""
Materialize the Aero runtime-tier cases (1/3/5/10/24/48/72 h) as ready-to-run
OpenFOAM-14 models under a target root -- WITHOUT solving them.

For each tier and dim (3d + 2d slab companion):
  <root>/<dim>/aero_<tier>_<dim>/
      0/ constant/ system/         (OF14 case tree; uniform fields, no mesh yet)
      Allrun                        (self-contained blockMesh->...->foamRun script)
      MODEL_INFO.json               (grid, cells, est runtime/iters, ABL params)

Also writes at <root>:
  manifest.json          machine-readable summary of every model
  RUNTIME_REPORT.md      human report of expected solve durations
  runtime_tiers.csv      tier, dim, cells, est_runtime_s/h, est_iters

The hour labels are the 3D runtime calibration (exec_s = 7.156e-6 * cells^1.2736,
R^2=0.97, fit on ssh:openfoam). The 2D slab shares the x-y grid with nz=8, so it
is a much smaller/cheaper companion (its own est runtime is reported separately).

Usage: python3 materialize_tiers.py <root>
"""
import os, sys, json, stat
import aero_case_generator as acg

TIERS = ['1h', '3h', '5h', '10h', '24h', '48h', '72h']
DIMS = ['3d', '2d']

ALLRUN = """#!/bin/bash
# Aero {tier} {dim} model -- full OpenFOAM-14 urban RANS pipeline.
# Mesh (blockMesh -> topoSet -> subsetMesh -> name buildings wall) then solve
# (foamRun incompressibleFluid, steady kEpsilon SIMPLE). Run from this dir:
#   source /opt/openfoam14/etc/bashrc && ./Allrun
# WARNING: this tier is calibrated to ~{est_h} h of foamRun wall-clock
# ({cells:,} cells). Ensure adequate RAM/time before launching.
set -e
cd "$(dirname "$0")"

blockMesh                           > log.blockMesh  2>&1
topoSet                             > log.topoSet    2>&1
subsetMesh -cellSet fluid -noFields > log.subsetMesh 2>&1
sed -i 's/oldInternalFaces/buildings/' constant/polyMesh/boundary
foamDictionary constant/polyMesh/boundary -entry entry0/buildings/type -set wall > /dev/null 2>&1
checkMesh                           > log.checkMesh  2>&1

echo "mesh done: $(date)"
foamRun                             > log.foamRun    2>&1
RC=$?
echo "foamRun_exit=$RC"
echo "n_outer_iters=$(grep -c 'Solving for p,' log.foamRun)"
echo "final_time=$(foamListTimes 2>/dev/null | tail -1)"
exit $RC
"""

# est 2D runtime from the same power-law fit applied to the 2D cell count
def est_2d(cells):
    return acg._FIT_A * cells ** acg._FIT_B, int(round(acg._ITER_A * cells ** acg._ITER_B))


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'aero_scratch/tiers')
    os.makedirs(root, exist_ok=True)
    manifest = []

    for tier in TIERS:
        pinfo = acg.preset(tier)  # 3D calibrated preset
        for dim in DIMS:
            name = f'aero_{tier}_{dim}'
            cdir = os.path.join(root, dim, name)
            os.makedirs(cdir, exist_ok=True)
            params, grid = acg.build(cdir, tier, case_id=0, dim=dim)
            cells = grid['nx'] * grid['ny'] * grid['nz']
            if dim == '3d':
                est_s, est_it = pinfo['est_runtime_s'], pinfo['est_iters']
            else:
                est_s, est_it = est_2d(cells)
            est_s = round(est_s, 1)

            info = dict(
                model=name, tier=tier, dim=dim,
                grid=[grid['nx'], grid['ny'], grid['nz']], cells=cells,
                domain=[params['Lx'], params['Ly'], params['Lz']],
                n_buildings=len(params['buildings']),
                Uref=params['Uref'], alpha=params['alpha'], nu=params['nu'],
                endTime=params['endTime'],
                est_runtime_s=est_s, est_runtime_h=round(est_s / 3600, 3),
                est_iters=est_it,
                note=('3D runtime-calibrated tier' if dim == '3d'
                      else '2D extruded slab (nz=8), same x-y grid; cheaper companion'))
            with open(os.path.join(cdir, 'MODEL_INFO.json'), 'w') as f:
                json.dump(info, f, indent=2)

            ap = os.path.join(cdir, 'Allrun')
            with open(ap, 'w') as f:
                f.write(ALLRUN.format(tier=tier, dim=dim, cells=cells,
                                      est_h=round(est_s / 3600, 2)))
            os.chmod(ap, os.stat(ap).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            manifest.append(info)
            print(f"  {name:16s} cells={cells:>12,}  est={est_s/3600:7.2f} h  "
                  f"iters~{est_it}  buildings={len(params['buildings'])}")

    with open(os.path.join(root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # CSV
    with open(os.path.join(root, 'runtime_tiers.csv'), 'w') as f:
        f.write('model,tier,dim,nx,ny,nz,cells,est_runtime_s,est_runtime_h,est_iters,n_buildings\n')
        for m in manifest:
            g = m['grid']
            f.write(f"{m['model']},{m['tier']},{m['dim']},{g[0]},{g[1]},{g[2]},{m['cells']},"
                    f"{m['est_runtime_s']},{m['est_runtime_h']},{m['est_iters']},{m['n_buildings']}\n")

    # Markdown report
    md = ['# Aero runtime-tier models — expected solver durations', '',
          'Ready-to-run OpenFOAM-14 urban RANS cases materialized (mesh **not** built, '
          'solver **not** run). Each `<dim>/aero_<tier>_<dim>/` holds a full case tree, a '
          'self-contained `Allrun` script, and `MODEL_INFO.json`.', '',
          'Runtime estimates come from the ssh:openfoam calibration '
          '`exec_s = 7.156e-6 · cells^1.2736` (R²=0.97). Hour labels are the **3D** tiers; '
          'the 2D slab (nz=8) shares the x-y grid and is a much cheaper companion.', '',
          '## 3D runtime tiers', '',
          '| tier | grid | cells | est. foamRun wall-clock | est. outer iters | buildings |',
          '|------|------|-------|--------------------------|------------------|-----------|']
    for m in manifest:
        if m['dim'] != '3d':
            continue
        g = m['grid']
        md.append(f"| **{m['tier']}** | {g[0]}×{g[1]}×{g[2]} | {m['cells']:,} | "
                  f"{m['est_runtime_h']:.2f} h ({m['est_runtime_s']:,.0f} s) | {m['est_iters']} | {m['n_buildings']} |")
    md += ['', '## 2D slab companions (nz=8, same x-y grid)', '',
           '| tier | grid | cells | est. foamRun wall-clock | est. outer iters | buildings |',
           '|------|------|-------|--------------------------|------------------|-----------|']
    for m in manifest:
        if m['dim'] != '2d':
            continue
        g = m['grid']
        md.append(f"| {m['tier']} | {g[0]}×{g[1]}×{g[2]} | {m['cells']:,} | "
                  f"{m['est_runtime_s']:,.0f} s | {m['est_iters']} | {m['n_buildings']} |")
    md += ['', '## How to run a model', '',
           '```bash', 'source /opt/openfoam14/etc/bashrc',
           'cd 3d/aero_1h_3d && ./Allrun     # meshes then solves this tier', '```', '',
           '> The Aero ML warm-start seeds `0/U` and `0/p` before `foamRun` to cut outer '
           'iterations; the numerics (schemes, solvers, tolerances) are identical to this '
           'cold-start baseline.', '']
    with open(os.path.join(root, 'RUNTIME_REPORT.md'), 'w') as f:
        f.write('\n'.join(md))

    print(f"\nMaterialized {len(manifest)} models under {root}")
    print("wrote manifest.json, runtime_tiers.csv, RUNTIME_REPORT.md")


if __name__ == '__main__':
    main()
