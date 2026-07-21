"""Aero multi-tier parametric case generator (2D + 3D urban RANS, OpenFOAM 14).

Wraps the validated case_gen_3d.build_case pipeline with:
  * runtime-tier presets (1/3/5/10/24/48/72 h) calibrated on ssh:openfoam
    (fit exec_s = 7.16e-6 * cells^1.27, R^2=0.97), plus a fast 'dev' tier
    (~40k cells, ~5 s) for the tractable dataset we actually solve here;
  * domain that scales with grid to hold physical cell size ~constant
    (dx ~ 0.05 m), so larger tiers are physically bigger cities, not just
    finer meshes of the same box;
  * per-case randomized ABL (Uref, alpha, nu) and building layout, with the
    building count and footprint scaling with domain area;
  * dim='3d' (full ABL) or dim='2d' (thin extruded slab, nz=8, full-height
    buildings) — matching the unified 2D/3D ONNX convention.

API:
  preset(tier)                      -> dict(grid, note, est_runtime_s, ...)
  make_case_params(tier, case_id, dim='3d', base_seed=...) -> (params, grid)
  build(case_dir, tier, case_id, dim)  -> writes case tree, returns (params,grid)

CLI: python aero_case_generator.py <case_dir> <tier> <case_id> <dim>
"""
import os, math, json, random
import numpy as np
import case_gen_3d

DX = 0.05  # target physical cell size (m); base 60x30x24 over 3.0x1.5x1.0

# Calibrated tier -> grid (nx,ny,nz), aspect ~2:1:0.8. 'dev' is the tractable
# tier we solve on this host; the h-tiers ship as ready-to-launch presets.
TIER_GRIDS = {
    'dev':  (60,  30,  24),    # ~42k cells, ~5 s  -- realistic dataset tier
    '1h':   (324, 162, 130),   # ~6.8M
    '3h':   (432, 216, 173),   # ~16.1M
    '5h':   (492, 246, 197),   # ~23.8M
    '10h':  (592, 296, 237),   # ~41.5M
    '24h':  (744, 372, 298),   # ~82.5M
    '48h':  (892, 446, 357),   # ~142M
    '72h':  (992, 496, 397),   # ~195M
}
# power-law fit from runtime_calibration.json
_FIT_A, _FIT_B = 7.156e-06, 1.2736
_ITER_A, _ITER_B = 8.143, 0.2652


def preset(tier):
    if tier not in TIER_GRIDS:
        raise KeyError(f"unknown tier {tier!r}; choose from {list(TIER_GRIDS)}")
    nx, ny, nz = TIER_GRIDS[tier]
    cells = nx * ny * nz
    est_s = _FIT_A * cells ** _FIT_B
    est_it = int(round(_ITER_A * cells ** _ITER_B))
    return dict(tier=tier, grid=(nx, ny, nz), block_cells=cells,
                domain=(round(nx*DX, 3), round(ny*DX, 3), round(nz*DX, 3)),
                est_runtime_s=round(est_s, 1), est_runtime_h=round(est_s/3600, 3),
                est_iters=est_it)


def _for_2d(nx, ny, nz):
    """Thin extruded slab: keep x,y resolution, collapse z to 8 cells."""
    return nx, ny, 8


def make_case_params(tier, case_id, dim='3d', base_seed=1234):
    """Return (params, grid) for a solvable urban case at the given tier.

    Geometry and ABL are randomized per (tier, case_id, dim). Building count
    and footprint scale with domain area so larger cities stay realistically
    populated rather than sparse.
    """
    nx, ny, nz = TIER_GRIDS[tier]
    if dim == '2d':
        nx, ny, nz = _for_2d(nx, ny, nz)
    Lx, Ly, Lz = nx * DX, ny * DX, nz * DX

    seed = base_seed + hash((tier, dim)) % 100000 + case_id
    random.seed(seed); np.random.seed(seed % (2**32))

    Uref  = float(np.random.uniform(0.8, 1.2))
    alpha = float(np.random.uniform(0.15, 0.30))
    nu    = float(np.random.uniform(1.2e-3, 2.0e-3))

    # building count scales with plan area relative to the base 3.0x1.5 domain
    area_ratio = (Lx * Ly) / (3.0 * 1.5)
    n_base = np.random.randint(2, 5)
    n_buildings = max(2, int(round(n_base * area_ratio)))
    n_buildings = min(n_buildings, 400)  # keep topoSet tractable

    # building footprint stays ~constant physical size (dense-urban blocks);
    # count scales with area, so larger cities are more populated, not chunkier
    buildings, occupied = [], []
    margin_x = 0.15 * Lx
    for _ in range(n_buildings):
        for _attempt in range(12):
            bx = np.random.uniform(0.30, 0.80)
            by = np.random.uniform(0.30, 0.60)
            x0 = np.random.uniform(margin_x, Lx - margin_x - bx)
            y0 = np.random.uniform(0.1 * Ly, 0.9 * Ly - by)
            x1, y1 = x0 + bx, y0 + by
            if dim == '2d':
                zt = Lz                      # full-height (extruded) obstacle
            else:
                zt = float(np.random.uniform(0.25, 0.65) * Lz / 1.0)
                zt = min(zt, 0.7 * Lz)
            overlaps = any(not (x1 < ox0 or x0 > ox1 or y1 < oy0 or y0 > oy1)
                           for ox0, ox1, oy0, oy1 in occupied)
            if not overlaps and x0 > 0 and y0 > 0:
                buildings.append((round(x0,4), round(x1,4), round(y0,4), round(y1,4), round(zt,4)))
                occupied.append((x0, x1, y0, y1))
                break

    # iteration cap scaled from calibration (headroom above expected iters)
    est_it = int(round(_ITER_A * (nx*ny*nz) ** _ITER_B))
    endTime = int(max(120, est_it * 1.5))

    params = dict(Lx=round(Lx,4), Ly=round(Ly,4), Lz=round(Lz,4),
                  Uref=Uref, Zref=round(0.5*Lz,4), alpha=alpha, nu=nu,
                  buildings=buildings, endTime=endTime, I=0.10, resid_tol=1e-3,
                  tier=tier, dim=dim, case_id=case_id)
    grid = dict(nx=nx, ny=ny, nz=nz)
    return params, grid


def build(case_dir, tier, case_id, dim='3d', base_seed=1234):
    params, grid = make_case_params(tier, case_id, dim, base_seed)
    case_gen_3d.build_case(case_dir, params, grid)
    return params, grid


if __name__ == '__main__':
    import sys
    case_dir = sys.argv[1] if len(sys.argv) > 1 else 'aero_case'
    tier     = sys.argv[2] if len(sys.argv) > 2 else 'dev'
    case_id  = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dim      = sys.argv[4] if len(sys.argv) > 4 else '3d'
    p, g = build(case_dir, tier, case_id, dim)
    print(json.dumps(dict(tier=tier, dim=dim, case_id=case_id, grid=g,
                          n_buildings=len(p['buildings']),
                          domain=[p['Lx'], p['Ly'], p['Lz']],
                          Uref=p['Uref'], alpha=p['alpha'], endTime=p['endTime']),
                     indent=2))
