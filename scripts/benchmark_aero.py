#!/usr/bin/env python3
"""
Aero warm-start benchmark: cold-start foamRun vs ML warm-start (U,p only),
on real OpenFOAM solves, held-out cases. Identical mesh, numerics, schemes,
convergence criteria per arm -- the only difference is the initial 0/U,0/p.

For each held-out case (dim,cid):
  1. Deterministically rebuild + mesh the case (aero_case_generator 'dev' tier),
     asserting the fluid-cell count matches the stored prediction length.
  2. COLD arm : copy meshed base, run foamRun from uniform 0/ fields.
  3. WARM arm : copy meshed base, overwrite 0/U,0/p with ML prediction, run.
  Record outer-iteration count + ExecutionTime + wall for each arm.

reduction_iters = cold_iters / warm_iters ;  speedup_exec = cold_exec/warm_exec.

Usage (ssh:openfoam):
  source /opt/openfoam14/etc/bashrc
  python3 benchmark_aero.py benchmark_inputs.npz results_bench.json <case_root>
"""
import os, sys, json, shutil, subprocess, time, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aero_case_generator as acg

FOAM = 'source /opt/openfoam14/etc/bashrc'

def sh(cmd, cwd):
    return subprocess.run(['bash','-lc', f'{FOAM} >/dev/null 2>&1; {cmd}'],
                          cwd=cwd, capture_output=True, text=True)

def read_hdr(path):
    with open(path) as f: L = f.read().split('\n')
    for i, ln in enumerate(L):
        if ln.strip().startswith('internalField'): return L, i
    raise RuntimeError('no internalField in '+path)

def wvec(path, arr):
    L, i = read_hdr(path); N = len(arr)
    body = ['internalField   nonuniform List<vector> ', str(N), '('] + \
           [f'({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})' for v in arr] + [')', ';']
    j = i
    while ';' not in L[j]: j += 1
    open(path,'w').write('\n'.join(L[:i]+body+L[j+1:]))

def wsca(path, arr):
    L, i = read_hdr(path); N = len(arr)
    body = ['internalField   nonuniform List<scalar> ', str(N), '('] + \
           [f'{v:.6g}' for v in arr] + [')', ';']
    j = i
    while ';' not in L[j]: j += 1
    open(path,'w').write('\n'.join(L[:i]+body+L[j+1:]))

def count_internal(path):
    """number of internalField cell values (vector or scalar)."""
    L, i = read_hdr(path)
    for j in range(i, len(L)):
        if L[j].strip().isdigit():
            return int(L[j].strip())
    return None

def mesh_case(cd):
    r = sh('blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 && '
           'subsetMesh -cellSet fluid -noFields > log.subsetMesh 2>&1 && '
           "sed -i 's/oldInternalFaces/buildings/' constant/polyMesh/boundary && "
           'foamDictionary constant/polyMesh/boundary -entry entry0/buildings/type -set wall >/dev/null 2>&1 && '
           'checkMesh > log.checkMesh 2>&1', cd)
    return r.returncode == 0

def clean_times(cd):
    for d in os.listdir(cd):
        if d.isdigit() and d != '0':
            shutil.rmtree(os.path.join(cd, d), ignore_errors=True)

def run_foam(cd):
    t0 = time.time()
    r = sh('foamRun > log.foamRun 2>&1', cd)
    wall = time.time() - t0
    log = open(os.path.join(cd,'log.foamRun')).read()
    iters = log.count('Solving for p,')
    m = re.findall(r'ExecutionTime = ([\d.]+) s', log)
    return dict(iters=iters, exec_time=float(m[-1]) if m else float('nan'),
                wall=round(wall,3),
                converged='SIMPLE solution converged' in log, rc=r.returncode)

def main():
    inp   = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_inputs.npz'
    out   = sys.argv[2] if len(sys.argv) > 2 else 'results_bench.json'
    root  = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else '/tmp/aero_bench'
    os.makedirs(root, exist_ok=True)

    data = np.load(inp, allow_pickle=True)
    meta = json.loads(str(data['meta']))
    results = []
    for m in meta:
        dim, cid, n_fluid = m['dim'], int(m['cid']), int(m['n_fluid'])
        tag = f'{dim}_{cid}'
        rec = dict(dim=dim, cid=cid, n_fluid=n_fluid,
                   U_ml=m.get('U_ml'), p_ml=m.get('p_ml'))
        base = os.path.join(root, f'{tag}_base')
        # 1. mesh once (deterministic rebuild)
        if not os.path.exists(os.path.join(base,'constant','polyMesh','boundary')):
            if os.path.exists(base): shutil.rmtree(base)
            acg.build(base, 'dev', cid, dim)
            if not mesh_case(base):
                rec['error'] = 'mesh_failed'; results.append(rec)
                print(f'{tag}: MESH FAILED', flush=True); continue
        ncell = count_internal(os.path.join(base,'0','U'))
        rec['mesh_cells'] = ncell
        if ncell != n_fluid:
            rec['error'] = f'cell_mismatch {ncell}!={n_fluid}'
            results.append(rec); print(f'{tag}: {rec["error"]}', flush=True); continue

        # 2. COLD arm
        cold = os.path.join(root, f'{tag}_cold')
        if os.path.exists(cold): shutil.rmtree(cold)
        shutil.copytree(base, cold); clean_times(cold)
        rec['cold'] = run_foam(cold)

        # 3. WARM arm (ML U,p)
        warm = os.path.join(root, f'{tag}_warm')
        if os.path.exists(warm): shutil.rmtree(warm)
        shutil.copytree(base, warm); clean_times(warm)
        wvec(os.path.join(warm,'0','U'), data[f'{tag}_ml_U'])
        wsca(os.path.join(warm,'0','p'), data[f'{tag}_ml_p'])
        rec['warm'] = run_foam(warm)

        c, w = rec['cold'], rec['warm']
        rec['iter_reduction'] = round(c['iters']/w['iters'], 3) if w['iters'] else None
        rec['exec_speedup']   = round(c['exec_time']/w['exec_time'], 3) if w['exec_time'] else None
        results.append(rec)
        print(f"{tag}: cold it={c['iters']} exec={c['exec_time']:.2f}s | "
              f"warm it={w['iters']} exec={w['exec_time']:.2f}s | "
              f"iterx={rec['iter_reduction']} execx={rec['exec_speedup']} "
              f"(conv {c['converged']}/{w['converged']})", flush=True)
        json.dump(results, open(out,'w'), indent=2)

    # summary
    ok = [r for r in results if 'iter_reduction' in r and r['iter_reduction']]
    def med(key, rs): return float(np.median([r[key] for r in rs])) if rs else None
    r3 = [r for r in ok if r['dim']=='3d']; r2 = [r for r in ok if r['dim']=='2d']
    summary = dict(
        n=len(ok), n3d=len(r3), n2d=len(r2),
        iterx_med=med('iter_reduction', ok), execx_med=med('exec_speedup', ok),
        iterx_med_3d=med('iter_reduction', r3), execx_med_3d=med('exec_speedup', r3),
        iterx_med_2d=med('iter_reduction', r2), execx_med_2d=med('exec_speedup', r2))
    json.dump(dict(results=results, summary=summary), open(out,'w'), indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
