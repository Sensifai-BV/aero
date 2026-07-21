"""Generate a parametric 3D urban RANS case for OpenFOAM 14 (foamRun /
incompressibleFluid, steady kEpsilon).

Mesh strategy: a single uniform structured blockMesh block covers the domain;
axis-aligned building boxes are carved out with topoSet(boxToCell)+subsetMesh,
which exposes the building faces as a 'buildings' wall patch. Uniform cells give
an exact cell<->voxel mapping so a 3D CNN can read/write fields with no
interpolation.

Domain: x = streamwise (inlet->outlet), y = spanwise, z = vertical (ground->top).
Patches: inlet(x-), outlet(x+), ground(z-), top(z+), sides(y-,y+), buildings.
Inlet uses a power-law ABL profile via the built-in fixedProfile BC.

Usage: build_case(case_dir, params, grid) writes the full case tree.
"""
import os, math

FOAM_HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | SeedFOAM Hybrid 3D case generator
    \\\\  /    A nd           | Version:  14
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _w(path, cls, loc, obj, body):
    with open(path, "w") as f:
        f.write(FOAM_HDR.format(cls=cls, loc=loc, obj=obj))
        f.write("\n" + body + "\n// ************************************************************************* //\n")


def abl_profile(Uref, Zref, alpha, nz, Lz):
    """Power-law U(z)=Uref*(z/Zref)^alpha sampled at nz+1 heights for fixedProfile."""
    rows = []
    for i in range(nz + 1):
        z = Lz * i / nz
        u = Uref * (max(z, 1e-4) / Zref) ** alpha
        rows.append(f"    ({z:.5f} ({u:.5f} 0 0))")
    return "(\n" + "\n".join(rows) + "\n)"


def build_case(case_dir, params, grid):
    """params: dict(Lx,Ly,Lz,Uref,Zref,alpha,nu,buildings=[(x0,x1,y0,y1,zt),...],
                    endTime, I(turb intensity))
       grid:   dict(nx,ny,nz)  (uniform)
    """
    Lx, Ly, Lz = params["Lx"], params["Ly"], params["Lz"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    Uref = params["Uref"]; Zref = params.get("Zref", 0.5 * Lz)
    alpha = params["alpha"]; nu = params["nu"]
    I = params.get("I", 0.10); endTime = params.get("endTime", 2000)
    Cmu = 0.09

    for d in ["0", "system", "constant"]:
        os.makedirs(os.path.join(case_dir, d), exist_ok=True)

    # ---- blockMesh: single uniform block, 6 boundary patches ----
    v = [(0,0,0),(Lx,0,0),(Lx,Ly,0),(0,Ly,0),(0,0,Lz),(Lx,0,Lz),(Lx,Ly,Lz),(0,Ly,Lz)]
    verts = "\n".join(f"    ({x} {y} {z})" for x,y,z in v)
    bm = f"""scale 1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet   {{ type patch; faces ((0 4 7 3)); }}
    outlet  {{ type patch; faces ((1 2 6 5)); }}
    ground  {{ type wall;  faces ((0 3 2 1)); }}
    top     {{ type patch; faces ((4 5 6 7)); }}
    sides   {{ type symmetry; faces ((0 1 5 4) (3 7 6 2)); }}
);

mergePatchPairs ();
"""
    _w(os.path.join(case_dir, "system/blockMeshDict"), "dictionary", "system", "blockMeshDict", bm)

    # ---- topoSet: union of building boxes, invert to fluid ----
    acts = []
    for k,(x0,x1,y0,y1,zt) in enumerate(params["buildings"]):
        act = "new" if k == 0 else "add"
        acts.append(f"""    {{
        name    buildingCells; type cellSet; action {act};
        source  boxToCell; box ({x0} {y0} 0)({x1} {y1} {zt});
    }}""")
    acts.append("""    {
        name    fluid; type cellSet; action new;
        source  cellToCell; set buildingCells;
    }
    {
        name    fluid; type cellSet; action invert;
    }""")
    ts = "actions\n(\n" + "\n".join(acts) + "\n);\n"
    _w(os.path.join(case_dir, "system/topoSetDict"), "dictionary", "system", "topoSetDict", ts)

    # ---- controlDict ----
    cd = f"""solver          incompressibleFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {endTime};
deltaT          1;
writeControl    timeStep;
writeInterval   {endTime};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    _w(os.path.join(case_dir, "system/controlDict"), "dictionary", "system", "controlDict", cd)

    # ---- fvSchemes ----
    fs = """ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""
    _w(os.path.join(case_dir, "system/fvSchemes"), "dictionary", "system", "fvSchemes", fs)

    # ---- fvSolution ----
    rtol = params.get("resid_tol", 1e-3)
    fv = f"""solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-08;
        relTol          0.01;
        smoother        GaussSeidel;
    }}
    "(U|k|epsilon)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      no;
    residualControl
    {{
        p               {rtol};
        U               {rtol};
    }}
}}

relaxationFactors
{{
    fields  {{ p 0.3; }}
    equations
    {{
        U               0.7;
        "(k|epsilon)"   0.7;
    }}
}}
"""
    _w(os.path.join(case_dir, "system/fvSolution"), "dictionary", "system", "fvSolution", fv)

    # ---- constant ----
    _w(os.path.join(case_dir, "constant/momentumTransport"), "dictionary", "constant",
       "momentumTransport",
       "simulationType RAS;\n\nRAS\n{\n    model kEpsilon;\n    turbulence on;\n    viscosityModel Newtonian;\n}\n")
    _w(os.path.join(case_dir, "constant/physicalProperties"), "dictionary", "constant",
       "physicalProperties", f"viscosityModel constant;\n\nnu    {nu};\n")

    # ---- 0/ initial+BC fields ----
    kIn = 1.5 * (I * Uref) ** 2
    epsIn = Cmu ** 0.75 * kIn ** 1.5 / (0.07 * Lz)
    prof = abl_profile(Uref, Zref, alpha, nz, Lz)

    U = f"""dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            fixedProfile;
        profile         table {prof};
        direction       (0 0 1);
        origin          0;
    }}
    outlet    {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}
    ground    {{ type noSlip; }}
    buildings {{ type noSlip; }}
    top       {{ type slip; }}
    sides     {{ type symmetry; }}
}}
"""
    _w(os.path.join(case_dir, "0/U"), "volVectorField", "0", "U", U)

    p = """dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet     { type zeroGradient; }
    outlet    { type fixedValue; value uniform 0; }
    ground    { type zeroGradient; }
    buildings { type zeroGradient; }
    top       { type zeroGradient; }
    sides     { type symmetry; }
}
"""
    _w(os.path.join(case_dir, "0/p"), "volScalarField", "0", "p", p)

    kf = f"""dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {kIn:.6g};
boundaryField
{{
    inlet     {{ type fixedValue; value uniform {kIn:.6g}; }}
    outlet    {{ type inletOutlet; inletValue uniform {kIn:.6g}; value uniform {kIn:.6g}; }}
    ground    {{ type kqRWallFunction; value uniform {kIn:.6g}; }}
    buildings {{ type kqRWallFunction; value uniform {kIn:.6g}; }}
    top       {{ type slip; }}
    sides     {{ type symmetry; }}
}}
"""
    _w(os.path.join(case_dir, "0/k"), "volScalarField", "0", "k", kf)

    ef = f"""dimensions      [0 2 -3 0 0 0 0];
internalField   uniform {epsIn:.6g};
boundaryField
{{
    inlet     {{ type fixedValue; value uniform {epsIn:.6g}; }}
    outlet    {{ type inletOutlet; inletValue uniform {epsIn:.6g}; value uniform {epsIn:.6g}; }}
    ground    {{ type epsilonWallFunction; value uniform {epsIn:.6g}; }}
    buildings {{ type epsilonWallFunction; value uniform {epsIn:.6g}; }}
    top       {{ type slip; }}
    sides     {{ type symmetry; }}
}}
"""
    _w(os.path.join(case_dir, "0/epsilon"), "volScalarField", "0", "epsilon", ef)

    nutf = """dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet     { type calculated; value uniform 0; }
    outlet    { type calculated; value uniform 0; }
    ground    { type nutkWallFunction; value uniform 0; }
    buildings { type nutkWallFunction; value uniform 0; }
    top       { type calculated; value uniform 0; }
    sides     { type symmetry; }
}
"""
    _w(os.path.join(case_dir, "0/nut"), "volScalarField", "0", "nut", nutf)

    return dict(kIn=kIn, epsIn=epsIn, nx=nx, ny=ny, nz=nz)


if __name__ == "__main__":
    import sys, json
    
    case_dir = sys.argv[1] if len(sys.argv) > 1 else "case3d"
    
    # Load params from JSON string (argv[2]) or use defaults
    if len(sys.argv) > 2:
        try:
            params = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            params = dict(Lx=3.0, Ly=1.5, Lz=1.0, Uref=1.0, Zref=0.5, alpha=0.20, nu=1.5e-3,
                          buildings=[(1.0, 1.35, 0.55, 0.95, 0.5)], endTime=112, I=0.10, resid_tol=1e-3)
    else:
        params = dict(Lx=3.0, Ly=1.5, Lz=1.0, Uref=1.0, Zref=0.5, alpha=0.20, nu=1.5e-3,
                      buildings=[(1.0, 1.35, 0.55, 0.95, 0.5)], endTime=112, I=0.10, resid_tol=1e-3)
    
    grid = dict(nx=60, ny=30, nz=24)
    info = build_case(case_dir, params, grid)
    print(json.dumps(info))
