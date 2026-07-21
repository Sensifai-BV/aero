/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Aero Warm-Start Solver (aeroRun)
     \\/     M anipulation  | Version:  14 (with C++ ONNX ML Binding)
\*---------------------------------------------------------------------------*/

#include "volFields.H"
#include "surfaceFields.H"
#include "argList.H"
#include "solver.H"
#include "pimpleSingleRegionControl.H"
#include "setDeltaT.H"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <numeric>
#include <vector>

using namespace Foam;

// Helper to read AERO raw binary format if pre-generated raw tensor exists
static bool readAeroRaw(const std::string& path, std::vector<int64_t>& dims, std::vector<float>& data)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "AERO", 4) != 0) return false;

    int32_t ndim = 0;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(int32_t));
    if (ndim <= 0 || ndim > 8) return false;

    dims.resize(ndim);
    for (int i = 0; i < ndim; ++i)
    {
        int32_t d = 0;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        dims[i] = d;
    }

    int64_t n = std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
    data.resize(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
}

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "solver",
        "name",
        "Solver name (default: fluid / incompressibleFluid)"
    );

    argList::addOption
    (
        "onnxModel",
        "path",
        "Path to ONNX warm-start model"
    );

    #include "setRootCase.H"
    #include "createTime.H"

    // Read solverName from controlDict or command-line
    word solverName(runTime.controlDict().lookupOrDefault("solver", word("incompressibleFluid")));
    args.optionReadIfPresent("solver", solverName);

    if (solverName == word::null)
    {
        solverName = "incompressibleFluid";
    }

    Info<< nl << "=======================================================" << endl;
    Info<< "=== aeroRun: OpenFOAM 14 Deep Learning Warm-Start   ===" << endl;
    Info<< "=======================================================" << endl;
    Info<< "Loading OpenFOAM solver module: " << solverName << endl;

    // Load OpenFOAM solver library
    solver::load(solverName);

    // Create default single region mesh
    #include "createMesh.H"

    // Instantiate selected solver
    autoPtr<solver> solverPtr(solver::New(solverName, mesh));
    solver& solver = solverPtr();

    bool enableWarmStart = runTime.controlDict().lookupOrDefault("enableWarmStart", true);
    fileName modelPath = runTime.controlDict().lookupOrDefault<fileName>(
        "onnxModel",
        "/root/c/src/aero_warmstart.onnx"
    );

    word modelPathArg;
    if (args.optionReadIfPresent("onnxModel", modelPathArg))
    {
        modelPath = modelPathArg;
    }

    if (enableWarmStart)
    {
        Info<< "=== aeroRun: Checking ML Warm-Start Initialization ===" << endl;
        label nCells = mesh.nCells();

        if (mesh.foundObject<volVectorField>("U"))
        {
            volVectorField& U = const_cast<volVectorField&>(
                mesh.lookupObject<volVectorField>("U")
            );

            std::string inputRawPath = runTime.path() / "0" / "u_ml.raw";
            std::vector<int64_t> dims;
            std::vector<float> data;

            if (readAeroRaw(inputRawPath, dims, data))
            {
                Info<< "Seeding U field from AERO raw tensor: " << inputRawPath.c_str() << endl;
                int64_t nElem = data.size();
                if (nElem == nCells * 3)
                {
                    for (label celli = 0; celli < nCells; ++celli)
                    {
                        U.primitiveFieldRef()[celli].x() = data[celli * 3 + 0];
                        U.primitiveFieldRef()[celli].y() = data[celli * 3 + 1];
                        U.primitiveFieldRef()[celli].z() = data[celli * 3 + 2];
                    }
                    U.correctBoundaryConditions();
                    Info<< "SUCCESS: 0/U field seeded from AERO ML tensor (" << nCells << " cells)." << endl;
                }
            }
            else if (isFile(modelPath))
            {
                Info<< "Loading ONNX Warm-Start Model: " << modelPath << endl;
                try
                {
                    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "aeroRun");
                    Ort::SessionOptions opts;
                    opts.SetIntraOpNumThreads(4);
                    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                    Ort::Session session(env, std::string(modelPath).c_str(), opts);
                    Info<< "ONNX Model session initialized successfully in aeroRun." << endl;
                }
                catch (const std::exception& ex)
                {
                    WarningInFunction << "ONNX initialization note: " << ex.what() << endl;
                }
            }
            else
            {
                Info<< "No ML warm-start model/tensor found. Proceeding with standard initial fields." << endl;
            }
        }
    }

    pimpleSingleRegionControl pimple(solver.pimple);

    setDeltaT(runTime, solver);

    Info<< nl << "Starting aeroRun time loop\n" << endl;

    while (pimple.run(runTime))
    {
        solver.preSolve();
        adjustDeltaT(runTime, solver);
        runTime++;

        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        while (pimple.loop())
        {
            if (solver.pimple.flow())
            {
                solver.moveMesh();
                solver.motionCorrector();
            }

            if (solver.pimple.models())
            {
                solver.fvModels().correct();
            }

            solver.prePredictor();

            if (solver.pimple.predictTransport())
            {
                if (solver.pimple.flow())
                {
                    solver.momentumTransportPredictor();
                }

                if (solver.pimple.thermophysics())
                {
                    solver.thermophysicalTransportPredictor();
                }
            }

            if (solver.pimple.flow())
            {
                solver.momentumPredictor();
            }

            if (solver.pimple.thermophysics())
            {
                solver.thermophysicalPredictor();
            }

            if (solver.pimple.flow())
            {
                solver.pressureCorrector();
            }

            if (solver.pimple.correctTransport())
            {
                if (solver.pimple.flow())
                {
                    solver.momentumTransportCorrector();
                }

                if (solver.pimple.thermophysics())
                {
                    solver.thermophysicalTransportCorrector();
                }
            }
        }

        solver.postSolve();

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;
    return 0;
}
