#include "setDeltaT.H"

void Foam::setDeltaT(Time& runTime, const solver& solver)
{
    if
    (
        runTime.timeIndex() == 0
     && runTime.controlDict().lookupOrDefault("adjustTimeStep", false)
     && solver.transient()
    )
    {
        const scalar deltaT =
            min(solver.maxDeltaT(), runTime.functionObjects().maxDeltaT());

        if (deltaT < rootVGreat)
        {
            runTime.setDeltaT(min(runTime.deltaTValue(), deltaT));
        }
    }
}

void Foam::adjustDeltaT(Time& runTime, const solver& solver)
{
    if
    (
        runTime.controlDict().lookupOrDefault("adjustTimeStep", false)
     && solver.transient()
    )
    {
        const scalar deltaT =
            min(solver.maxDeltaT(), runTime.functionObjects().maxDeltaT());

        if (deltaT < rootVGreat)
        {
            runTime.setDeltaT
            (
                min(solver::deltaTFactor*runTime.deltaTValue(), deltaT)
            );
            Info<< "deltaT = " <<  runTime.deltaTValue() << endl;
        }
    }
}
