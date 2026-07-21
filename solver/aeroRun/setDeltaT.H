#ifndef setDeltaT_H
#define setDeltaT_H

#include "solver.H"

namespace Foam
{

void setDeltaT(Time& runTime, const solver& solver);
void adjustDeltaT(Time& runTime, const solver& solver);

}

#endif
