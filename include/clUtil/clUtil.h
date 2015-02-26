#ifndef CL_UTIL_H
#define CL_UTIL_H 

#include "clError.h"
#include "clFile.h"
#include "clRuntime.h"
#include "clProfiler.h"

namespace clHelper
{

#define ENABLE_PROFILE 1

#if ENABLE_PROFILE
#define clEnqueueNDRangeKernel clTimeNDRangeKernel
#endif

#ifndef clSVMFreeSafe
#define clSVMFreeSafe(ctx, ptr) if(ptr) clSVMFree(ctx, ptr)
#endif

} // namespace clHelper

#endif
