#ifndef CL_UTIL_H
#define CL_UTIL_H 

#include "clError.h"
#include "clFile.h"
#include "clRuntime.h"

namespace clHelper
{

#ifndef clSVMFreeSafe
#define clSVMFreeSafe(ctx, ptr) if(ptr) clSVMFree(ctx, ptr)
#endif

} // namespace clHelper

#endif
