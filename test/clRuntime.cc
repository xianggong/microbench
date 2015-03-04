#include <clUtil/clRuntime.h>

using namespace clHelper;

int main(int argc, char const *argv[])
{
	
		clRuntime *runtime = clRuntime::getInstance();
        cl_platform_id platform = runtime->getPlatformID();
        cl_device_id device = runtime->getDevice();
        cl_context context = runtime->getContext();
        cl_command_queue cmdQueue = runtime->getCmdQueue(0);
        
		return 0;
}