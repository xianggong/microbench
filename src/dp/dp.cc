#include "dp.h"

#include <memory>

DynamicParallelism::DynamicParallelism()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
}

int main(int argc, char const *argv[])
{
	
	return 0;
}