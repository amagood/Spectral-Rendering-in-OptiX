#pragma once
#include <optixu/optixu_vector_types.h>

// Eye Pass Per Ray Data
struct eye_prd
{
	optix::float3 result;
	optix::float3 radiance;
	optix::float3 attenuation;
	optix::float3 origin;
	optix::float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
	int scatter;
	int wavelength;
	optix::float3 normal;
};

// Light Pass Per Ray Data
struct light_prd
{
	optix::float3 origin;
	optix::float3 direction;
	optix::float3 color;
	optix::float3 normal;
	optix::float3 direction_outward;
	unsigned int seed;
	int depth;
	int caustic;
	int done;
	int scatter; // check if surface can scatter
	int wavelength;
	int split; // if the light have split its wavelength
	float intensity;
};

// information for storing photon
struct Photon
{
	optix::float3 position;
	optix::float3 direction; // incoming direction
	optix::float3 color;
	optix::float3 normal;
	optix::float3 direction_outward;
	int depth;
	int scatter; // check if surface can scatter
	int caustic;
	int wavelength;
	int split;
	float energy;
};