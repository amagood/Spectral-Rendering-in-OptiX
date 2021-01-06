/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include <common.h>
#include "optixPathTracer.h"
#include "random.h"
#include "prd.h"
#include "color.cuh"

using namespace optix;

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      light_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  light_depth, , ); // light max depth

rtBuffer<Photon, 1>              photon_buffer;
rtBuffer<ParallelogramLight>     lights;
rtBuffer<float3, 1>				 lambda_buffer;

__device__ __host__ float3 getRGB(int lambda)
{
	float3 tmp = lambda_buffer[lambda];
	return XYZ2RGB(tmp);
}

RT_PROGRAM void LightPass()
{
	unsigned int seed = tea<16>(launch_index.x, frame_number); // todo (might need change)

	// init light position
	ParallelogramLight light = lights[int(rnd(seed)*lights.size())]; // choose a random light
	float z1 = rnd(seed);
	float z2 = rnd(seed);
	const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
	//float3 emission = make_float3(1.0f) / photon_buffer.size();
	//float3 emission = normalize(light.emission) / photon_buffer.size();
	float3 emission = light.emission;

	// set up light direciton
	z1 = rnd(seed);
	z2 = rnd(seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(-light.normal);
	onb.inverse_transform(p);

	// set up light prd
	light_prd prd;
	prd.origin = light_pos;
	prd.direction = p;
	prd.color = emission;
	//prd.intensity = length(light.emission);
	//prd.intensity = photon_buffer.size() * length(light.emission);
	prd.intensity = 1.0f;
	prd.seed = seed;
	prd.scatter = true;
	prd.caustic = false;
	prd.done = false;
	prd.depth = 0;
	prd.normal = -light.normal;

	prd.wavelength = int(rnd(prd.seed) * 400) + 380; // random wavelenght
	prd.color *= getRGB(prd.wavelength);

	Photon photon;
	//photon.direction = -light.normal; // might change
	photon.color = prd.color;
	photon.wavelength = prd.wavelength;

	// trace light
	for (; prd.depth < light_depth;) // light with max depth of light_depth
	{
		// update photon data
		photon.position = prd.origin;
		photon.normal = prd.normal;
		photon.color = prd.color;
		photon.energy = prd.intensity;
		photon.depth = prd.depth;
		photon.scatter = prd.scatter;
		photon.caustic = prd.caustic;
		photon.direction_outward = prd.direction;
		// save photon to buffer
		photon_buffer[launch_index.x * light_depth + prd.depth] = photon;

		photon.direction = prd.direction;
		prd.intensity *= 0.5;
		
		// cast ray
		Ray ray = make_Ray(prd.origin, prd.direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(light_object, ray, prd);
		
		if (prd.done) // miss
		{
			if (prd.scatter == true)
			{
				z1 = rnd(prd.seed);
				z2 = rnd(prd.seed);
				cosine_sample_hemisphere(z1, z2, p);
				optix::Onb onb(prd.normal);
				onb.inverse_transform(p);
				prd.direction = p;
				//rtPrintf("recast\n");
			}
			else break;
		}

		prd.depth++;
	}
}

rtDeclareVariable(light_prd, current_prd, rtPayload, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, diffuse_color, , );

RT_PROGRAM void diffuse()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	current_prd.origin = hitpoint;
	current_prd.normal = ffnormal;
	//current_prd.color = current_prd.color * diffuse_color;
	current_prd.color = color(current_prd.color, diffuse_color);

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;

	current_prd.done = false;
	current_prd.scatter = true;

	// update light intensity (energy) todo //
}

// glass material
//rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(float3, refraction_color, , );
rtDeclareVariable(float3, reflection_color, , );

rtDeclareVariable(float3, extinction, , );

rtDeclareVariable(float, B, , );
rtDeclareVariable(float, C, , );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 offset(const optix::float3& hit_point, const optix::float3& normal)
{
	using namespace optix;

	const float epsilon = 1.0e-4f;
	const float offset = 4096.0f * 2.0f;

	float3 offset_point = hit_point;
	if ((__float_as_int(hit_point.x) & 0x7fffffff) < __float_as_int(epsilon)) {
		offset_point.x += epsilon * normal.x;
	}
	else {
		offset_point.x = __int_as_float(__float_as_int(offset_point.x) + int(copysign(offset, hit_point.x) * normal.x));
	}

	if ((__float_as_int(hit_point.y) & 0x7fffffff) < __float_as_int(epsilon)) {
		offset_point.y += epsilon * normal.y;
	}
	else {
		offset_point.y = __int_as_float(__float_as_int(offset_point.y) + int(copysign(offset, hit_point.y) * normal.y));
	}

	if ((__float_as_int(hit_point.z) & 0x7fffffff) < __float_as_int(epsilon)) {
		offset_point.z += epsilon * normal.z;
	}
	else {
		offset_point.z = __int_as_float(__float_as_int(offset_point.z) + int(copysign(offset, hit_point.z) * normal.z));
	}

	return offset_point;
}

static __device__ __inline__ float fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
	const float rs = (cos_theta_i - cos_theta_t * eta) /
		(cos_theta_i + eta * cos_theta_t);
	const float rp = (cos_theta_i * eta - cos_theta_t) /
		(cos_theta_i * eta + cos_theta_t);

	return 0.5f * (rs * rs + rp * rp);
}

static __device__ __inline__ float3 logf(float3 v)
{
	return make_float3(logf(v.x), logf(v.y), logf(v.z));
}

// -----------------------------------------------------------------------------

RT_PROGRAM void glass()
{
	const float3 w_out = -ray.direction;
	current_prd.done = false;
	//float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	//float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	//float3 normal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	float cos_theta_i = optix::dot(w_out, normal);

	float refraction_index = cauchyRefractionIndex(current_prd.wavelength / 1000.0f, B, C);

	float eta;
	float3 transmittance = make_float3(1.0f);
	if (cos_theta_i > 0.0f) {
		// Ray is entering 
		eta = refraction_index;  // Note: does not handle nested dielectrics
	}
	else {
		// Ray is exiting; apply Beer's Law.
		// This is derived in Shirley's Fundamentals of Graphics book.
		transmittance = expf(-extinction * t_hit);

		eta = 1.0f / refraction_index;
		cos_theta_i = -cos_theta_i;
		normal = -normal;
	}

	float3 w_t;
	const bool tir = !refract(w_t, -w_out, normal, eta);

	const float cos_theta_t = -dot(normal, w_t);
	const float R = tir ?
		1.0f :
		fresnel(cos_theta_i, cos_theta_t, eta);

	// Importance sample the Fresnel term
	const float z = rnd(current_prd.seed);
	current_prd.done = false;
	current_prd.scatter = false;
	current_prd.caustic = true;
	if (z <= R) {
		// Reflect
		const float3 w_in = reflect(normalize(-w_out), normalize(normal));
		current_prd.origin = hitpoint;
		current_prd.direction = w_in;
		current_prd.color = current_prd.color * reflection_color;
	}
	else {
		// Refract
		const float3 w_in = w_t;
		current_prd.origin = hitpoint;
		current_prd.direction = w_in;
		current_prd.color = current_prd.color * refraction_color;
	}
}


RT_PROGRAM void metal()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;
	float3 reflect_dir = normalize(reflect(ray.direction, ffnormal));

	current_prd.origin = hitpoint;
	current_prd.normal = ffnormal;
	current_prd.direction = reflect_dir;
	current_prd.done = false;
	current_prd.scatter = false;
	current_prd.caustic = true;
}

//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void miss()
{
	// terminate light subpath (if miss no recasting for now)
	current_prd.done = true;
}