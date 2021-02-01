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

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      eye_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(eye_prd, current_prd, rtPayload, );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );
rtDeclareVariable(float,		 light_path_count, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<Photon, 1>				 photon_buffer;
rtBuffer<ParallelogramLight>     lights;
rtBuffer<float3, 1>				 lambda_buffer;

rtDeclareVariable(uint, photon_count, , );

__device__ __host__ float3 getRGB(int lambda)
{
	float3 tmp = lambda_buffer[lambda];
	return XYZ2RGB(tmp);
}

RT_PROGRAM void bidirectionalpathtrace_camera()
{
	size_t2 screen = output_buffer.size();

	float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;
	unsigned int samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
	float3 result = make_float3(0.0f);

	unsigned int path_count = 0; // how many connect path
	unsigned int specular_path = 0, caustic_path = 0, bidirectional_path = 0;

	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number);
	//float total_p = 0.0f;
	do
	{
		path_count = 0; // how many connect path
		float3 connect_result = make_float3(0.f);
		//
		// Sample pixel using jittering
		//
		float total_p = 0.0f;

		unsigned int x = samples_per_pixel % sqrt_num_samples;
		unsigned int y = samples_per_pixel / sqrt_num_samples;
		float2 jitter = make_float2(x - rnd(seed), y - rnd(seed));
		float2 d = pixel + jitter * jitter_scale;
		float3 ray_origin = eye;
		float3 ray_direction = normalize(d.x * U + d.y * V + W);

		// Initialze per-ray data
		eye_prd prd;
		prd.result = make_float3(0.f);
		prd.radiance = make_float3(1.f);
		prd.countEmitted = true;
		prd.done = false;
		prd.seed = seed;
		prd.depth = 0;
		prd.scatter = true;
		prd.split = false;
		prd.wavelength = int(rnd(prd.seed) * 400) + 380; // random wavelenght
		//prd.wavelength = frame_number % 400 + 380; // frame wavelenght
		//prd.wavelength = 380;
		//prd.attenuation = getRGB(prd.wavelength); // ray started as a single wavelength
		//prd.attenuation = make_float3(1.0f); // ray started as full wavelength
		prd.attenuation = make_float3(0.300985f, 0.274355f, 0.216741f); // ray started as full wavelength
		prd.p = 1.0f; 

		// check for eyesubpath = 0 or 1
		for (int i = 0; i < photon_count; i++)
		{
			if (photon_buffer[i].energy < 1e-6) continue; // skip miss path
			float3 photon_eye_dir = photon_buffer[i].position - ray_origin;
			if (dot(normalize(photon_eye_dir), normalize(ray_direction)) >= 0.9999999f)
			{
				PerRayData_pathtrace_shadow caustic_prd;
				caustic_prd.inShadow = false;
				Ray causticRay = make_Ray(ray_origin, ray_direction, SHADOW_RAY_TYPE, scene_epsilon, length(photon_eye_dir) - scene_epsilon);
				rtTrace(eye_object, causticRay, caustic_prd);
				if (!caustic_prd.inShadow) // if connect
				{
					if (photon_buffer[i].scatter == true) // eye subpath = 1
					{
						total_p += photon_buffer[i].p * prd.p;
						prd.result += photon_buffer[i].color * (photon_buffer[i].energy) *prd.attenuation;// *make_float3(100.0f, 0.f, 0.f);
						caustic_path = 1;
						//prd.result += make_float3(100.0f, 0.f, 0.f);
						//path_count ++;
					}
					//prd.result += photon_buffer[i].color * (photon_buffer[i].energy) * prd.attenuation * 10.f;// *make_float3(100.0f, 0.f, 0.f);
					if(dot(normalize(photon_buffer[i].direction_outward), -normalize(ray_direction)) >= 0.9f) // eye subpath = 0
					{
						total_p += photon_buffer[i].p;
						prd.result += photon_buffer[i].color * (photon_buffer[i].energy);
						specular_path = 1;
						//path_count++;
					}
				}
			}
		}

		// Each iteration is a segment of the ray path.  The closest hit will
		// return new segments to be traced here.
		for (;;)
		{
			Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(eye_object, ray, prd);


			if (prd.done)
			{
				// We have hit the background or a luminaire
				total_p += prd.p;
				prd.result += prd.radiance * prd.attenuation;// / (prd.p / total_p);
				//path_count++;
				break;
			}

			if (prd.scatter == true)
			{
				for (int i = 0; i < photon_count; i++)
				{
					if (photon_buffer[i].energy < 1e-6) continue; // skip miss path
					// check connection with two subpath
					if (photon_buffer[i].scatter == true && ((!prd.split || !photon_buffer[i].split) || photon_buffer[i].wavelength == prd.wavelength))
					{
						float3 connect_direction = photon_buffer[i].position - prd.origin;
						float C = length(connect_direction);
						float eye_cos = dot(normalize(connect_direction), prd.normal);
						float light_cos = dot(-normalize(connect_direction), photon_buffer[i].normal);
						if (eye_cos > 0 && light_cos > 0)
						{
							PerRayData_pathtrace_shadow connectRay_prd;
							connectRay_prd.inShadow = false;
							Ray connectRay = make_Ray(prd.origin, normalize(connect_direction), SHADOW_RAY_TYPE, scene_epsilon, C - scene_epsilon);
							rtTrace(eye_object, connectRay, connectRay_prd);
							if (!connectRay_prd.inShadow) // if connect
							{
								// connecting light and eye subpath
								total_p += photon_buffer[i].p * prd.p;
								connect_result += (prd.attenuation * photon_buffer[i].color) * (photon_buffer[i].energy) * eye_cos * light_cos; // still using more like normal path tracing way to calculate color...
								//prd.result += (prd.attenuation * photon_buffer[i].color) * (photon_buffer[i].energy) * eye_cos * light_cos * (!photon_buffer[i].split? prd.p * photon_buffer[i].p:1.0f);
								//prd.result += color(prd.attenuation, photon_buffer[i].color) * (photon_buffer[i].energy) * eye_cos * light_cos; // connect ray's color might be wrong with the way of spectral.
								path_count++;
								bidirectional_path = 1;
							}
						}
					}
				}
			}
			

			// Russian roulette termination 
			if (prd.depth >= rr_begin_depth)
			{
				float pcont = fmaxf(prd.attenuation);
				if (rnd(prd.seed) >= pcont)
				{
					//path_count++; // failed path
					break;
				}
					
				prd.attenuation /= pcont;
			}

			prd.depth++;

			// Update ray data for the next path segment
			ray_origin = prd.origin;
			ray_direction = prd.direction;
		}

		//rtPrintf("%d\n", total_path_asdf);
		connect_result /= (path_count == 0?1.f: path_count);
		prd.result += connect_result;
		result += prd.result/(1 + specular_path + caustic_path + bidirectional_path);
		//result += prd.result / (path_count);
		seed = prd.seed;
	} while (--samples_per_pixel);

	//
	// Update the output buffer
	//
	//float3 pixel_color = result / (sqrt_num_samples * sqrt_num_samples + connect_path);
	float3 pixel_color = result / (sqrt_num_samples * sqrt_num_samples);
	//float3 pixel_color = result / path_count;

	//rtPrintf("%f %f %f\n", pixel_color.x, pixel_color.y, pixel_color.z);

	if (frame_number > 1)
	{
		float a = 1.0f / (float)frame_number;
		float b = (float)frame_number;
		float3 old_color = make_float3(output_buffer[launch_index]);
		//output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a), 1.0f);
		output_buffer[launch_index] = make_float4((old_color + pixel_color / (b - 1)) * ((b - 1) / b), 1.0f);// *total_p; // test
		//output_buffer[launch_index] += make_float4(pixel_color,1.0f);// *total_p; // test
	}
	else
	{
		output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
	}
}

//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void diffuseEmitter()
{
	//current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
	current_prd.attenuation *= emission_color;
	current_prd.done = true;
	current_prd.scatter = true;
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	current_prd.origin = hitpoint;
	//current_prd.t = t_hit;
	//current_prd.scatter = true;
	//current_prd.attenuation = emission_color;
}

//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float,	  Kd, , );
rtDeclareVariable(float3,     diffuse_color, , );
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3,	  bg_color, , );

RT_PROGRAM void diffuse()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;
	current_prd.done = false;
	current_prd.scatter = true;
	current_prd.normal = ffnormal;

	float distance = length(hitpoint - ray.origin);
	current_prd.p *= abs(dot(normalize(ffnormal), normalize(-ray.direction))) / (distance * distance) * M_1_PIf * 0.5f;

	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	//current_prd.attenuation *= diffuse_color * Kd;
	current_prd.attenuation = color(current_prd.attenuation * Kd, diffuse_color); // Kd
	//current_prd.attenuation = getRGB(current_prd.wavelength);
	current_prd.countEmitted = false;
}

RT_PROGRAM void diffuse_direct_light()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

	float3 hitpoint = ray.origin + t_hit * ray.direction;

	//
	// Generate a reflection ray.  This will be traced back in ray-gen.
	//
	current_prd.origin = hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p;
	cosine_sample_hemisphere(z1, z2, p);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);
	current_prd.direction = p;
	current_prd.done = true;

	//diffuse_color = make_float3(255, 255, 255);

	// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
	// with cosine density.
	current_prd.attenuation = current_prd.attenuation * diffuse_color;
	current_prd.countEmitted = false;

	//
	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	//rtPrintf("%d",num_lights);
	float ambient;
	float ambientStrength = 0.05f;
	ambient = ambientStrength;
	float specularStrength = 0.5f;
	float shininess = 32.0f;
	float3 viewDir = normalize(-ray.direction);

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
		float3 emission = normalize(light.emission);
		float3 reflectDir = normalize(reflect(normalize(hitpoint - light_pos), ffnormal));

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		//const float3 L = normalize(make_float3(0,-1,0));
		const float  nDl = dot(ffnormal, L);
		float tmp = dot(light.normal, L);
		const float  LnDl = tmp > 0 ? tmp : -tmp;

		// cast shadow ray
		if (nDl > 0.0f && LnDl > 0.0f)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
			Ray shadow_ray = make_Ray(hitpoint, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(eye_object, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{
				float diffuse = max(dot(ffnormal, L), 0.0f);
				float specular = pow(float(max(dot(viewDir, reflectDir), 0.0f)), shininess);
				result += (specularStrength * specular + diffuse + ambient) * emission;
			}
			else result += ambient * emission;
		}
		else result += ambient * emission;
	}
	current_prd.radiance = result;

}

rtDeclareVariable(float3, metal_color, , );

RT_PROGRAM void metal()
{
	current_prd.done = false;
	current_prd.countEmitted = true;
	current_prd.scatter = false;
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
	
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	current_prd.origin = hitpoint;
	float3 reflect_dir = normalize(reflect(ray.direction, ffnormal));
	current_prd.direction = reflect_dir;
	current_prd.attenuation *= metal_color;
}

//rtDeclareVariable(float,        refraction_index, , );
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
		current_prd.countEmitted = true;
		current_prd.done = false;
		current_prd.scatter = false;
		current_prd.origin = hitpoint;
		current_prd.normal = normal;

		if (z <= R) {
			// Reflect
			const float3 w_in = reflect(normalize(-w_out), normalize(normal));
			current_prd.direction = w_in;
			current_prd.attenuation *= reflection_color;
			current_prd.radiance *= reflection_color * transmittance;
		}
		else {
			// Refract
			// split the light into single wavelength
			if (current_prd.split == false)
			{
				current_prd.attenuation *= getRGB(current_prd.wavelength) * make_float3(1.0f/0.300985f, 1.0f / 0.274355f, 1.0f / 0.216741f);
				current_prd.split = true;
				current_prd.p /= 400;
			}
			//rtPrintf("%d\n",current_prd.wavelength);
			//rtPrintf("%f %f %f\n", current_prd.attenuation.x, current_prd.attenuation.y, current_prd.attenuation.z);
			const float3 w_in = w_t;
			current_prd.direction = w_in;
			current_prd.attenuation *= refraction_color;
			current_prd.radiance *= refraction_color * transmittance;
		}
}
//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
}


//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_float4(bad_color, 1.0f);
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void miss()
{
    //current_prd.radiance = bg_color;
	current_prd.attenuation = make_float3(0.0f);
    current_prd.done = true;
}



