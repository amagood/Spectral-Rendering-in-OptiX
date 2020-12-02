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


#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <common.h>
#include "optixDenoiser.h"
#include "random.h"
#include "prd.h"
#include "color.cuh"

#define LIGHT_RAY_TYPE 2
#define CONNECTIVE_RAY_TYPE 3


using namespace optix;

struct PerRayData_pathtrace_shadow
{
    bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_radiance, current_prd, rtPayload, );



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(Matrix3x3,     normal_matrix, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              input_albedo_buffer;
rtBuffer<float4, 2>              input_normal_buffer;
rtBuffer<ParallelogramLight>     lights;
rtBuffer<float3, 1> lambda_buffer;
rtBuffer<PerRayData_radiance, 1> light_cache;


__device__ __host__ float3 getRGB(int lambda)
{
    float3 tmp = lambda_buffer[lambda];
    return XYZ2RGB(tmp);
}

RT_PROGRAM void pathtrace_camera()
{
    
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    float num_samples = sqrt_num_samples * sqrt_num_samples;
    unsigned int samples_per_pixel = num_samples;
    float3 result = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);
    float3 normal = make_float3(0.0f);

	int sample;

    unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
    do 
    {
        //
        // Sample pixel using jittering
        //
        unsigned int x = samples_per_pixel%sqrt_num_samples;
        unsigned int y = samples_per_pixel/sqrt_num_samples;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        // Initialze per-ray data
		PerRayData_radiance prd;
		prd.radiance = make_float3(0.f); // same as emitt
        prd.result = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;
		//prd.lambda = int(rnd(prd.seed) * 400) + 380; // random wavelenght
		prd.lambda = frame_number%400 + 380; // random wavelenght
		//prd.lambda = frame_number%10*40 + int(rnd(prd.seed) * 40) + 380;
		//prd.lambda = 390;
		prd.attenuation = getRGB(prd.lambda);
		prd.scatter = true;
		prd.light = false;

        // Initialze per-ray data (for light ray)
        PerRayData_radiance lightRay_prd;
        lightRay_prd.result = make_float3(0.f);
        lightRay_prd.attenuation = make_float3(1.f);
		lightRay_prd.radiance = make_float3(15.f); // same as emitt
        lightRay_prd.countEmitted = true;
        lightRay_prd.done = false;
        lightRay_prd.seed = seed;
        lightRay_prd.depth = 0;
		lightRay_prd.lambda = prd.lambda; // follow prd
		lightRay_prd.scatter = true;
		lightRay_prd.light = true;
        //prd.lambda = 390;
		lightRay_prd.attenuation = prd.attenuation;

		//init light ray data
		ParallelogramLight light = lights[0]; // todo random will be preferable (only one light for current scene)
		const float z1 = rnd(lightRay_prd.seed);
		const float z2 = rnd(lightRay_prd.seed);
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
		lightRay_prd.origin = light_pos;
		lightRay_prd.normal = -light.normal;
		const float A = length(cross(light.v1, light.v2));

		//uniform sample on sphere
		float theta = 2.0f * M_PIf * rnd(lightRay_prd.seed);
		float phi = acos(1.0f - 2.0f * rnd(lightRay_prd.seed));
		float3 direction = make_float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
		lightRay_prd.direction = direction;

		light_cache[0] = lightRay_prd;

		int light_depth = 20;

		//lightRay_prd.origin = light_pos;
		//lightRay_prd.direction = direction;

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
		sample = 0;
        for(;;)
        {
			//if (lightRay_prd.depth > 2)
			//{
			Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
			rtTrace(top_object, ray, prd);
			//}
			
			
			if (lightRay_prd.depth <= light_depth) // cast light ray
			{
				Ray lightRay = make_Ray(lightRay_prd.origin, lightRay_prd.direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
				//Ray lightRay = make_Ray(light_pos, direction, LIGHT_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
				rtTrace(top_object, lightRay, lightRay_prd);

				light_cache[lightRay_prd.depth+1] = lightRay_prd;
			}

            if(prd.done)
            {
                // We have hit the background or a luminaire
                prd.result += prd.radiance * prd.attenuation;
                break;
            }

			// connect path
			for (int i = 0; i <= lightRay_prd.depth + 1; i++)
			{
				if (prd.scatter && light_cache[i].scatter)
				{
					//PerRayData_radiance connectRay_prd;
					PerRayData_pathtrace_shadow connectRay_prd;
					//connectRay_prd.done = false;
					connectRay_prd.inShadow = false;
					float3 connect_direction = normalize(prd.origin - light_cache[i].origin);
					float C = length(prd.origin - light_cache[i].origin);
					if ((dot(light_cache[i].normal, connect_direction) > 0.0f && dot(prd.normal, -connect_direction) > 0.0f) || C < 0.05)
					{
						Ray connectRay = make_Ray(light_cache[i].origin, connect_direction, SHADOW_RAY_TYPE, scene_epsilon, C - scene_epsilon);
						//Ray connectRay = make_Ray(light_cache[i].origin, connect_direction, CONNECTIVE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
						rtTrace(top_object, connectRay, connectRay_prd);
						//if (C >= length(light_cache[i].origin - connectRay_prd.origin))
						if (!connectRay_prd.inShadow || C < 0.05)
						{
							//if (C < 0.05) rtPrintf("shit");
							//prd.radiance += light_cache[i].radiance * dot(light_cache[i].normal, connect_direction) * dot(prd.normal, -connect_direction); //todo need add weight
							//prd.radiance += make_float3(0.1f);
							//prd.radiance = make_float3(15.f);
							prd.result += ((prd.radiance + light_cache[i].radiance) * dot(light_cache[i].normal, connect_direction) * dot(prd.normal, -connect_direction)) * color(light_cache[i].attenuation, prd.attenuation) / (M_PIf * C * C);
							//prd.attenuation *= light_cache[i].attenuation;
							//prd.attenuation = new_color(light_cache[i].attenuation, prd.attenuation);
							//prd.attenuation = make_float3(1.0f, 0.0f, 0.0f);
							sample++;
						}
					}
				}
			}

            // Russian roulette termination 
            if(prd.depth >= rr_begin_depth)
            {
                float pcont = fmaxf(prd.attenuation);
                if(rnd(prd.seed) >= pcont)
                    break;
                prd.attenuation /= pcont;
            }

            prd.depth++;
			if (lightRay_prd.depth <= light_depth) lightRay_prd.depth++;
            prd.result += prd.radiance * prd.attenuation;
			//prd.radiance = make_float3(0.0f);

            // Update ray data for the next path segment
            ray_origin = prd.origin;
            ray_direction = prd.direction;

        }

        result += prd.result;
        albedo += prd.albedo;
        float3 normal_eyespace = (length(prd.normal) > 0.f) ? normalize(normal_matrix * prd.normal) : make_float3(0.f, 0.f, 1.f);
        normal += normal_eyespace;
        seed = prd.seed;
        
    }while (--samples_per_pixel);
    
    //
    // Update the output buffer
    //
    unsigned int spp = num_samples + sample;
    float3 pixel_color = result/(spp);
    float3 pixel_albedo = albedo/(spp);
    float3 pixel_normal = normal/(spp);

    if (frame_number > 1)
    {
        float a = 1.0f / (float)frame_number;
        float3 old_color = make_float3(output_buffer[launch_index]);
        float3 old_albedo = make_float3(input_albedo_buffer[launch_index]);
        float3 old_normal = make_float3(input_normal_buffer[launch_index]);
        output_buffer[launch_index] = make_float4(lerp( old_color, pixel_color, a), 1.0f);
        input_albedo_buffer[launch_index] = make_float4(lerp( old_albedo, pixel_albedo, a), 1.0f);

        // this is not strictly a correct accumulation of normals, but it will do for this sample
        float3 accum_normal = lerp( old_normal, pixel_normal, a);
        input_normal_buffer[launch_index] = make_float4( (length(accum_normal) > 0.f) ? normalize(accum_normal) : pixel_normal, 1.0f);
    }
    else
    {
        output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
        input_albedo_buffer[launch_index] = make_float4(pixel_albedo, 1.0f);
        input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
    }
}


//-----------------------------------------------------------------------------
//
//  Emissive surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,        emission_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

RT_PROGRAM void diffuseEmitter()
{
    //current_prd.radiance = current_prd.countEmitted ? emission_color : make_float3(0.f);
	current_prd.radiance = emission_color;
    //current_prd.done = true;
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	current_prd.origin = hitpoint;
	current_prd.t = t_hit;
	current_prd.scatter = true;
	//current_prd.attenuation = make_float3(1.0f,0.f,0.f);

    // TODO: Find out what the albedo buffer should really have. For now just set to white for 
    // light sources.
    if (current_prd.depth == 0)
    {
        current_prd.albedo = make_float3(1.0f, 1.0f, 1.0f);

        float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
        float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
        float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

        current_prd.normal = ffnormal;
    }
	//current_prd.radiance = make_float3(1, 1, 1);
}


//-----------------------------------------------------------------------------
//
//  Lambertian surface closest-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3,     diffuse_color, , );


static __device__ float checkerboard3(float3 loc)
{
  loc += make_float3(0.001f); // small epsilon so planes don't coincide with scene geometry
  float checkerboard_width = 40.f;
  int3 c;
  
  c.x = abs((int)floor((loc.x / checkerboard_width)));
  c.y = abs((int)floor((loc.y / checkerboard_width)));
  c.z = abs((int)floor((loc.z / checkerboard_width)));
  
  if ((c.x % 2) ^ (c.y % 2) ^ (c.z % 2)) return 1.0f;
  return 0.0f;  
}


RT_PROGRAM void diffuse()
{
    float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    float3 hitpoint = ray.origin + t_hit * ray.direction;

    // texture is modulated by a procedural checkerboard for more detail
    //float3 modulated_diffuse_color = diffuse_color *  (0.2f + 0.8f * checkerboard3(hitpoint));
    float3 modulated_diffuse_color = diffuse_color;

    // The albedo buffer should contain an approximation of the overall surface albedo (i.e. a single
    // color value approximating the ratio of irradiance reflected to the irradiance received over the
    // hemisphere). This can be approximated for very simple materials by using the diffuse color of
    // the first hit.
    if (current_prd.depth == 0)
    {
        current_prd.albedo = modulated_diffuse_color;
        current_prd.normal = ffnormal;
    }
	current_prd.scatter = true;
	current_prd.t = t_hit;
    //
    // Generate a reflection ray.  This will be traced back in ray-gen.
    //
    current_prd.origin = hitpoint;

    float z1=rnd(current_prd.seed);
    float z2=rnd(current_prd.seed);
    float3 p;
    cosine_sample_hemisphere(z1, z2, p);
    optix::Onb onb( ffnormal );
    onb.inverse_transform( p );
    current_prd.direction = p;

    // NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
    // with cosine density.
    //current_prd.attenuation = current_prd.attenuation * modulated_diffuse_color;
	current_prd.attenuation = color(current_prd.attenuation, modulated_diffuse_color);
	//current_prd.attenuation = new_color(current_prd.attenuation, modulated_diffuse_color);
    current_prd.countEmitted = false;

    //
    // Next event estimation (compute direct lighting).
    //
    unsigned int num_lights = lights.size();
    float3 result = make_float3(0.0f);

    for(int i = 0; i < num_lights; ++i)
    {
        // Choose random point on light
        ParallelogramLight light = lights[i];
        const float z1 = rnd(current_prd.seed);
        const float z2 = rnd(current_prd.seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float  Ldist = length(light_pos - hitpoint);
        const float3 L     = normalize(light_pos - hitpoint);
        const float  nDl   = dot( ffnormal, L );
        const float  LnDl  = dot( light.normal, L );

        // cast shadow ray
        if ( nDl > 0.0f && LnDl > 0.0f )
        {
            PerRayData_pathtrace_shadow shadow_prd;
            shadow_prd.inShadow = false;
            // Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
            Ray shadow_ray = make_Ray( hitpoint, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon );
            rtTrace(top_object, shadow_ray, shadow_prd);

            if(!shadow_prd.inShadow)
            {
                const float A = length(cross(light.v1, light.v2));
                // convert area based pdf to solid angle
                const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
                
				//result += light.emission * weight;
				if (!current_prd.light) result += light.emission * weight;
				else result = current_prd.radiance * weight;
            }
        }
    }

    current_prd.radiance = result;
    
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
    rtPrintf("");
    //output_buffer[launch_index] = make_float4(bad_color, 1.0f);
    
}


//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, bg_color, , );

RT_PROGRAM void miss()
{
    //current_prd.radiance = bg_color;
    current_prd.done = true;
	float theta = 2.0f * M_PIf * rnd(current_prd.seed);
	float phi = acos(1.0f - 2.0f * rnd(current_prd.seed));
	current_prd.direction = make_float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)); // assign new direction
	current_prd.scatter = false;
	current_prd.t = 2;
	if (!current_prd.light)
	{
		current_prd.radiance = bg_color;
	}

    // TODO: Find out what the albedo buffer should really have. For now just set to black for misses.
    if (current_prd.depth == 0)
    {
      current_prd.albedo = make_float3(0.f, 0.f, 0.f);
      //current_prd.normal = make_float3(0, 0, 0);
    }
	//current_prd.depth--;
}


