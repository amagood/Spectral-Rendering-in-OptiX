/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#include "prd.h"
#include "random.h"
#include "color.cuh"

using namespace optix;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
//rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
//rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

//rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );

rtDeclareVariable(float3,       extinction, , );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

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

static __device__ __inline__ float fresnel( float cos_theta_i, float cos_theta_t, float eta )
{
    const float rs = ( cos_theta_i - cos_theta_t*eta ) / 
                     ( cos_theta_i + eta*cos_theta_t );
    const float rp = ( cos_theta_i*eta - cos_theta_t ) /
                     ( cos_theta_i*eta + cos_theta_t );

    return 0.5f * ( rs*rs + rp*rp );
}

static __device__ __inline__ float3 logf( float3 v )
{
    return make_float3( logf(v.x), logf(v.y), logf(v.z) );
}

// -----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_radiance()
{
    const float3 w_out = -ray.direction;
	//float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	//float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	//float3 normal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);
	float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	float3 center_to_hit = hitpoint - make_float3(0.0f, 2.0f, 0.0f);
	float normal_test = optix::dot(center_to_hit, normal);
	if (normal_test < 0)
	{
		normal = -normal; // test if normal is outward
	}
	float cos_theta_i = optix::dot(w_out, normal);
	//prd_radiance.depth--;
	if (prd_radiance.depth == 0)
	{
		prd_radiance.albedo = make_float3(1.0f);
		prd_radiance.normal = normal;
	}
	prd_radiance.scatter = false;
	prd_radiance.t = t_hit;

	float refraction_index = cauchyRefractionIndex(prd_radiance.lambda/1000.0f, B, C );

    float eta;
    float3 transmittance = make_float3( 1.0f );
    if( cos_theta_i > 0.0f ) {
        // Ray is entering 
        eta = refraction_index;  // Note: does not handle nested dielectrics
    } else {
        // Ray is exiting; apply Beer's Law.
        // This is derived in Shirley's Fundamentals of Graphics book.
        transmittance = expf( -extinction * t_hit );
		
        eta         = 1.0f / refraction_index;
        cos_theta_i = -cos_theta_i;
        normal      = -normal;
    }

    float3 w_t;
    const bool tir           = !refract( w_t, -w_out, normal, eta );

    const float cos_theta_t  = -dot( normal, w_t );
    const float R            = tir  ?
                               1.0f :
                               fresnel( cos_theta_i, cos_theta_t, eta );

    // Importance sample the Fresnel term
    const float z = rnd( prd_radiance.seed );
	prd_radiance.countEmitted = false;
	prd_radiance.done = false;
    if( z <= R ) {
        // Reflect
        const float3 w_in = reflect( -w_out, normal ); 
        //const float3 fhp = offset(hitpoint, normal);
		//prd_radiance.origin = fhp;
        prd_radiance.origin = hitpoint;
        prd_radiance.direction = w_in; 
		//prd_radiance.attenuation = prd_radiance.attenuation * reflection_color;
		prd_radiance.radiance *= reflection_color *transmittance;
    } else {
        // Refract
        const float3 w_in = w_t;
		//const float3 w_in = reflect(-w_out, normal);
        //const float3 bhp = offset(hitpoint, -normal);
		//prd_radiance.origin = bhp;
        prd_radiance.origin = hitpoint;
        prd_radiance.direction = w_in; 
		//prd_radiance.attenuation = prd_radiance.attenuation * refraction_color;
		prd_radiance.radiance *= refraction_color *transmittance;
    }

    // Note: we do not trace the ray for the next bounce here, we just set it up for
    // the ray-gen program using per-ray data. 

}


