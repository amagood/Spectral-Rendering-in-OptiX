//
// Created by amagood on 2020/10/07.
//

#define CLAMP0255_XY(x, y) (((x) < 0) ? 0 : ((x) > (y) ? (y) : (x)))

__device__ __host__ float cauchyRefractionIndex(float lambda, float B, float C)
{
    return  B + C / pow(lambda, 2.0f);
}

__host__ __device__ float3 color(float3 light, float3 local)
{
	return make_float3((light.x > local.x) ? local.x : light.x, (light.y > local.y) ? local.y : light.y, (light.z > local.z) ? local.z : light.z);
}

__host__ __device__ float3 new_color(float3 light, float3 local)
{
	float scale;
	if (light.x / local.x < light.y / local.y) scale = light.x / local.x;
	else scale = light.y / local.y;
	if (scale > light.z / local.z) scale = light.z / local.z;
	return light*scale;
}



__device__ __host__ void RGB2XYZ(float R, float G, float B, float *X, float *Y, float *Z)
{
    *X = 0.412453f * R + 0.357580f * G + 0.180423f * B;
    *Y = 0.212671f * R + 0.715160f * G + 0.072169f * B;
    *Z = 0.019334f * R + 0.119193f * G + 0.950227f * B;
}


__device__ __host__ float3 XYZ2RGB(float3 XYZ)
{
    float RR, GG, BB;
    RR =  3.240479f * XYZ.x - 1.537150f * XYZ.y - 0.498535f * XYZ.z;
    GG = -0.969256f * XYZ.x + 1.875992f * XYZ.y + 0.041556f * XYZ.z;
    BB =  0.055648f * XYZ.x - 0.204043f * XYZ.y + 1.057311f * XYZ.z;

	return make_float3(CLAMP0255_XY(RR, 1.0f), CLAMP0255_XY(GG, 1.0f),  CLAMP0255_XY(BB, 1.0f));
}

