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

 //-----------------------------------------------------------------------------
 //
 // optixPathTracer: simple interactive path tracer
 //
 //-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>
#include <OptiXMesh.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include "prd.h"

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width = 512;
uint32_t       height = 512;
bool           use_pbo = true;

int			   output_mode = 1; // identify which mode is active (for changing botton text)
int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 2;
int			   light_depth = 10;
int			   inital_photon_count = 100;
int			   photon_count = inital_photon_count;
int			   photon_thread = photon_count / light_depth;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

bool           use_tri_api = true;
bool           ignore_mats = false;
optix::Aabb    aabb;

// Buffer
Buffer photon_buffer;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}


void createContext()
{
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(2);
	context->setStackSize(18000);
	context->setMaxTraceDepth(2);

	context->setPrintEnabled(true);
	context->setPrintBufferSize(4096);

	context["scene_epsilon"]->setFloat(1.e-3f);
	context["rr_begin_depth"]->setUint(rr_begin_depth);
	context["light_depth"]->setUint(light_depth);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["output_buffer"]->set(buffer);

	//Setup for Eye pass
	const char* eye_ptx = sutil::getPtxString(SAMPLE_NAME, "EyePass.cu");
	context->setRayGenerationProgram(0, context->createProgramFromPTXString(eye_ptx, "bidirectionalpathtrace_camera"));
	context->setExceptionProgram(0, context->createProgramFromPTXString(eye_ptx, "exception"));
	context->setMissProgram(0, context->createProgramFromPTXString(eye_ptx, "miss"));

	//Setup for Light pass
	const char* light_ptx = sutil::getPtxString(SAMPLE_NAME, "LightPass.cu");
	context->setRayGenerationProgram(1, context->createProgramFromPTXString(light_ptx, "LightPass"));

	photon_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	photon_buffer->setFormat(RT_FORMAT_USER);
	photon_buffer->setElementSize(sizeof(Photon));
	photon_buffer->setSize(photon_count);
	context["photon_buffer"]->set(photon_buffer);
	memset(photon_buffer->map(), 0, sizeof(Photon) * photon_count);
	photon_buffer->unmap();
	context["photon_count"]->setUint(photon_count);

	context["sqrt_num_samples"]->setUint(sqrt_num_samples);
	context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
	context["bg_color"]->setFloat(make_float3(0.0f));
	context["light_path_count"]->setFloat(photon_thread);
}

void loadMesh(const std::string& filename)
{
	OptiXMesh mesh;
	mesh.context = context;
	mesh.use_tri_api = use_tri_api;
	mesh.ignore_mats = ignore_mats;
	loadMesh(filename, mesh);

	aabb.set(mesh.bbox_min, mesh.bbox_max);

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["eye_object"]->set(geometry_group);
	context["eye_shadower"]->set(geometry_group);
	context["light_object"]->set(geometry_group);
}

void lambdaInitiallBuffer()
{
	Buffer lambda_buffer = context->createBuffer(RT_BUFFER_INPUT);
	lambda_buffer->setFormat(RT_FORMAT_FLOAT3);
	lambda_buffer->setSize(781u);
	context["lambda_buffer"]->setBuffer(lambda_buffer);

	float3 lambdas[781];
	for (auto& i : lambdas)
		i = make_float3(0.0f, 0.0f, 0.0f);

	lambdas[380] = make_float3(0.0014, 0.0000, 0.0065);
	lambdas[381] = make_float3(0.0015, 0.0000, 0.0070);
	lambdas[382] = make_float3(0.0016, 0.0000, 0.0077);
	lambdas[383] = make_float3(0.0018, 0.0001, 0.0085);
	lambdas[384] = make_float3(0.0020, 0.0001, 0.0094);
	lambdas[385] = make_float3(0.0022, 0.0001, 0.0105);
	lambdas[386] = make_float3(0.0025, 0.0001, 0.0120);
	lambdas[387] = make_float3(0.0029, 0.0001, 0.0136);
	lambdas[388] = make_float3(0.0033, 0.0001, 0.0155);
	lambdas[389] = make_float3(0.0037, 0.0001, 0.0177);
	lambdas[390] = make_float3(0.0042, 0.0001, 0.0201);
	lambdas[391] = make_float3(0.0048, 0.0001, 0.0225);
	lambdas[392] = make_float3(0.0053, 0.0002, 0.0252);
	lambdas[393] = make_float3(0.0060, 0.0002, 0.0284);
	lambdas[394] = make_float3(0.0068, 0.0002, 0.0320);
	lambdas[395] = make_float3(0.0077, 0.0002, 0.0362);
	lambdas[396] = make_float3(0.0088, 0.0002, 0.0415);
	lambdas[397] = make_float3(0.0100, 0.0003, 0.0473);
	lambdas[398] = make_float3(0.0113, 0.0003, 0.0536);
	lambdas[399] = make_float3(0.0128, 0.0004, 0.0605);
	lambdas[400] = make_float3(0.0143, 0.0004, 0.0679);
	lambdas[401] = make_float3(0.0156, 0.0004, 0.0741);
	lambdas[402] = make_float3(0.0171, 0.0005, 0.0810);
	lambdas[403] = make_float3(0.0188, 0.0005, 0.0891);
	lambdas[404] = make_float3(0.0208, 0.0006, 0.0988);
	lambdas[405] = make_float3(0.0232, 0.0006, 0.1102);
	lambdas[406] = make_float3(0.0263, 0.0007, 0.1249);
	lambdas[407] = make_float3(0.0298, 0.0008, 0.1418);
	lambdas[408] = make_float3(0.0339, 0.0009, 0.1612);
	lambdas[409] = make_float3(0.0384, 0.0011, 0.1830);
	lambdas[410] = make_float3(0.0435, 0.0012, 0.2074);
	lambdas[411] = make_float3(0.0489, 0.0014, 0.2334);
	lambdas[412] = make_float3(0.0550, 0.0015, 0.2625);
	lambdas[413] = make_float3(0.0618, 0.0017, 0.2949);
	lambdas[414] = make_float3(0.0693, 0.0019, 0.3311);
	lambdas[415] = make_float3(0.0776, 0.0022, 0.3713);
	lambdas[416] = make_float3(0.0871, 0.0025, 0.4170);
	lambdas[417] = make_float3(0.0976, 0.0028, 0.4673);
	lambdas[418] = make_float3(0.1089, 0.0031, 0.5221);
	lambdas[419] = make_float3(0.1212, 0.0035, 0.5815);
	lambdas[420] = make_float3(0.1344, 0.0040, 0.6456);
	lambdas[421] = make_float3(0.1497, 0.0046, 0.7201);
	lambdas[422] = make_float3(0.1657, 0.0052, 0.7980);
	lambdas[423] = make_float3(0.1820, 0.0058, 0.8780);
	lambdas[424] = make_float3(0.1985, 0.0065, 0.9588);
	lambdas[425] = make_float3(0.2148, 0.0073, 1.0391);
	lambdas[426] = make_float3(0.2299, 0.0081, 1.1141);
	lambdas[427] = make_float3(0.2445, 0.0089, 1.1868);
	lambdas[428] = make_float3(0.2584, 0.0098, 1.2566);
	lambdas[429] = make_float3(0.2716, 0.0107, 1.3230);
	lambdas[430] = make_float3(0.2839, 0.0116, 1.3856);
	lambdas[431] = make_float3(0.2948, 0.0126, 1.4419);
	lambdas[432] = make_float3(0.3047, 0.0136, 1.4939);
	lambdas[433] = make_float3(0.3136, 0.0146, 1.5414);
	lambdas[434] = make_float3(0.3216, 0.0157, 1.5844);
	lambdas[435] = make_float3(0.3285, 0.0168, 1.6230);
	lambdas[436] = make_float3(0.3343, 0.0180, 1.6561);
	lambdas[437] = make_float3(0.3391, 0.0192, 1.6848);
	lambdas[438] = make_float3(0.3430, 0.0204, 1.7094);
	lambdas[439] = make_float3(0.3461, 0.0217, 1.7301);
	lambdas[440] = make_float3(0.3483, 0.0230, 1.7471);
	lambdas[441] = make_float3(0.3496, 0.0243, 1.7599);
	lambdas[442] = make_float3(0.3501, 0.0256, 1.7695);
	lambdas[443] = make_float3(0.3500, 0.0270, 1.7763);
	lambdas[444] = make_float3(0.3493, 0.0284, 1.7805);
	lambdas[445] = make_float3(0.3481, 0.0298, 1.7826);
	lambdas[446] = make_float3(0.3464, 0.0313, 1.7833);
	lambdas[447] = make_float3(0.3444, 0.0329, 1.7823);
	lambdas[448] = make_float3(0.3420, 0.0345, 1.7800);
	lambdas[449] = make_float3(0.3392, 0.0362, 1.7765);
	lambdas[450] = make_float3(0.3362, 0.0380, 1.7721);
	lambdas[451] = make_float3(0.3333, 0.0398, 1.7688);
	lambdas[452] = make_float3(0.3301, 0.0418, 1.7647);
	lambdas[453] = make_float3(0.3267, 0.0438, 1.7593);
	lambdas[454] = make_float3(0.3229, 0.0458, 1.7525);
	lambdas[455] = make_float3(0.3187, 0.0480, 1.7441);
	lambdas[456] = make_float3(0.3140, 0.0502, 1.7335);
	lambdas[457] = make_float3(0.3089, 0.0526, 1.7208);
	lambdas[458] = make_float3(0.3033, 0.0550, 1.7060);
	lambdas[459] = make_float3(0.2973, 0.0574, 1.6889);
	lambdas[460] = make_float3(0.2908, 0.0600, 1.6692);
	lambdas[461] = make_float3(0.2839, 0.0626, 1.6473);
	lambdas[462] = make_float3(0.2766, 0.0653, 1.6226);
	lambdas[463] = make_float3(0.2687, 0.0680, 1.5946);
	lambdas[464] = make_float3(0.2602, 0.0709, 1.5632);
	lambdas[465] = make_float3(0.2511, 0.0739, 1.5281);
	lambdas[466] = make_float3(0.2406, 0.0770, 1.4849);
	lambdas[467] = make_float3(0.2297, 0.0803, 1.4386);
	lambdas[468] = make_float3(0.2184, 0.0837, 1.3897);
	lambdas[469] = make_float3(0.2069, 0.0872, 1.3392);
	lambdas[470] = make_float3(0.1954, 0.0910, 1.2876);
	lambdas[471] = make_float3(0.1844, 0.0949, 1.2382);
	lambdas[472] = make_float3(0.1735, 0.0991, 1.1887);
	lambdas[473] = make_float3(0.1628, 0.1034, 1.1394);
	lambdas[474] = make_float3(0.1523, 0.1079, 1.0904);
	lambdas[475] = make_float3(0.1421, 0.1126, 1.0419);
	lambdas[476] = make_float3(0.1322, 0.1175, 0.9943);
	lambdas[477] = make_float3(0.1226, 0.1226, 0.9474);
	lambdas[478] = make_float3(0.1133, 0.1279, 0.9015);
	lambdas[479] = make_float3(0.1043, 0.1334, 0.8567);
	lambdas[480] = make_float3(0.0956, 0.1390, 0.8130);
	lambdas[481] = make_float3(0.0873, 0.1446, 0.7706);
	lambdas[482] = make_float3(0.0793, 0.1504, 0.7296);
	lambdas[483] = make_float3(0.0718, 0.1564, 0.6902);
	lambdas[484] = make_float3(0.0646, 0.1627, 0.6523);
	lambdas[485] = make_float3(0.0580, 0.1693, 0.6162);
	lambdas[486] = make_float3(0.0519, 0.1763, 0.5825);
	lambdas[487] = make_float3(0.0463, 0.1836, 0.5507);
	lambdas[488] = make_float3(0.0412, 0.1913, 0.5205);
	lambdas[489] = make_float3(0.0364, 0.1994, 0.4920);
	lambdas[490] = make_float3(0.0320, 0.2080, 0.4652);
	lambdas[491] = make_float3(0.0279, 0.2171, 0.4399);
	lambdas[492] = make_float3(0.0241, 0.2267, 0.4162);
	lambdas[493] = make_float3(0.0207, 0.2368, 0.3939);
	lambdas[494] = make_float3(0.0175, 0.2474, 0.3730);
	lambdas[495] = make_float3(0.0147, 0.2586, 0.3533);
	lambdas[496] = make_float3(0.0121, 0.2702, 0.3349);
	lambdas[497] = make_float3(0.0099, 0.2824, 0.3176);
	lambdas[498] = make_float3(0.0079, 0.2952, 0.3014);
	lambdas[499] = make_float3(0.0063, 0.3087, 0.2862);
	lambdas[500] = make_float3(0.0049, 0.3230, 0.2720);
	lambdas[501] = make_float3(0.0037, 0.3385, 0.2588);
	lambdas[502] = make_float3(0.0029, 0.3548, 0.2464);
	lambdas[503] = make_float3(0.0024, 0.3717, 0.2346);
	lambdas[504] = make_float3(0.0022, 0.3893, 0.2233);
	lambdas[505] = make_float3(0.0024, 0.4073, 0.2123);
	lambdas[506] = make_float3(0.0029, 0.4256, 0.2010);
	lambdas[507] = make_float3(0.0038, 0.4443, 0.1899);
	lambdas[508] = make_float3(0.0052, 0.4635, 0.1790);
	lambdas[509] = make_float3(0.0070, 0.4830, 0.1685);
	lambdas[510] = make_float3(0.0093, 0.5030, 0.1582);
	lambdas[511] = make_float3(0.0122, 0.5237, 0.1481);
	lambdas[512] = make_float3(0.0156, 0.5447, 0.1384);
	lambdas[513] = make_float3(0.0195, 0.5658, 0.1290);
	lambdas[514] = make_float3(0.0240, 0.5870, 0.1201);
	lambdas[515] = make_float3(0.0291, 0.6082, 0.1117);
	lambdas[516] = make_float3(0.0349, 0.6293, 0.1040);
	lambdas[517] = make_float3(0.0412, 0.6502, 0.0968);
	lambdas[518] = make_float3(0.0480, 0.6707, 0.0901);
	lambdas[519] = make_float3(0.0554, 0.6906, 0.0839);
	lambdas[520] = make_float3(0.0633, 0.7100, 0.0782);
	lambdas[521] = make_float3(0.0716, 0.7280, 0.0733);
	lambdas[522] = make_float3(0.0805, 0.7453, 0.0687);
	lambdas[523] = make_float3(0.0898, 0.7619, 0.0646);
	lambdas[524] = make_float3(0.0995, 0.7778, 0.0608);
	lambdas[525] = make_float3(0.1096, 0.7932, 0.0573);
	lambdas[526] = make_float3(0.1202, 0.8082, 0.0539);
	lambdas[527] = make_float3(0.1311, 0.8225, 0.0507);
	lambdas[528] = make_float3(0.1423, 0.8363, 0.0477);
	lambdas[529] = make_float3(0.1538, 0.8495, 0.0449);
	lambdas[530] = make_float3(0.1655, 0.8620, 0.0422);
	lambdas[531] = make_float3(0.1772, 0.8738, 0.0395);
	lambdas[532] = make_float3(0.1891, 0.8849, 0.0369);
	lambdas[533] = make_float3(0.2011, 0.8955, 0.0344);
	lambdas[534] = make_float3(0.2133, 0.9054, 0.0321);
	lambdas[535] = make_float3(0.2257, 0.9149, 0.0298);
	lambdas[536] = make_float3(0.2383, 0.9237, 0.0277);
	lambdas[537] = make_float3(0.2511, 0.9321, 0.0257);
	lambdas[538] = make_float3(0.2640, 0.9399, 0.0238);
	lambdas[539] = make_float3(0.2771, 0.9472, 0.0220);
	lambdas[540] = make_float3(0.2904, 0.9540, 0.0203);
	lambdas[541] = make_float3(0.3039, 0.9602, 0.0187);
	lambdas[542] = make_float3(0.3176, 0.9660, 0.0172);
	lambdas[543] = make_float3(0.3314, 0.9712, 0.0159);
	lambdas[544] = make_float3(0.3455, 0.9760, 0.0146);
	lambdas[545] = make_float3(0.3597, 0.9803, 0.0134);
	lambdas[546] = make_float3(0.3741, 0.9841, 0.0123);
	lambdas[547] = make_float3(0.3886, 0.9874, 0.0113);
	lambdas[548] = make_float3(0.4034, 0.9904, 0.0104);
	lambdas[549] = make_float3(0.4183, 0.9929, 0.0095);
	lambdas[550] = make_float3(0.4334, 0.9950, 0.0087);
	lambdas[551] = make_float3(0.4488, 0.9967, 0.0080);
	lambdas[552] = make_float3(0.4644, 0.9981, 0.0074);
	lambdas[553] = make_float3(0.4801, 0.9992, 0.0068);
	lambdas[554] = make_float3(0.4960, 0.9998, 0.0062);
	lambdas[555] = make_float3(0.5121, 1.0000, 0.0057);
	lambdas[556] = make_float3(0.5283, 0.9998, 0.0053);
	lambdas[557] = make_float3(0.5447, 0.9993, 0.0049);
	lambdas[558] = make_float3(0.5612, 0.9983, 0.0045);
	lambdas[559] = make_float3(0.5778, 0.9969, 0.0042);
	lambdas[560] = make_float3(0.5945, 0.9950, 0.0039);
	lambdas[561] = make_float3(0.6112, 0.9926, 0.0036);
	lambdas[562] = make_float3(0.6280, 0.9897, 0.0034);
	lambdas[563] = make_float3(0.6448, 0.9865, 0.0031);
	lambdas[564] = make_float3(0.6616, 0.9827, 0.0029);
	lambdas[565] = make_float3(0.6784, 0.9786, 0.0027);
	lambdas[566] = make_float3(0.6953, 0.9741, 0.0026);
	lambdas[567] = make_float3(0.7121, 0.9692, 0.0024);
	lambdas[568] = make_float3(0.7288, 0.9639, 0.0023);
	lambdas[569] = make_float3(0.7455, 0.9581, 0.0022);
	lambdas[570] = make_float3(0.7621, 0.9520, 0.0021);
	lambdas[571] = make_float3(0.7785, 0.9454, 0.0020);
	lambdas[572] = make_float3(0.7948, 0.9385, 0.0019);
	lambdas[573] = make_float3(0.8109, 0.9312, 0.0019);
	lambdas[574] = make_float3(0.8268, 0.9235, 0.0018);
	lambdas[575] = make_float3(0.8425, 0.9154, 0.0018);
	lambdas[576] = make_float3(0.8579, 0.9070, 0.0018);
	lambdas[577] = make_float3(0.8731, 0.8983, 0.0017);
	lambdas[578] = make_float3(0.8879, 0.8892, 0.0017);
	lambdas[579] = make_float3(0.9023, 0.8798, 0.0017);
	lambdas[580] = make_float3(0.9163, 0.8700, 0.0017);
	lambdas[581] = make_float3(0.9298, 0.8598, 0.0016);
	lambdas[582] = make_float3(0.9428, 0.8494, 0.0016);
	lambdas[583] = make_float3(0.9553, 0.8386, 0.0015);
	lambdas[584] = make_float3(0.9672, 0.8276, 0.0015);
	lambdas[585] = make_float3(0.9786, 0.8163, 0.0014);
	lambdas[586] = make_float3(0.9894, 0.8048, 0.0013);
	lambdas[587] = make_float3(0.9996, 0.7931, 0.0013);
	lambdas[588] = make_float3(1.0091, 0.7812, 0.0012);
	lambdas[589] = make_float3(1.0181, 0.7692, 0.0012);
	lambdas[590] = make_float3(1.0263, 0.7570, 0.0011);
	lambdas[591] = make_float3(1.0340, 0.7448, 0.0011);
	lambdas[592] = make_float3(1.0410, 0.7324, 0.0011);
	lambdas[593] = make_float3(1.0471, 0.7200, 0.0010);
	lambdas[594] = make_float3(1.0524, 0.7075, 0.0010);
	lambdas[595] = make_float3(1.0567, 0.6949, 0.0010);
	lambdas[596] = make_float3(1.0597, 0.6822, 0.0010);
	lambdas[597] = make_float3(1.0617, 0.6695, 0.0009);
	lambdas[598] = make_float3(1.0628, 0.6567, 0.0009);
	lambdas[599] = make_float3(1.0630, 0.6439, 0.0008);
	lambdas[600] = make_float3(1.0622, 0.6310, 0.0008);
	lambdas[601] = make_float3(1.0608, 0.6182, 0.0008);
	lambdas[602] = make_float3(1.0585, 0.6053, 0.0007);
	lambdas[603] = make_float3(1.0552, 0.5925, 0.0007);
	lambdas[604] = make_float3(1.0509, 0.5796, 0.0006);
	lambdas[605] = make_float3(1.0456, 0.5668, 0.0006);
	lambdas[606] = make_float3(1.0389, 0.5540, 0.0005);
	lambdas[607] = make_float3(1.0313, 0.5411, 0.0005);
	lambdas[608] = make_float3(1.0226, 0.5284, 0.0004);
	lambdas[609] = make_float3(1.0131, 0.5157, 0.0004);
	lambdas[610] = make_float3(1.0026, 0.5030, 0.0003);
	lambdas[611] = make_float3(0.9914, 0.4905, 0.0003);
	lambdas[612] = make_float3(0.9794, 0.4781, 0.0003);
	lambdas[613] = make_float3(0.9665, 0.4657, 0.0003);
	lambdas[614] = make_float3(0.9529, 0.4534, 0.0003);
	lambdas[615] = make_float3(0.9384, 0.4412, 0.0002);
	lambdas[616] = make_float3(0.9232, 0.4291, 0.0002);
	lambdas[617] = make_float3(0.9072, 0.4170, 0.0002);
	lambdas[618] = make_float3(0.8904, 0.4050, 0.0002);
	lambdas[619] = make_float3(0.8728, 0.3930, 0.0002);
	lambdas[620] = make_float3(0.8544, 0.3810, 0.0002);
	lambdas[621] = make_float3(0.8349, 0.3689, 0.0002);
	lambdas[622] = make_float3(0.8148, 0.3568, 0.0002);
	lambdas[623] = make_float3(0.7941, 0.3447, 0.0001);
	lambdas[624] = make_float3(0.7729, 0.3328, 0.0001);
	lambdas[625] = make_float3(0.7514, 0.3210, 0.0001);
	lambdas[626] = make_float3(0.7296, 0.3094, 0.0001);
	lambdas[627] = make_float3(0.7077, 0.2979, 0.0001);
	lambdas[628] = make_float3(0.6858, 0.2867, 0.0001);
	lambdas[629] = make_float3(0.6640, 0.2757, 0.0001);
	lambdas[630] = make_float3(0.6424, 0.2650, 0.0000);
	lambdas[631] = make_float3(0.6217, 0.2548, 0.0000);
	lambdas[632] = make_float3(0.6013, 0.2450, 0.0000);
	lambdas[633] = make_float3(0.5812, 0.2354, 0.0000);
	lambdas[634] = make_float3(0.5614, 0.2261, 0.0000);
	lambdas[635] = make_float3(0.5419, 0.2170, 0.0000);
	lambdas[636] = make_float3(0.5226, 0.2081, 0.0000);
	lambdas[637] = make_float3(0.5035, 0.1995, 0.0000);
	lambdas[638] = make_float3(0.4847, 0.1911, 0.0000);
	lambdas[639] = make_float3(0.4662, 0.1830, 0.0000);
	lambdas[640] = make_float3(0.4479, 0.1750, 0.0000);
	lambdas[641] = make_float3(0.4298, 0.1672, 0.0000);
	lambdas[642] = make_float3(0.4121, 0.1596, 0.0000);
	lambdas[643] = make_float3(0.3946, 0.1523, 0.0000);
	lambdas[644] = make_float3(0.3775, 0.1451, 0.0000);
	lambdas[645] = make_float3(0.3608, 0.1382, 0.0000);
	lambdas[646] = make_float3(0.3445, 0.1315, 0.0000);
	lambdas[647] = make_float3(0.3286, 0.1250, 0.0000);
	lambdas[648] = make_float3(0.3131, 0.1188, 0.0000);
	lambdas[649] = make_float3(0.2980, 0.1128, 0.0000);
	lambdas[650] = make_float3(0.2835, 0.1070, 0.0000);
	lambdas[651] = make_float3(0.2696, 0.1015, 0.0000);
	lambdas[652] = make_float3(0.2562, 0.0962, 0.0000);
	lambdas[653] = make_float3(0.2432, 0.0911, 0.0000);
	lambdas[654] = make_float3(0.2307, 0.0863, 0.0000);
	lambdas[655] = make_float3(0.2187, 0.0816, 0.0000);
	lambdas[656] = make_float3(0.2071, 0.0771, 0.0000);
	lambdas[657] = make_float3(0.1959, 0.0728, 0.0000);
	lambdas[658] = make_float3(0.1852, 0.0687, 0.0000);
	lambdas[659] = make_float3(0.1748, 0.0648, 0.0000);
	lambdas[660] = make_float3(0.1649, 0.0610, 0.0000);
	lambdas[661] = make_float3(0.1554, 0.0574, 0.0000);
	lambdas[662] = make_float3(0.1462, 0.0539, 0.0000);
	lambdas[663] = make_float3(0.1375, 0.0507, 0.0000);
	lambdas[664] = make_float3(0.1291, 0.0475, 0.0000);
	lambdas[665] = make_float3(0.1212, 0.0446, 0.0000);
	lambdas[666] = make_float3(0.1136, 0.0418, 0.0000);
	lambdas[667] = make_float3(0.1065, 0.0391, 0.0000);
	lambdas[668] = make_float3(0.0997, 0.0366, 0.0000);
	lambdas[669] = make_float3(0.0934, 0.0342, 0.0000);
	lambdas[670] = make_float3(0.0874, 0.0320, 0.0000);
	lambdas[671] = make_float3(0.0819, 0.0300, 0.0000);
	lambdas[672] = make_float3(0.0768, 0.0281, 0.0000);
	lambdas[673] = make_float3(0.0721, 0.0263, 0.0000);
	lambdas[674] = make_float3(0.0677, 0.0247, 0.0000);
	lambdas[675] = make_float3(0.0636, 0.0232, 0.0000);
	lambdas[676] = make_float3(0.0598, 0.0218, 0.0000);
	lambdas[677] = make_float3(0.0563, 0.0205, 0.0000);
	lambdas[678] = make_float3(0.0529, 0.0193, 0.0000);
	lambdas[679] = make_float3(0.0498, 0.0181, 0.0000);
	lambdas[680] = make_float3(0.0468, 0.0170, 0.0000);
	lambdas[681] = make_float3(0.0437, 0.0159, 0.0000);
	lambdas[682] = make_float3(0.0408, 0.0148, 0.0000);
	lambdas[683] = make_float3(0.0380, 0.0138, 0.0000);
	lambdas[684] = make_float3(0.0354, 0.0128, 0.0000);
	lambdas[685] = make_float3(0.0329, 0.0119, 0.0000);
	lambdas[686] = make_float3(0.0306, 0.0111, 0.0000);
	lambdas[687] = make_float3(0.0284, 0.0103, 0.0000);
	lambdas[688] = make_float3(0.0264, 0.0095, 0.0000);
	lambdas[689] = make_float3(0.0245, 0.0088, 0.0000);
	lambdas[690] = make_float3(0.0227, 0.0082, 0.0000);
	lambdas[691] = make_float3(0.0211, 0.0076, 0.0000);
	lambdas[692] = make_float3(0.0196, 0.0071, 0.0000);
	lambdas[693] = make_float3(0.0182, 0.0066, 0.0000);
	lambdas[694] = make_float3(0.0170, 0.0061, 0.0000);
	lambdas[695] = make_float3(0.0158, 0.0057, 0.0000);
	lambdas[696] = make_float3(0.0148, 0.0053, 0.0000);
	lambdas[697] = make_float3(0.0138, 0.0050, 0.0000);
	lambdas[698] = make_float3(0.0129, 0.0047, 0.0000);
	lambdas[699] = make_float3(0.0121, 0.0044, 0.0000);
	lambdas[700] = make_float3(0.0114, 0.0041, 0.0000);
	lambdas[701] = make_float3(0.0106, 0.0038, 0.0000);
	lambdas[702] = make_float3(0.0099, 0.0036, 0.0000);
	lambdas[703] = make_float3(0.0093, 0.0034, 0.0000);
	lambdas[704] = make_float3(0.0087, 0.0031, 0.0000);
	lambdas[705] = make_float3(0.0081, 0.0029, 0.0000);
	lambdas[706] = make_float3(0.0076, 0.0027, 0.0000);
	lambdas[707] = make_float3(0.0071, 0.0026, 0.0000);
	lambdas[708] = make_float3(0.0066, 0.0024, 0.0000);
	lambdas[709] = make_float3(0.0062, 0.0022, 0.0000);
	lambdas[710] = make_float3(0.0058, 0.0021, 0.0000);
	lambdas[711] = make_float3(0.0054, 0.0020, 0.0000);
	lambdas[712] = make_float3(0.0051, 0.0018, 0.0000);
	lambdas[713] = make_float3(0.0047, 0.0017, 0.0000);
	lambdas[714] = make_float3(0.0044, 0.0016, 0.0000);
	lambdas[715] = make_float3(0.0041, 0.0015, 0.0000);
	lambdas[716] = make_float3(0.0038, 0.0014, 0.0000);
	lambdas[717] = make_float3(0.0036, 0.0013, 0.0000);
	lambdas[718] = make_float3(0.0033, 0.0012, 0.0000);
	lambdas[719] = make_float3(0.0031, 0.0011, 0.0000);
	lambdas[720] = make_float3(0.0029, 0.0010, 0.0000);
	lambdas[721] = make_float3(0.0027, 0.0010, 0.0000);
	lambdas[722] = make_float3(0.0025, 0.0009, 0.0000);
	lambdas[723] = make_float3(0.0024, 0.0008, 0.0000);
	lambdas[724] = make_float3(0.0022, 0.0008, 0.0000);
	lambdas[725] = make_float3(0.0020, 0.0007, 0.0000);
	lambdas[726] = make_float3(0.0019, 0.0007, 0.0000);
	lambdas[727] = make_float3(0.0018, 0.0006, 0.0000);
	lambdas[728] = make_float3(0.0017, 0.0006, 0.0000);
	lambdas[729] = make_float3(0.0015, 0.0006, 0.0000);
	lambdas[730] = make_float3(0.0014, 0.0005, 0.0000);
	lambdas[731] = make_float3(0.0013, 0.0005, 0.0000);
	lambdas[732] = make_float3(0.0012, 0.0004, 0.0000);
	lambdas[733] = make_float3(0.0012, 0.0004, 0.0000);
	lambdas[734] = make_float3(0.0011, 0.0004, 0.0000);
	lambdas[735] = make_float3(0.0010, 0.0004, 0.0000);
	lambdas[736] = make_float3(0.0009, 0.0003, 0.0000);
	lambdas[737] = make_float3(0.0009, 0.0003, 0.0000);
	lambdas[738] = make_float3(0.0008, 0.0003, 0.0000);
	lambdas[739] = make_float3(0.0007, 0.0003, 0.0000);
	lambdas[740] = make_float3(0.0007, 0.0002, 0.0000);
	lambdas[741] = make_float3(0.0006, 0.0002, 0.0000);
	lambdas[742] = make_float3(0.0006, 0.0002, 0.0000);
	lambdas[743] = make_float3(0.0006, 0.0002, 0.0000);
	lambdas[744] = make_float3(0.0005, 0.0002, 0.0000);
	lambdas[745] = make_float3(0.0005, 0.0002, 0.0000);
	lambdas[746] = make_float3(0.0004, 0.0002, 0.0000);
	lambdas[747] = make_float3(0.0004, 0.0001, 0.0000);
	lambdas[748] = make_float3(0.0004, 0.0001, 0.0000);
	lambdas[749] = make_float3(0.0004, 0.0001, 0.0000);
	lambdas[750] = make_float3(0.0003, 0.0001, 0.0000);
	lambdas[751] = make_float3(0.0003, 0.0001, 0.0000);
	lambdas[752] = make_float3(0.0003, 0.0001, 0.0000);
	lambdas[753] = make_float3(0.0003, 0.0001, 0.0000);
	lambdas[754] = make_float3(0.0003, 0.0001, 0.0000);
	lambdas[755] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[756] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[757] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[758] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[759] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[760] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[761] = make_float3(0.0002, 0.0001, 0.0000);
	lambdas[762] = make_float3(0.0001, 0.0001, 0.0000);
	lambdas[763] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[764] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[765] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[766] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[767] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[768] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[769] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[770] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[771] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[772] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[773] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[774] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[775] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[776] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[777] = make_float3(0.0001, 0.0000, 0.0000);
	lambdas[778] = make_float3(0.0000, 0.0000, 0.0000);
	lambdas[779] = make_float3(0.0000, 0.0000, 0.0000);
	lambdas[780] = make_float3(0.0000, 0.0000, 0.0000);

	memcpy(lambda_buffer->map(), lambdas, sizeof(float3) * 781);
	lambda_buffer->unmap();
}


Transform m_transform;
Group m_group;

Transform l_transform;
Group l_group;

void loadGeometry(int diffuse_id)
{
	using namespace std;

	m_group = context->createGroup();
	l_group = context->createGroup();
	ParallelogramLight light;

	// set up light pass program
	const char* light_ptx = sutil::getPtxString(SAMPLE_NAME, "LightPass.cu");
	Program photon_diffuse_ch = context->createProgramFromPTXString(light_ptx, "diffuse");
	Program photon_glass_ch = context->createProgramFromPTXString(light_ptx, "glass");
	Program photon_metal_ch = context->createProgramFromPTXString(light_ptx, "metal");

	// set up material for eye pass
	const char* eye_ptx = sutil::getPtxString(SAMPLE_NAME, "EyePass.cu");
	Program diffuse_ch = context->createProgramFromPTXString(eye_ptx, "diffuse");
	Program diffuse_direct_light_ch = context->createProgramFromPTXString(eye_ptx, "diffuse_direct_light"); // debug closest hit
	Program diffuse_ah = context->createProgramFromPTXString(eye_ptx, "shadow");
	Program diffuse_em = context->createProgramFromPTXString(eye_ptx, "diffuseEmitter");
	Program glass_ch = context->createProgramFromPTXString(eye_ptx, "glass");
	Program metal_ch = context->createProgramFromPTXString(eye_ptx, "metal");

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	//const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);
	const float3 bluescreen = make_float3(42.0f / 255.0f, 107.0f / 255.0f, 220.0f / 255.0f);

	OptiXMesh mesh_photon_diffuse;
	mesh_photon_diffuse.context = context;
	mesh_photon_diffuse.use_tri_api = use_tri_api;
	mesh_photon_diffuse.ignore_mats = ignore_mats;
	mesh_photon_diffuse.closest_hit = photon_diffuse_ch;

	OptiXMesh mesh_photon_glass;
	mesh_photon_glass.context = context;
	mesh_photon_glass.use_tri_api = use_tri_api;
	mesh_photon_glass.ignore_mats = ignore_mats;
	mesh_photon_glass.closest_hit = photon_glass_ch;

	OptiXMesh mesh_photon_metal;
	mesh_photon_metal.context = context;
	mesh_photon_metal.use_tri_api = use_tri_api;
	mesh_photon_metal.ignore_mats = ignore_mats;
	mesh_photon_metal.closest_hit = photon_metal_ch;

	OptiXMesh mesh_light;
	mesh_light.use_tri_api = true;
	mesh_light.ignore_mats = false;
	mesh_light.closest_hit = diffuse_em;
	mesh_light.any_hit = diffuse_ah;
	mesh_light.context = context;

	OptiXMesh mesh_glass;
	mesh_glass.context = context;
	mesh_glass.use_tri_api = use_tri_api;
	mesh_glass.ignore_mats = ignore_mats;
	mesh_glass.any_hit = diffuse_ah;
	mesh_glass.closest_hit = glass_ch;

	OptiXMesh mesh_diffuse;
	mesh_diffuse.context = context;
	mesh_diffuse.use_tri_api = use_tri_api;
	mesh_diffuse.ignore_mats = ignore_mats;
	mesh_diffuse.any_hit = diffuse_ah;

	if (diffuse_id < 3) mesh_diffuse.closest_hit = diffuse_ch;
	if (diffuse_id == 3) mesh_diffuse.closest_hit = diffuse_direct_light_ch;
	
	OptiXMesh mesh_metal;
	mesh_metal.use_tri_api = true;
	mesh_metal.ignore_mats = false;
	mesh_metal.closest_hit = metal_ch;
	mesh_metal.any_hit = diffuse_ah;
	mesh_metal.context = context;

	//fstream  Input("scene1.txt");
	fstream  Input("scene.txt");
	string input_type;
	float tmp_float, tmp_C, tmp_x, tmp_y, tmp_z;
	while (Input >> input_type) // input the scene
	{
		if (input_type[0] == 'e')
		{
			Input >> tmp_x >> tmp_y >> tmp_z;
			if(diffuse_id == 0) camera_eye = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			if (diffuse_id == 0) camera_lookat = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			if (diffuse_id == 0) camera_up = make_float3(tmp_x, tmp_y, tmp_z);
			Input.ignore();
		}
		else if (input_type[0] == 'l')
		{
			Input >> tmp_x >> tmp_y >> tmp_z;
			light.corner = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			light.v1 = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			light.v2 = make_float3(tmp_x, tmp_y, tmp_z);
			light.normal = -normalize(cross(light.v1, light.v2));
			Input >> tmp_x >> tmp_y >> tmp_z;
			light.emission = make_float3(tmp_x, tmp_y, tmp_z);
			Input.ignore();
		}
		else if (input_type[0] == 'o')
		{
			string obj_name, material;
			Input >> obj_name;
			Input >> material;
			GeometryGroup geometry_group = context->createGeometryGroup();
			GeometryGroup light_geometry_group = context->createGeometryGroup();
			if (material == "diffuse")
			{
				loadMesh(obj_name, mesh_diffuse);
				Input >> tmp_float;
				Input >> tmp_x >> tmp_y >> tmp_z;
				mesh_diffuse.geom_instance["Kd"]->setFloat(tmp_float);
				mesh_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				geometry_group->addChild(mesh_diffuse.geom_instance);

				// set up light pass
				loadMesh(obj_name, mesh_photon_diffuse);
				mesh_photon_diffuse.geom_instance["Kd"]->setFloat(tmp_float);
				mesh_photon_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				light_geometry_group->addChild(mesh_photon_diffuse.geom_instance);
			}
			else if (material == "metal")
			{
				Input >> tmp_x >> tmp_y >> tmp_z;
				
				loadMesh(obj_name, mesh_metal);
				mesh_metal.geom_instance["metal_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				geometry_group->addChild(mesh_metal.geom_instance);
				
				// set up light pass
				loadMesh(obj_name, mesh_photon_metal);
				light_geometry_group->addChild(mesh_photon_metal.geom_instance);
				
			}
			else if (material == "glass")
			{
				Input >> tmp_float;
				Input >> tmp_C;
				Input >> tmp_x >> tmp_y >> tmp_z;
				
				loadMesh(obj_name, mesh_glass);
				mesh_glass.geom_instance["fresnel_exponent"]->setFloat(4.0f);
				mesh_glass.geom_instance["fresnel_minimum"]->setFloat(0.1f);
				mesh_glass.geom_instance["fresnel_maximum"]->setFloat(1.0f);
				//mesh_glass.geom_instance["refraction_index"]->setFloat(tmp_float); 
				mesh_glass.geom_instance["B"]->setFloat(tmp_float); // Base IOR
				mesh_glass.geom_instance["C"]->setFloat(tmp_C); // 
				mesh_glass.geom_instance["refraction_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				mesh_glass.geom_instance["reflection_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				mesh_glass.geom_instance["extintion"]->setFloat(-(make_float3(log(0.905f), log(0.63f), log(0.3))));
				geometry_group->addChild(mesh_glass.geom_instance);
				
				// set up light pass
				loadMesh(obj_name, mesh_photon_glass);
				mesh_photon_glass.geom_instance["fresnel_exponent"]->setFloat(4.0f);
				mesh_photon_glass.geom_instance["fresnel_minimum"]->setFloat(0.1f);
				mesh_photon_glass.geom_instance["fresnel_maximum"]->setFloat(1.0f);
				mesh_photon_glass.geom_instance["refraction_index"]->setFloat(tmp_float);
				mesh_photon_glass.geom_instance["refraction_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				mesh_photon_glass.geom_instance["reflection_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				mesh_photon_glass.geom_instance["extintion"]->setFloat(-(make_float3(log(0.905f), log(0.63f), log(0.3))));
				light_geometry_group->addChild(mesh_photon_glass.geom_instance);
			}
			geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
			light_geometry_group->setAcceleration(context->createAcceleration("Trbvh"));

			Transform transform = context->createTransform();
			Transform light_transform = context->createTransform();
			transform->setChild(geometry_group);
			light_transform->setChild(light_geometry_group);
			// Rotation around (world) x-axis 
			Matrix4x4 m = Matrix4x4::identity();
			Input >> tmp_x >> tmp_y >> tmp_z;
			m = m * m.translate(make_float3(tmp_x, tmp_y, tmp_z));
			Input >> tmp_x >> tmp_y >> tmp_z;
			m = m * m.scale(make_float3(tmp_x, tmp_y, tmp_z));
			Input >> tmp_float >> tmp_x >> tmp_y >> tmp_z;
			m = m * m.rotate(tmp_float, make_float3(tmp_x, tmp_y, tmp_z));

			transform->setMatrix(false, m.getData(), NULL);
			light_transform->setMatrix(false, m.getData(), NULL);
			transform->validate();
			light_transform->validate();

			m_group->addChild(transform);
			l_group->addChild(light_transform);

			Input.ignore();
		}
	}

	//light
	GeometryGroup geometry_group = context->createGeometryGroup();
	loadMesh("1x1plane.obj", mesh_light);
	mesh_light.geom_instance["emission_color"]->setFloat(light.emission);
	geometry_group->addChild(mesh_light.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));

	Transform transform = context->createTransform();
	transform->setChild(geometry_group);
	// Rotation around (world) x-axis 
	Matrix4x4 m = Matrix4x4::identity();
	
	m = m * m.translate(light.corner + 0.5*light.v1 + 0.5*light.v2);
	m = m * m.scale(make_float3(length(light.v1), 1.0f, length(light.v2)));
	//m = m * m.scale(make_float3(0.02f, 1.0f, 0.02f)); //point
	//m = m * m.rotate(tmp_float, make_float3(tmp_x, tmp_y, tmp_z));

	transform->setMatrix(false, m.getData(), NULL);
	transform->validate();

	m_group->addChild(transform);


	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight)); // not good
	light_buffer->setSize(1u);
	memcpy(light_buffer->map(), &light, sizeof(ParallelogramLight));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);


	m_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["eye_object"]->set(m_group);
	context["eye_shadower"]->set(m_group);

	l_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["light_object"]->set(l_group);
	context["light_shadower"]->set(l_group);

	lambdaInitiallBuffer();
}


void setupCamera()
{
	//camera_eye = make_float3(15.0f, 4.0f, 0.0f);
	//camera_lookat = make_float3(0.0f, 4.0f, 0.0f);
	//camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}


void updateCamera()
{
	const float fov = 40.0f;
	const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

	const Matrix4x4 frame = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

	camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
	camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
	camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = Matrix4x4::identity();

	if (camera_changed) // reset accumulation
		frame_number = 1;
	camera_changed = false;

	context["frame_number"]->setUint(frame_number++);
	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);

}


void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
	photon_thread = photon_count / light_depth;
	context->launch(1, photon_thread); // light pass
	updateCamera();
	context->launch(0, width, height); // eye pass

	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	// botton text
	if      (output_mode == 1) sutil::displayText("Bidiretional Path Tracing", 140, 10);
	else if (output_mode == 2) sutil::displayText("Path Tracing", 140, 10);
	else if (output_mode == 3) sutil::displayText("Direct Light (Debug)", 140, 10);

	char str[64];
	sprintf(str, "#%d", frame_number);
	sutil::displayText(str, (float)width - 50, (float)height - 20);

	glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{
	
	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('s'):
	{
		const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer(), false);
		break;
	}
	case('0'): // reset camera
	{
		loadGeometry(0);
		frame_number = 1;
		context["frame_number"]->setUint(frame_number);
		photon_count = inital_photon_count;
		context["photon_count"]->setUint(photon_count);
		output_mode = 1;
		break;
	}
	case('1'): // bidirectional path tracing
	{
		loadGeometry(1);
		frame_number = 1;
		context["frame_number"]->setUint(frame_number);
		photon_count = inital_photon_count;
		context["photon_count"]->setUint(photon_count);
		output_mode = 1;
		break;
	}
	case('2'): // path tracing
	{
		loadGeometry(2);
		frame_number = 1;
		context["frame_number"]->setUint(frame_number);
		photon_count = 0;
		context["photon_count"]->setUint(photon_count);
		output_mode = 2;
		break;
	}
	case('3'): // direct light
	{
		loadGeometry(3);
		frame_number = 1;
		context["frame_number"]->setUint(frame_number);
		photon_count = 0;
		context["photon_count"]->setUint(photon_count);
		output_mode = 3;
		break;
	}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}


void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
		camera_changed = true;
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
							  static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
							  static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera_rotate = arcball.rotate(b, a);
		camera_changed = true;
	}

	mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	camera_changed = true;

	width = w;
	height = h;
	sutil::ensureMinimumSize(width, height);

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0)
{
	std::cerr << "\nUsage: " << argv0 << " [options]\n";
	std::cerr <<
		"App Options:\n"
		"  -h | --help               Print this usage message and exit.\n"
		"  -f | --file               Save single frame to file and exit.\n"
		"  -n | --nopbo              Disable GL interop for display buffer.\n"
		"  -d | --dim=<width>x<height> Set image dimensions. Defaults to 512x512\n"
		"App Keystrokes:\n"
		"  q  Quit\n"
		"  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
		<< std::endl;

	exit(1);
}


int main(int argc, char** argv)
{
	std::string out_file;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "-f" || arg == "--file")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			out_file = argv[++i];
		}
		else if (arg == "-n" || arg == "--nopbo")
		{
			use_pbo = false;
		}
		else if (arg.find("-d") == 0 || arg.find("--dim") == 0)
		{
			size_t index = arg.find_first_of('=');
			if (index == std::string::npos)
			{
				std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
				printUsageAndExit(argv[0]);
			}
			std::string dim = arg.substr(index + 1);
			try
			{
				sutil::parseDimensions(dim.c_str(), (int&)width, (int&)height);
			}
			catch (const Exception&)
			{
				std::cerr << "Option '" << arg << " is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
				printUsageAndExit(argv[0]);
			}
		}
		else
		{
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	try
	{
		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif

		createContext();
		setupCamera();
		loadGeometry(0);

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			//context->launch(1, photon_thread);
			updateCamera();
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
			destroyContext();
		}

		return 0;
	}
	SUTIL_CATCH(context->get())
}

