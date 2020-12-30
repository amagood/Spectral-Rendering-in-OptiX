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

#include <optixPathTracer\LightCut.h>
#include <optixPathTracer\KDTree.h>

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

int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 2;
int			   light_depth = 8;
int			   inital_photon_count = 200;
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
Buffer kdtree_buffer;
Buffer lightcut_buffer;

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
	//context->setRayGenerationProgram(0, context->createProgramFromPTXString(eye_ptx, "pathtrace_camera"));
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

	// Edge Edited - START
	lightcut_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	lightcut_buffer->setFormat(RT_FORMAT_USER);
	lightcut_buffer->setElementSize(sizeof(LightcutNode));
	lightcut_buffer->setSize(2 * photon_count);
	context["lightcut_buffer"]->set(lightcut_buffer);
	memset(lightcut_buffer->map(), 0, 2 * sizeof(LightcutNode) * photon_count);
	lightcut_buffer->unmap();
	context["lightcut_count"]->setUint(2 * photon_count);

	kdtree_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	kdtree_buffer->setFormat(RT_FORMAT_USER);
	kdtree_buffer->setElementSize(sizeof(KDTreeNode));
	kdtree_buffer->setSize(photon_count);
	context["kdtree_buffer"]->set(kdtree_buffer);
	memset(kdtree_buffer->map(), 0, sizeof(KDTreeNode) * photon_count);
	kdtree_buffer->unmap();
	// Edge Edited - END

	context["sqrt_num_samples"]->setUint(sqrt_num_samples);
	context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
	context["bg_color"]->setFloat(make_float3(0.0f));

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

	if (diffuse_id == 1) mesh_diffuse.closest_hit = diffuse_ch;
	
	OptiXMesh mesh_metal;
	mesh_metal.use_tri_api = true;
	mesh_metal.ignore_mats = false;
	mesh_metal.closest_hit = metal_ch;
	mesh_metal.any_hit = diffuse_ah;
	mesh_metal.context = context;

	//fstream  Input("scene1.txt");
	fstream  Input("scene.txt");
	string input_type;
	float tmp_float, tmp_x, tmp_y, tmp_z;
	while (Input >> input_type) // input the scene
	{
		if (input_type[0] == 'e')
		{
			Input >> tmp_x >> tmp_y >> tmp_z;
			camera_eye = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			camera_lookat = make_float3(tmp_x, tmp_y, tmp_z);
			Input >> tmp_x >> tmp_y >> tmp_z;
			camera_up = make_float3(tmp_x, tmp_y, tmp_z);
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
				Input >> tmp_x >> tmp_y >> tmp_z;
				mesh_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				geometry_group->addChild(mesh_diffuse.geom_instance);

				// set up light pass
				loadMesh(obj_name, mesh_photon_diffuse);
				mesh_photon_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
				light_geometry_group->addChild(mesh_photon_diffuse.geom_instance);
			}
			else if (material == "metal")
			{
				Input >> tmp_x >> tmp_y >> tmp_z;
				if (diffuse_id == 3)
				{
					loadMesh(obj_name, mesh_diffuse);
					mesh_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
					geometry_group->addChild(mesh_diffuse.geom_instance);
				}
				else
				{
					loadMesh(obj_name, mesh_metal);
					mesh_metal.geom_instance["metal_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
					geometry_group->addChild(mesh_metal.geom_instance);
					
					// set up light pass
					loadMesh(obj_name, mesh_photon_metal);
					light_geometry_group->addChild(mesh_photon_metal.geom_instance);
				}
			}
			else if (material == "glass")
			{
				Input >> tmp_float;
				Input >> tmp_x >> tmp_y >> tmp_z;
				if (diffuse_id == 3)
				{
					loadMesh(obj_name, mesh_diffuse);
					mesh_diffuse.geom_instance["diffuse_color"]->setFloat(make_float3(tmp_x, tmp_y, tmp_z));
					geometry_group->addChild(mesh_diffuse.geom_instance);
				}
				else
				{
					loadMesh(obj_name, mesh_glass);
					mesh_glass.geom_instance["fresnel_exponent"]->setFloat(4.0f);
					mesh_glass.geom_instance["fresnel_minimum"]->setFloat(0.1f);
					mesh_glass.geom_instance["fresnel_maximum"]->setFloat(1.0f);
					mesh_glass.geom_instance["refraction_index"]->setFloat(tmp_float); 
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
	updateCamera();
	context->launch(0, width, height);

	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

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
	case('1'):
	{
		loadGeometry(1);
		frame_number = 1;
		context["frame_number"]->setUint(frame_number);
		photon_count = inital_photon_count;
		context->launch(1, photon_thread);

		Photon* photonArray = (Photon*)photon_buffer->map();
		int last_valid = 0;
		int reset_count = 0;
		bool gap = false;
		Photon photon;
		for (int i = 0; i < photon_count; ++i) 
		{
			//printf("%f %f %f\n", photonArray[i].color.x, photonArray[i].color.y, photonArray[i].color.z);
			if (photonArray[i].energy >= 1e-6 && length(photonArray[i].color) > 0)
			{
				photonArray[last_valid] = photonArray[i];
				last_valid++;
				reset_count++;
			}
		}
		photon_buffer->unmap();
		photon_count = reset_count;
		context["photon_count"]->setUint(photon_count);
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
		loadGeometry(1);

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			updateCamera();
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
			destroyContext();
		}

		return 0;
	}
	SUTIL_CATCH(context->get())
}

