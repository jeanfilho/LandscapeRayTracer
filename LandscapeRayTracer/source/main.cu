#include <gl/freeglut.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void init();
void display(void);
void centerOnScreen();
void updatePixelBuffer();
glm::vec3 castRay(int pixel_x, int pixel_y);
glm::vec3 fineStep(glm::vec3 ray_origin, glm::vec3 ray_direction);
void getRaycastParameter(float ray_origin, float ray_direction, float cell_size, float *d, float *t);
void exit();
void loadPointData();


//  define the window position on screen
int window_x;
int window_y;

//  variables representing the window size
int window_width = 512;
int window_height = 512;

//  variable representing the window title
char *window_title = "Landscape Raytracer";

// grid
glm::vec3 grid_origin(0, 0, 0);
int grid_size = 1000;
float cell_size = 1.0f;
char *grid;
int coarse_factor = 10;
int coarse_grid_size;
float coarse_cell_size;
char *coarse_grid;

// camera variables
glm::vec3 camera_position(128,700,128);
glm::vec3 camera_forward(0, -1, 0);
glm::vec3 camera_up(1, 0, 0);
float frame_distance = 5;

// pixel array
glm::vec3 *pixel_array;
float frame_width = 256;
float frame_height = 256;

// macros
#define get_pixel(x, y) pixel_array[y * window_width + x]
#define get_voxel(x, y, z) grid[z * grid_size * grid_size + y * grid_size + x]
#define get_coarse_voxel(x, y, z) coarse_grid[z * coarse_grid_size * coarse_grid_size + y * coarse_grid_size + x]
#define sign(a) ((a > 0) - (a < 0))

// visualization parameters
int max_height = INT_MIN;
int min_height = INT_MAX;


// clock
std::chrono::system_clock sys_clock;
std::chrono::time_point<std::chrono::system_clock> last_frame, current_frame;
std::chrono::duration<float> delta_time;

//-------------------------------------------------------------------------
//  DEVICE
//-------------------------------------------------------------------------
__device__ char *d_grid;
__device__ char *d_coarse_grid;
__device__ glm::vec3 d_grid_origin(0, 0, 0);
__device__ int d_grid_size = 1000;
__device__ float d_cell_size = 1.0f;
__device__ int d_coarse_factor = 10;
__device__ int d_coarse_grid_size;
__device__ float d_coarse_cell_size;


//-------------------------------------------------------------------------
//  Program Main method.
//-------------------------------------------------------------------------
int main(int argc, char **argv)
{
	//  Connect to the windowing system + create a window
	//  with the specified dimensions and position
	//  + set the display mode + specify the window title.
	glutInit(&argc, argv);
	centerOnScreen();
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(window_x, window_y);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow(window_title);

	//  Set OpenGL program initial state.
	init();

	// Set the callback functions
	glutDisplayFunc(display);
	glutIdleFunc(display);
	atexit(exit);

	//  Start GLUT event processing loop
	glutMainLoop();

	return 0;
}

//-------------------------------------------------------------------------
//  Set OpenGL program initial state.
//-------------------------------------------------------------------------
void init()
{
	std::cout << "Setting up..." << std::endl;
	//  Set the frame buffer clear color to black. 
	glClearColor(0.0, 0.0, 0.0, 0.0);

	//initialize grids
	grid = new char[grid_size * grid_size * grid_size]();

	coarse_grid_size = grid_size / coarse_factor;
	coarse_cell_size = cell_size * coarse_factor;
	coarse_grid = new char[coarse_grid_size * coarse_grid_size * coarse_grid_size]();
	pixel_array = new glm::vec3[window_height * window_width]{glm::vec3(0,0,0)};
	loadPointData();
	
	current_frame = last_frame = sys_clock.now();

	std::cout << "Set up finished. Starting ray tracing..." << std::endl;
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//  OpenGL contents on the window.
//-------------------------------------------------------------------------
void display(void)
{
	//  Clear the window or more specifically the frame buffer...
	//  This happens by replacing all the contents of the frame
	//  buffer by the clear color (black in our case)
	glClear(GL_COLOR_BUFFER_BIT);

	//  Cast rays
	updatePixelBuffer();

	current_frame = sys_clock.now();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;

	std::cout << "FPS " << 1 / delta_time.count() << std::endl;

	//  Draw Pixels
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixel_array);

	//  Swap contents of backward and forward frame buffers
	glutSwapBuffers();
}

//-------------------------------------------------------------------------
//  Update the pixel buffer - cast a ray for each pixel
//-------------------------------------------------------------------------
void updatePixelBuffer()
{
	for(int y = 0; y < window_height; y++)
		for (int x = 0; x < window_width; x++)
		{
			glm::vec3 result = castRay(x, y);
			if(result.x >= 0)
			{
				get_pixel(x, y).r = result.r;
				get_pixel(x, y).g = result.g;
				get_pixel(x, y).b = result.b;
			}
			else
			{
				get_pixel(x, y).r = 0;
				get_pixel(x, y).g = 0;
				get_pixel(x, y).b = 0;
			}
		}
}

//-------------------------------------------------------------------------
//  Cast a ray through the coarse grid
//  Parameters: -pixel_x: position of the pixel on the frame along x axis
//				-pixel_y: position of the pixel on the frame along y axis
//
//  Return: - Color values of the intersected cell
//			- glm::vec3(-1,-1,-1) if there is no intersection
//
//  source: http://www.scratchapixel.com/lessons/advanced-rendering/introduction-acceleration-structure/grid
//-------------------------------------------------------------------------
glm::vec3 castRay(int pixel_x, int pixel_y)
{
	glm::vec3 ray_direction, ray_origin;
	//t: current offset from component; t_: offset from ray in one axis; d_: grid step to next intersection
	float t, tx, ty, tz, dx, dy, dz;
	int x, y, z;
	
	ray_direction = camera_forward * frame_distance
		+ camera_up * ((pixel_y - window_height/2)/frame_height)
		+ glm::cross(camera_forward, camera_up) * ((pixel_x - window_width/2)/frame_width);
	ray_origin = camera_position + ray_direction;
	ray_direction = glm::normalize(ray_direction);

	getRaycastParameter(ray_origin.x, ray_direction.x, coarse_cell_size, &dx, &tx);
	getRaycastParameter(ray_origin.y, ray_direction.y, coarse_cell_size, &dy, &ty);
	getRaycastParameter(ray_origin.z, ray_direction.z, coarse_cell_size, &dz, &tz);

	// Coarse traversal step
	if (tx < 0)	tx = FLT_MAX;
	if (ty < 0) ty = FLT_MAX;
	if (tz < 0)	tz = FLT_MAX;
	
	x = (int)(ray_origin.x / coarse_factor);
	y = (int)(ray_origin.y / coarse_factor);
	z = (int)(ray_origin.z / coarse_factor);

	if (x >= coarse_grid_size || x < 0 
		|| y >= coarse_grid_size || y < 0
		|| z >= coarse_grid_size || z < 0)
		return glm::vec3(-1, -1, -1);

	while (get_coarse_voxel(x, y, z) == 0)
	{
		if (tx < ty)
		{
			if (tx < tz)
			{
				x += sign(ray_direction.x);
				t = tx;
				tx += dx;
				if (x >= coarse_grid_size || x < 0)
					return glm::vec3(-1,-1,-1);
			}
			else
			{
				z += sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= coarse_grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
		else
		{
			if (ty < tz)
			{
				y += sign(ray_direction.y);
				t = ty;
				ty += dy;
				if (y >= coarse_grid_size || y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= coarse_grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}

	return fineStep(ray_origin + ray_direction*t, ray_direction);
}

//-------------------------------------------------------------------------
// Set the parameters for the ray casting
//-------------------------------------------------------------------------
void getRaycastParameter(float ray_org, float ray_dir, float cell_size, float *d, float *t)
{
	float org_grid = ray_org - grid_origin.x;
	float org_cell = org_grid / cell_size;

	if (ray_dir == 0)
	{
		*t = -1;
		*d = 0;
	}
	else if (ray_dir > 0)
	{
		*d = cell_size / ray_dir;
		*t = ((glm::floor(org_cell) + 1) * cell_size - org_grid) / ray_dir;
	}
	else
	{
		*d = -cell_size / ray_dir;
		*t = (glm::floor(org_cell) * cell_size - org_grid) / ray_dir;
	}
}

//-------------------------------------------------------------------------
//  Cast a ray through the finer grid
//  Parameters: -ray_direction: direction of the ray (has to be normalized!)
//				-ray_origin: point of intersection with the coarse grid
//
//  Return: - Color values of the intersected cell
//			- glm::vec3(-1,-1,-1) if there is no intersection
//
//  source: http://www.scratchapixel.com/lessons/advanced-rendering/introduction-acceleration-structure/grid
//-------------------------------------------------------------------------
glm::vec3 fineStep(glm::vec3 ray_origin, glm::vec3 ray_direction)
{
	float t, tx, ty, tz, dx, dy, dz;
	int x, y, z;

	getRaycastParameter(ray_origin.x, ray_direction.x, cell_size, &dx, &tx);
	getRaycastParameter(ray_origin.y, ray_direction.y, cell_size, &dy, &ty);
	getRaycastParameter(ray_origin.z, ray_direction.z, cell_size, &dz, &tz);

	// Coarse traversal step
	if (tx < 0)	tx = FLT_MAX;
	if (ty < 0) ty = FLT_MAX;
	if (tz < 0)	tz = FLT_MAX;

	x = (int)(ray_origin.x);
	y = (int)(ray_origin.y);
	z = (int)(ray_origin.z);
	
	if (x >= grid_size || y >= grid_size || z >= grid_size)
		return glm::vec3(-1, -1, -1);

	while (get_voxel(x, y, z) == 0)
	{
		if (tx < ty)
		{
			if (tx < tz)
			{
				x += sign(ray_direction.x);
				t = tx;
				tx += dx;
				if (x >= grid_size || x < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
		else
		{
			if (ty < tz)
			{
				y += sign(ray_direction.y);
				t = ty;
				ty += dy;
				if (y >= grid_size || y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}
	float r, g, b;
	float value = (float)y/(max_height - min_height);
	r = 1.0f;
	g = value;
	b = 0;
	return glm::vec3(r, g, b);
}

//-------------------------------------------------------------------------
//  Load point data - testing purposes only
//-------------------------------------------------------------------------
void loadPointData()
{
	std::ifstream file("../Data/data");
	std::string line, data;
	int x, y, z, coarse_x, coarse_y, coarse_z; 
	while (std::getline(file, line) && !line.empty())
	{
		int start = 0, end = line.find(" ");
		data = line.substr(start, end - start);
		x = (int)(stof(data) / cell_size);

		start = end + 1;
		end = line.find(" ", start + 1);
		data = line.substr(start, end - start);
		y = (int)(stof(data) / cell_size);

		start = end + 1; 
		data = line.substr(start, end - start);
		z = (int)(stof(data) / cell_size);

		coarse_x = (int)(x / coarse_cell_size);
		coarse_y = (int)(y / coarse_cell_size);
		coarse_z = (int)(z / coarse_cell_size);

		get_voxel(x, z, y) = 1;
		get_coarse_voxel(coarse_x, coarse_z, coarse_y) = 1;

		if (z > max_height) max_height = z;
		if (z < min_height) min_height = z;
	}
	file.close();
}

//-------------------------------------------------------------------------
//  This function sets the window x and y coordinates
//  such that the window becomes centered
//-------------------------------------------------------------------------
void centerOnScreen()
{
	window_x = (glutGet(GLUT_SCREEN_WIDTH) - window_width) / 2;
	window_y = (glutGet(GLUT_SCREEN_HEIGHT) - window_height) / 2;
}

//-------------------------------------------------------------------------
//  Clears any allocated memory
//-------------------------------------------------------------------------
void exit()
{
	delete(grid);
	delete(coarse_grid);
	delete(pixel_array);
}