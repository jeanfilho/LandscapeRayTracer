#include <gl/freeglut.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <fstream>
#include <string>
#include <iostream>

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
int window_height = 256;

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
glm::vec3 camera_position(0,0,0);
glm::vec3 camera_forward(1,0,0);
glm::vec3 camera_up(0,1,0);
float frame_distance = 10.0f;

// pixel array
GLfloat *pixel_array;
float frame_height = 50;
float frame_width = 100;

// macros
#define get_pixel(a, b) pixel_array[a * window_width + b]
#define get_voxel(x, y, z) grid[x * grid_size * grid_size + y * grid_size + z]
#define get_coarse_voxel(x, y, z) coarse_grid[x * coarse_grid_size * coarse_grid_size + y * coarse_grid_size + z]
#define sign(a) ((a > 0) - (a < 0))


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
	pixel_array = new GLfloat[window_height * window_width * 3]();
	loadPointData();
	
	std::cout << "Set up finished. Starting ray tracing..." << std::endl;
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//  OpenGL contents on the window.
//-------------------------------------------------------------------------
int frame_number = 0;
void display(void)
{
	//  Clear the window or more specifically the frame buffer...
	//  This happens by replacing all the contents of the frame
	//  buffer by the clear color (black in our case)
	glClear(GL_COLOR_BUFFER_BIT);

	//  Cast rays
	updatePixelBuffer();
	frame_number++;
	std::cout << "Frame " << frame_number << std::endl;

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
				get_pixel(x, y) = result.r;
				get_pixel(x, y + 1) = result.g;
				get_pixel(x, y + 2) = result.b;
			}
			else
			{
				get_pixel(x, y) = 1;
				get_pixel(x, y + 1) = 1;
				get_pixel(x, y + 2) = 0;
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
	if (tx == FLT_MAX) x = 0; else x = (int)(tx/coarse_factor);
	if (ty == FLT_MAX) y = 0; else y = (int)(ty/coarse_factor);
	if (tz == FLT_MAX) z = 0; else z = (int)(tz/coarse_factor);

	if (tx >= coarse_grid_size || ty >= coarse_grid_size || tz >= coarse_grid_size)
		return glm::vec3(-1, -1, -1);

	while (get_coarse_voxel(x, y, z) == 0)
	{
		if (tx < ty)
		{
			if (tx < tz)
			{
				x += sign(dx);
				t = tx;
				tx += dx;
				if (x >= coarse_grid_size || x < 0)
					return glm::vec3(-1,-1,-1);
			}
			else
			{
				z += sign(dz);
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
				y += sign(dy);
				t = ty;
				ty += dy;
				if (y >= coarse_grid_size || y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(dz);
				t = tz;
				tz += dz;
				if (z >= coarse_grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}

	return fineStep(ray_origin + glm::vec3(tx,ty,tz), ray_direction);
}

void getRaycastParameter(float ray_org, float ray_dir, float cell_size, float *d, float *t)
{
	float org_grid = ray_org - grid_origin.x;
	float org_cell = org_grid / cell_size;

	if (ray_dir == 0)
	{
		*t = FLT_MAX;
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
	if (tx == FLT_MAX) x = 0; else x = (int)(tx);
	if (ty == FLT_MAX) y = 0; else y = (int)(ty);
	if (tz == FLT_MAX) z = 0; else z = (int)(tz);

	while (get_voxel(x, y, z) == 0)
	{
		if (tx < ty)
		{
			if (tx < tz)
			{
				x += sign(dx);
				t = tx;
				tx += dx;
				if (x >= grid_size || x < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(dz);
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
				y += sign(dy);
				t = ty;
				ty += dy;
				if (y >= grid_size || y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += sign(dz);
				t = tz;
				tz += dz;
				if (z >= grid_size || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}
	return glm::vec3(.5f, .5f, 0);
}


//-------------------------------------------------------------------------
//  Load point data - testing purposes only
//-------------------------------------------------------------------------
void loadPointData()
{
	std::ifstream file("../Data/data");
	std::string line, data;
	while (std::getline(file, line) && !line.empty())
	{
		int start = 0, end = line.find(",");
		data = line.substr(start, end - start);
		int x = (int)(stof(data) / cell_size);

		start = end + 1;
		end = line.find(",", start + 1);
		data = line.substr(start, end - start);
		int y = (int)(stof(data) / cell_size);

		start = end + 1; 
		data = line.substr(start, end - start);
		int z = (int)(stof(data) / cell_size);
		x += 10;
		y += 10;
		get_voxel(x, z, y) = 1;
		get_coarse_voxel(x / coarse_grid_size, z / coarse_grid_size, y / coarse_grid_size) = 1;
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
	delete(pixel_array);
}