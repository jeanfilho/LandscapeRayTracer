/*-------------------------------------------------------------------
Coordinate system: Left-handed, +X(right) cross +Z(forward) = +Y (up)
---------------------------------------------------------------------*/
#include <fstream>
#include <string>
#include <iostream>
#include <utility>
#include <chrono>

#include <gl/freeglut.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"

#include "Grid.h"
#include "PointData.h"
#include "Camera.h"
#include "CudaWorker.h"

void init();
void display(void);
void centerOnScreen();
void updatePixelBuffer();
void exit();


//  define the window position on screen
int window_x;
int window_y;

//  variables representing the window size
int window_width = 512;
int window_height = 512;

//  variable representing the window title
char *window_title = "Landscape Raytracer";


Camera cam(glm::vec3(128, 400, 128), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), 5, 256, 256);
glm::vec3 *pixel_array;

// macros
#define get_pixel(x, y) pixel_array[y * window_width + x]

// visualization parameters
float max_height = FLT_MIN;
float min_height = FLT_MAX;

// clock
std::chrono::system_clock sys_clock;
std::chrono::time_point<std::chrono::system_clock> last_frame, current_frame;
std::chrono::duration<float> delta_time;


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

	//initialize point vector and pixel array
	pixel_array = new glm::vec3[window_height * window_width]{ glm::vec3(0,0,0) };
	CudaWorker::loadPoints(&max_height, &min_height, window_height, window_width, cam);

	current_frame = last_frame = sys_clock.now();

	std::cout << "Set up finished. Starting ray tracing..." << std::endl;
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//  OpenGL contents on the window.
//-------------------------------------------------------------------------
int frame_number = 0;
void display(void)
{
	current_frame = sys_clock.now();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;

	frame_number++;
	std::cout << "Frame " << frame_number;
	std::cout << " - FPS " << 1/delta_time.count() << std::endl;

	//  Clear the window or more specifically the frame buffer...
	//  This happens by replacing all the contents of the frame
	//  buffer by the clear color (black in our case)
	glClear(GL_COLOR_BUFFER_BIT);

	//  Cast rays
	updatePixelBuffer();

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
	for (int y = 0; y < window_height; y++)
		for (int x = 0; x < window_width; x++)
		{
			glm::vec3 ray_origin, ray_direction;
			glm::ivec3 cell(0, 0, 0);
			bool inGrid = true, found = false;
			Grid<PointData*> * subgrid;

			ray_direction = cam.forward * cam.frame_distance
				+ cam.up * ((y - window_height / 2) / cam.frame_height)
				+ cam.right * ((x - window_width / 2) / cam.frame_width);
			ray_origin = cam.position + ray_direction;

			while (inGrid && !found)
			{
				cell = grid.castRay(ray_origin, ray_direction, &ray_origin);
				if (cell.x < 0) {
					get_pixel(x, y) = glm::vec3(0, 0, 0);
					inGrid = false;
				}
				else {
					subgrid = grid(cell);
					cell = subgrid->castRay(ray_origin, ray_direction, &ray_origin);
					if (cell.x < 0)
						get_pixel(x, y) = glm::vec3(0, 0, 0);
					else
					{
						get_pixel(x, y) = (*subgrid)(cell)->color;
						found = true;
					}
				}
			}
		}
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

	CudaWorker::exitRoutine();
	delete(pixel_array);
}