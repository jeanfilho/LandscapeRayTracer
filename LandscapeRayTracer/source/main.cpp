#include <gl/freeglut.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <utility>

#include "Grid.h"
#include "PointData.h"
#include "Camera.h"

void init();
void display(void);
void centerOnScreen();
void updatePixelBuffer();
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

Grid<Grid<PointData*>*> coarse_grid(glm::vec3(0, 0, 0), 1000, 10000);
Grid<PointData*> grid(glm::vec3(0, 0, 0), 1000, 1000);

Camera cam(glm::vec3(128, 700, 128), glm::vec3(0, -1, 0), glm::vec3(1, 0, 0), 5, 256, 256);
glm::vec3 *pixel_array;

// macros
#define get_pixel(x, y) pixel_array[y * window_width + x]

// visualization parameters
int max_height = INT_MIN;
int min_height = INT_MAX;


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
	pixel_array = new glm::vec3[window_height * window_width]{glm::vec3(0,0,0)};
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
			glm::vec3 ray_origin, ray_direction;

			ray_direction = cam.forward * cam.frame_distance
				+ cam.up * (float)(y - window_height / 2) / cam.frame_height
				+ -cam.right * (float)(x - window_width / 2) / cam.frame_width;
			ray_origin = cam.position + ray_direction;

			glm::ivec3 cell = grid.castRay(ray_origin, ray_direction);
			Grid<bool> *subgrid = grid(cell);
			cell = subgrid->castRay(ray_origin, ray_direction);

			if((*subgrid)(cell))
			{
				get_pixel(x, y).r = 1.0f;
				get_pixel(x, y).g = 1.0f + cell.z/(max_height - min_height);
				get_pixel(x, y).b = 0;
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

		coarse_x = (int)(x / grid.cell_size);
		coarse_y = (int)(y / grid.cell_size);
		coarse_z = (int)(z / grid.cell_size);

		*grid(x, y, z) = PointData(glm::vec3(0, 0, 0));
		*coarse_grid(coarse_x, coarse_y, coarse_z) = std::pair<bool,PointData>(true, grid(x, y, z)));

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