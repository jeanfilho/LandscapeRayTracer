#include <gl/freeglut.h>
#include <glm/glm.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

#include "Camera.h"
#include "Heightmap.h"

void init();
void display(void);
void centerOnScreen();
void updatePixelBuffer();
void exit();
void loadPointData();


//  define the window position on screen
int window_x;
int window_y;

//  variable representing the window title
char *window_title = "Landscape Raytracer";

//  variables for fps counter
std::chrono::system_clock sys_clock;
std::chrono::time_point<std::chrono::system_clock> last_frame, current_frame;
std::chrono::duration<float> delta_time;

//  pixel buffer
glm::vec3 *pixel_array;

//  camera
Camera camera(1024, 512, glm::vec3(0.5f, -0.5f, 0), glm::vec3(0.5, 0.5, 0), glm::vec3(0, 20, 512), 5);

//  grid
Heightmap heightmap(1024, 3, glm::vec3(0, 0, 0), 1.0f);


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
	glutInitWindowSize(camera.window_width, camera.window_height);
	glutInitWindowPosition(window_x, window_y);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow(window_title);

	//  Set OpenGL program initial state.
	init();

	// Set the callback functions
	glutIdleFunc(display);
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

	//  Load Points
	loadPointData();

	//  initialize pixel array
	pixel_array = new glm::vec3[camera.window_height*camera.window_width]();
	
	current_frame = last_frame = sys_clock.now();

	std::cout << "Set up finished. Starting ray tracing..." << std::endl;
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//  OpenGL contents on the window.
//-------------------------------------------------------------------------
void display(void)
{
	// FPS Counter
	current_frame = sys_clock.now();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	std::cout << "FPS " << 1 / delta_time.count() << std::endl;

	//  Clear the window or more specifically the frame buffer...
	//  This happens by replacing all the contents of the frame
	//  buffer by the clear color (black in our case)
	glClear(GL_COLOR_BUFFER_BIT);

	//  Cast rays
	updatePixelBuffer();
	
	//  Draw Pixels
	glDrawPixels(camera.window_width, camera.window_height, GL_RGB, GL_FLOAT, pixel_array);

	//  Swap contents of backward and forward frame buffers
	glutSwapBuffers();
}

//-------------------------------------------------------------------------
//  Update the pixel buffer - cast a ray for each pixel
//-------------------------------------------------------------------------
void updatePixelBuffer()
{
	for (int y = 0; y < camera.window_height; y++)
		for (int x = 0; x < camera.window_width; x++)
		{
			glm::vec3 ray_origin, ray_direction;
			ray_direction = camera.forward * camera.frame_distance
				+ camera.up * ((float)y - camera.window_height / 2)
				+ camera.right * ((float)x - camera.window_width / 2);
			ray_origin = camera.position + ray_direction;

			pixel_array[y * camera.window_height + x] = heightmap.trace_ray(ray_origin, ray_direction);
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
		// TODO: read through file
	}
	file.close();
}

//-------------------------------------------------------------------------
//  This function sets the window x and y coordinates
//  such that the window becomes centered
//-------------------------------------------------------------------------
void centerOnScreen()
{
	window_x = (glutGet(GLUT_SCREEN_WIDTH) - camera.window_width) / 2;
	window_y = (glutGet(GLUT_SCREEN_HEIGHT) - camera.window_height) / 2;
}

//-------------------------------------------------------------------------
//  Clears any allocated memory
//-------------------------------------------------------------------------
void exit()
{
	delete(pixel_array);
}