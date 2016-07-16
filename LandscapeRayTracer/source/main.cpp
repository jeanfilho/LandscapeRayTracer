#include <gl/freeglut.h>
#include <glm/common.hpp>
#include <fstream>
#include <string>

void init();
void display(void);
void centerOnScreen();
void updatePixelBuffer();
glm::vec3 castRay(int x, int y);
void exit();
void loadPointData();

//  define the window position on screen
int window_x;
int window_y;

//  variables representing the window size
int window_width = 1024;
int window_height = 768;

//  variable representing the window title
char *window_title = "Landscape Raytracer";

// grid
int grid_size = 1000;
char *grid;
int coarse_grid_size = 10;
char *coarse_grid;

// camera variables
glm::vec3 camera_position;
glm::vec3 camera_forward;
float frame_distance = 10.0f;

// pixel array
GLfloat *pixel_array;

// macros
#define get_pixel(a, b) pixel_array[a * window_width + b]
#define get_voxel(x, y, z) grid[x * grid_size * grid_size + y * grid_size + z]
#define get_coarse_voxel(x, y, z) coarse_grid[x * coarse_grid_size * coarse_grid_size + y * coarse_grid_size + z]


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
	//  Set the frame buffer clear color to black. 
	glClearColor(0.0, 0.0, 0.0, 0.0);

	//initialize grids
	grid = new char[grid_size * grid_size * grid_size];
	coarse_grid = new char[coarse_grid_size * coarse_grid_size * coarse_grid_size];
	pixel_array = new GLfloat[window_height * window_width * 3];

	//camera initial parameters;
	camera_position = glm::vec3(-20, 10, -20);
	camera_forward = -camera_position;

	loadPointData();

	
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
			get_pixel(x, y) = result.x;
			get_pixel(x, y + 1) = result.y;
			get_pixel(x, y + 2) = result.z;
		}
}


//-------------------------------------------------------------------------
//  Cast a ray through the grid
//-------------------------------------------------------------------------
glm::vec3 castRay(int x, int y)
{
	glm::vec3 result;

	return result;
}

//-------------------------------------------------------------------------
//  Load point data - testing
//-------------------------------------------------------------------------
void loadPointData()
{
	std::ifstream file("../Data/data");
	std::string line, data;
	while (std::getline(file, line) && !line.empty())
	{
		int start = 0, end = line.find(",");
		data = line.substr(start, end - start);
		int x = (int)stof(data);

		start = end + 1;
		end = line.find(",", start + 1);
		data = line.substr(start, end - start);
		int y = (int)stof(data);

		start = end + 1; 
		data = line.substr(start, end - start);
		int z = (int)stof(data);

		get_voxel(x, y, z) = 1;
		get_coarse_voxel(x / coarse_grid_size, y / coarse_grid_size, z / coarse_grid_size) = 1;
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