#include <gl/freeglut.h>
#include <glm/common.hpp>


#define get_pixel(m, a, b) m[a * window_width + b]

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
int grid_size = 1000000;
char *grid;
int coarse_grid_size = 1000;
char *coarse_grid;

// camera variables
glm::vec3 camera_position;
glm::vec3 camera_forward;
float frame_distance = 10.0f;

// pixel array
GLfloat *pixel_array;

//-------------------------------------------------------------------------
//  Program Main method.
//-------------------------------------------------------------------------
void main(int argc, char **argv)
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
}

//-------------------------------------------------------------------------
//  Set OpenGL program initial state.
//-------------------------------------------------------------------------
void init()
{
	//  Set the frame buffer clear color to black. 
	glClearColor(0.0, 0.0, 0.0, 0.0);

	grid = new char[grid_size * grid_size * grid_size];
	coarse_grid = new char[coarse_grid_size * coarse_grid_size * coarse_grid_size];
	pixel_array = new GLfloat[window_height * window_width * 3];

	//camera initial parameters;
	camera_position = glm::vec3(-20, 10, -20);
	camera_forward = -camera_position;

	
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
			get_pixel(pixel_array, x, y) = result.x;
			get_pixel(pixel_array, x, y + 1) = result.y;
			get_pixel(pixel_array, x, y + 2) = result.z;
		}
}


//-------------------------------------------------------------------------
//  Cast a ray through the grid
//-------------------------------------------------------------------------
glm::vec3 castRay(int x, int y)
{

}

//-------------------------------------------------------------------------
//  Load point data - testing
//-------------------------------------------------------------------------
void loadPointData()
{

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