/*
 * author: Austin Dress
 * purpose: Intro to gpu programming class. Program generates voronoi diagrams on the gpu and cpu
 * controls:
 *
 * g - pressing g switches between gpu mode and cpu mode
 * v - switches between original image and voronoi diagram
 * s - writes out voronoi diagram as voronoi.jpg
 * c - clears all sites
 * t - runs a performance test on gpu and cpu
 * r - generates random points equal to NUM_ELEMENTS in voronoi.h
 */


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "voronoi.h"


using namespace cv;
using namespace std;

// Command line helper function
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

// Command line helper function
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

Vec3b generate_color(){
	static int seed = 0;
	RNG rng(seed);
	seed += 1;
	Vec3b color;
	color.val[0] = rng.uniform(0,255);
	color.val[1] = rng.uniform(0,255);
	color.val[2] = rng.uniform(0,255);
	return color;
}

void generate_xy_points(COORDS_T* sites, int num_sites, int height, int width) {
	// function for generating random sites equal to num_sites
	//initialize random generator - uses current time as seed
	srand((unsigned)time(0));

	// Clear any user generated points prior to generating random ones
	clear_sites(sites);

	if (num_sites > NUM_ELEMENTS) {
		num_sites = NUM_ELEMENTS;
	}

	for(int i = 0; i < num_sites; i++)
	{
		int x = rand() % width;
		int y = rand() % height;
		sites->x[i] = x;//rand() % width;
		sites->y[i] = y;//rand() % height;
		sites->next += 1;
	}
}

void clear_sites(COORDS_T* ptr){
	for (auto i = 0; i < ptr->next; ++i) {
		ptr->x[i] = 0;
		ptr->y[i] = 0;
		ptr->next = 0;
	}
}
void voronoi_cpu(cv::Mat& src, cv::Mat& dst, COORDS_T* sites_ptr) {
	unsigned char *input = (unsigned char*)(src.data);

	// loop through each pixel in the image and calculate closest site
	// update dst pixel with the closest site color
	for(int y = 0; y < src.rows; y++){
	    for(int x = 0; x < src.cols; x++){
			auto closest_site = 0;
    		auto closest = 1.0e30f;

    		for (auto index = 0; index < sites_ptr->next; ++index) {

	    		float distance = sqrt( (x - sites_ptr->x[index])*(x - sites_ptr->x[index]) +
	    						  (y - sites_ptr->y[index])*(y - sites_ptr->y[index]) );
	    		if (distance < closest) {
	    			closest_site = index;
	    			closest = distance;
	    		}
	    	}

	    	// Get the current pixel
	    	Vec3b color = dst.at<Vec3b>(Point(x,y));

    		// Change pixel color
    		color = src.at<Vec3b>(Point2f(sites_ptr->x[closest_site],sites_ptr->y[closest_site]));

			// Set pixel
    		dst.at<Vec3b>(Point(x,y)) = color;
	    }
	}
}

// Mouse call back function used for adding sites to sites struct
void onMouse(int evt, int x, int y, int flags, void* param) {
    if(evt == CV_EVENT_LBUTTONDOWN) {
    	COORDS_T *ptr = (COORDS_T *)param;
    	if (ptr->next < NUM_ELEMENTS){
    		ptr->x[ptr->next] = x;
    		ptr->y[ptr->next] = y;
    		ptr->next += 1;
    	}
    }
}

void help_print() {
	cout << "To run the application you must provide an input image." << endl;
	cout << "-i <image path>" << endl;
	cout << "-n <num_sites>" << endl;
	cout << "Example:" << endl;
	cout << "./Voronoi -i ../images/crab.jpg -n 3000" << endl;

	cout << "" << endl;
	cout << "Runtime Instructions: " << endl;
	cout << "g - pressing g switches between gpu mode and cpu mode" << endl;
	cout << "v - switches between original image and voronoi diagram" << endl;
	cout <<  "s - writes out voronoi diagram as voronoi.jpg" << endl;
	cout <<  "c - clears all sites" << endl;
	cout <<  "t - runs a performance test on gpu and cpu" << endl;
	cout <<  "r - generates random points equal to NUM_ELEMENTS in voronoi.h" << endl;
}

int main(int argc, char *argv[]) {
	COORDS_T *sites_ptr;
	COORDS_T sites_struct;
	sites_ptr = &sites_struct;
	sites_ptr->next = 0;

	std::string imagePath;
	int thd_size = 256;
	int num_sites = 10;
	bool show_voronoi = false;
	bool gpu_mode = false;

	float total_time = 0.0;
	float diff, seconds;
	clock_t t1,t2;
	char k;

	namedWindow("Display window",0);

	if(cmdOptionExists(argv, argv+argc, "-i"))
	{
		imagePath = getCmdOption(argv, argv + argc, "-i");
	}

	//Read input image from the disk
	Mat org = imread(imagePath, CV_LOAD_IMAGE_COLOR);
	if(org.empty())
	{
		cout << "Failed to open the file at: " << imagePath << endl;
		help_print();
		return -1;
	}

	// Create a copy of image for output and for clearing
	Mat vor = org.clone();
	Mat org_clone = org.clone();

	// Instantiate voronoi class
	Voronoi myvor(org_clone);
	Voronoi *myvor_ptr;
	myvor_ptr = &myvor;


	// Callback function for user using mouse
	setMouseCallback("Display window", onMouse, (void*)sites_ptr);

	if(cmdOptionExists(argv, argv+argc, "-n"))
	{
		num_sites = atoi(getCmdOption(argv, argv + argc, "-n"));
		generate_xy_points(sites_ptr, num_sites, vor.rows, vor.cols);
	}

	if(cmdOptionExists(argv, argv+argc, "--help"))
	{
		help_print();
		exit(1);
	}

	// Main loop for refreshing display
	while (1)
	{
		// Display voronoi mode if user presses 'v'
		if (show_voronoi) {
			if (sites_ptr->next > 0){
				myvor.update(sites_ptr);
				clock_t t1,t2;
				t1=clock();
				// User can swtich between gpu and cpu mode
				if (!gpu_mode){
					//voronoi_gpu(org_clone, vor, sites_ptr);
					voronoi_cpu(org_clone, vor, sites_ptr);
				} else {
					myvor.voronoi_gpu(vor);
					//voronoi_gpu(org_clone, vor, sites_ptr);
				}
				t2=clock();
				diff = ((float)t2-(float)t1);
				seconds = diff / CLOCKS_PER_SEC;
				cout << sites_ptr->next << " Sites" << " at " << 1/seconds << " " << "Frames Per Second" << endl;
				imshow("Display window", vor);
			}
			else {
				imshow("Display window", org);
			}
		} else {
			// If we are not displaying voronoi diagram, show the original image
			if (sites_ptr->next > 0){
				for (int i = 0; i < sites_ptr->next; i++){
					circle(org,Point(sites_ptr->x[i],sites_ptr->y[i]), 1, Scalar(0, 255, 0), 2, 8, 0);
				}
			}
			imshow("Display window", org);
		}

		//Wait for key press, follow section handles keys that can be pressed
		k = cvWaitKey(10);

		if ((k % 0x100) == 27)
			break;
		else if (k == 'v') {
			// switch between voronoi and site view
			if (show_voronoi) show_voronoi = false;
			else show_voronoi = true;
		}
		else if (k == 'c') {
			// clear sites
			clear_sites(sites_ptr);
			org = org_clone.clone();
			vor = org_clone.clone();
		} else if(k == 'g') {
			// switch to gpu
			if (gpu_mode) gpu_mode = false;
			else gpu_mode = true;
		} else if(k == 'r') {
			// generate random points equal to NUMELEMENTS
			generate_xy_points(sites_ptr, NUM_ELEMENTS, vor.rows, vor.cols);
		} else if(k == 's') {
			cout << "Saving file to disk." << endl;
			imwrite("voronoi.jpg", vor);
		} else if(k == 't') {
			cout << "Generating performance data..." << endl;

			ofstream performance;
			std::stringstream ss;
			performance.open("performance2.txt", ios::out | ios::app);

			cout << "Generating profile for cpu" << endl;
			for (int i = 2; i <= 8192; i*=2){
				generate_xy_points(sites_ptr, i, vor.rows, vor.cols);
				myvor.update(sites_ptr);
				t1=clock();

				voronoi_cpu(org_clone, vor, sites_ptr);

				t2=clock();
				diff = ((float)t2-(float)t1);
				seconds = diff / CLOCKS_PER_SEC;

				ss.str(std::string());
				ss << "1 " << i  << " " << seconds << "\n";
				performance << ss.rdbuf();
				ss.str(std::string());
			}
			cout << "Generating profile for gpu" << endl;
			for (int i = 2; i <= 8192; i*=2){
				generate_xy_points(sites_ptr, i, vor.rows, vor.cols);
				myvor.update(sites_ptr);
				t1=clock();

				myvor.voronoi_gpu(vor);

				t2=clock();
				diff = ((float)t2-(float)t1);
				seconds = diff / CLOCKS_PER_SEC;

				ss.str(std::string());
				ss << "2 " << i  << " " << seconds << "\n";
				performance << ss.rdbuf();
				ss.str(std::string());
			}
			performance.close();
		}
	}

	// Cleanup class as program is closed
	myvor.clean();
	return 0;
}
