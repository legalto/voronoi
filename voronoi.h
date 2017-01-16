/*
 * voronoi.h
 *
 *  Created on: Sep 30, 2015
 *      Author: dressag1
 */

#ifndef VORONOI_H_
#define VORONOI_H_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define NUM_ELEMENTS 10000
typedef int COORD_T[NUM_ELEMENTS];

// Struct container for holding all of the sites. Number of sites is limited to NUM_ELEMENTS
typedef struct {
	COORD_T x;
	COORD_T y;
	int next;
} COORDS_T;

void voronoi_gpu(const cv::Mat& input, cv::Mat& output, COORDS_T *sites);
void clear_sites(COORDS_T* ptr);

class Voronoi {
public:
	// create device pointers
	unsigned char  *d_input, *d_output;
	COORDS_T *d_sites;

	Voronoi(const cv::Mat& input);
	~Voronoi();

	void voronoi_gpu(cv::Mat& output);
	void update(COORDS_T *sites);
	void clean();
};

#endif /* VORONOI_H_ */
