/*
 ============================================================================
 Author        : Jack Weber
 Description   : Project 3
 To build use  : make, ./main
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>
#include "projection.h"
#include "ImageWriter.h"
#include "Packed3DArray.h"

//Function used by the kernel to determine the begining of a sheet. (not always a "sheet"). (z axis)
__device__ size_t getCoord(size_t z, int nRows, int nCols, int nSheets, int pt){
	size_t x_coord = ( blockDim.x * blockIdx.x + threadIdx.x);
	size_t y_coord = ( blockDim.y * blockIdx.y + threadIdx.y);

	if(pt == 1){
		return z * nRows * nCols + x_coord * nCols + y_coord;
	}
	else if(pt == 2){
		x_coord = (nRows - 1) - x_coord;
		return z * nRows * nCols + x_coord * nCols + y_coord;
	}
	else if(pt == 3){
		return x_coord * nSheets * nCols + z * nCols + y_coord;
	}
	else if(pt == 4){
		x_coord = (nSheets - 1) - x_coord;
		return x_coord * nSheets * nCols + z * nCols + y_coord;
	}
	else if(pt == 5){
		return y_coord * nSheets * nCols + x_coord * nSheets + z;
	}
	else if(pt == 6){
		return ((nCols - 1) - y_coord) * nSheets * nCols + x_coord * nSheets + z;
	} else {
		return 0;
	}
}

//Kernel for maz image calculation. fils maxImage with the corresponding data and fils sums with data needed for sumImageCalc
__global__ void maxImageCalc (int nRows, int nCols, int nSheets, unsigned char* buffer, unsigned char* maxImage, float* sums, int pt)
{

	size_t x_coord = ( blockDim.x * blockIdx.x + threadIdx.x);
	size_t y_coord = ( blockDim.y * blockIdx.y + threadIdx.y);

	if(pt == 1 || pt == 2){
		int myID = y_coord * nRows + x_coord;
		maxImage[myID] = 0;
		sums[myID] = 0.0;
		for(int i = 0; i < nSheets; i++){
			if(buffer[getCoord(i, nRows, nCols, nSheets, pt)] > maxImage[myID])
				maxImage[myID] = buffer[getCoord(i, nRows, nCols, nSheets, pt)];
			sums[myID] += (buffer[getCoord(i, nRows, nCols, nSheets, pt)]) * (i+1)/nSheets;
		}
	}
	else if(pt == 3 || pt == 4){
		int myID = y_coord * nRows + x_coord;
		maxImage[myID] = 0;
		for(int i = 0; i < nCols; i++){
			if(buffer[getCoord(i, nRows, nCols, nSheets, pt)] > maxImage[myID])
				maxImage[myID] = buffer[getCoord(i, nRows, nCols, nSheets, pt)];
			sums[myID] += (buffer[getCoord(i, nRows, nCols, nSheets, pt)]) * (i+1)/nSheets;
		}
	}
	else if(pt == 5 || pt == 6){
		int myID = y_coord * nSheets + x_coord;
		maxImage[myID] = 0;
		for(int i = 0; i < nRows; i++){
			if(buffer[getCoord(i, nRows, nCols, nSheets, pt)] > maxImage[myID])
				maxImage[myID] = buffer[getCoord(i, nRows, nCols, nSheets, pt)];
			sums[myID] += (buffer[getCoord(i, nRows, nCols, nSheets, pt)]) * (i+1)/nSheets;
		}
	}
}

//Kernel that computes the sum image data from the data found in the maxImageCalc kernel
__global__ void sumImageCalc(float* sumInfo, float maximum, unsigned char* sumImage, int nRows){
	size_t x_coord = ( blockDim.x * blockIdx.x + threadIdx.x);
	size_t y_coord = ( blockDim.y * blockIdx.y + threadIdx.y);
	int myID = y_coord * nRows + x_coord;
	sumImage[myID] = round((sumInfo[myID]/maximum)*255.0);
}

//Entry point into the program
int main(int argc, char** argv)
{
	//Give arguments to the projection class
    if(!argv[6]){
        std::cout << "Not enough arguments\n";
    }
    Projection* projection = new Projection();
    projection->nRows = atoi(argv[1]);
    projection->nCols = atoi(argv[2]);
    projection->nSheets = atoi(argv[3]);
    projection->filename = argv[4];
    projection->pt = atoi(argv[5]);
    projection->output = argv[6];
    
    //Read in a file for projection class
    projection->readFile();
    char* hostBuffer = projection->stream;
    
	//Declare pointers to be used on device
    unsigned char *d_maxImage, *d_sumImage, *d_buffer;
	float *d_sumWorking;

	//Copy voxel data to the cpu.
	cudaMalloc((void**)&d_buffer, projection->size());
    cudaMemcpy(d_buffer, hostBuffer, projection->size(), cudaMemcpyHostToDevice);
	
	//Determine the size of the images
	cudaMalloc((void**)&d_maxImage, projection->imageSize() * sizeof(char));
	cudaMalloc((void**)&d_sumImage, projection->imageSize() * sizeof(char));
	cudaMalloc((void**)&d_sumWorking, projection->imageSize() * sizeof(float));


	//Create grid sizes, then launch kernel for maxImageCalc
	if(projection->pt == 1 || projection->pt == 2){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nCols + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nRows + threadsPerBlock.y - 1)/threadsPerBlock.y);
		maxImageCalc<<<blocksPerGrid, threadsPerBlock>>>(projection->nCols, projection->nRows, projection->nSheets, d_buffer, d_maxImage, d_sumWorking, projection->pt);
	}
	else if(projection->pt == 3 || projection->pt == 4){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nSheets + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nRows + threadsPerBlock.y - 1)/threadsPerBlock.y);
		maxImageCalc<<<blocksPerGrid, threadsPerBlock>>>(projection->nSheets, projection->nRows, projection->nCols, d_buffer, d_maxImage, d_sumWorking, projection->pt);
	}
	else if(projection->pt == 5 || projection->pt == 6){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nCols + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nSheets + threadsPerBlock.y - 1)/threadsPerBlock.y);
		maxImageCalc<<<blocksPerGrid, threadsPerBlock>>>(projection->nSheets, projection->nRows, projection->nCols, d_buffer, d_maxImage, d_sumWorking, projection->pt);
	}
	else {
		std::cout << "WRONG PT";
	}

	//Get the result
	unsigned char* h_maxImage;
	h_maxImage = new unsigned char[projection->imageSize()];
	cudaMemcpy(h_maxImage, d_maxImage, projection->imageSize() * sizeof(char), cudaMemcpyDeviceToHost);
	const unsigned char* test = h_maxImage;

	float* h_sums = new float[projection->imageSize()];
	cudaMemcpy(h_sums, d_sumWorking, projection->imageSize() * sizeof(float), cudaMemcpyDeviceToHost);

	//Calculate the maxs
	float h_sum_maximum = 0.0;
	for(int i = 0; i < projection->imageSize(); i++){
		if(h_sum_maximum < h_sums[i])
			h_sum_maximum = h_sums[i];
	}

	//Create grid sizes, launch the next kernel (sumImageCalc)
	if(projection->pt == 1 || projection->pt == 2){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nCols + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nRows + threadsPerBlock.y - 1)/threadsPerBlock.y);
		sumImageCalc<<<blocksPerGrid, threadsPerBlock>>>(d_sumWorking, h_sum_maximum, d_sumImage, projection->nRows);
	}
	else if(projection->pt == 3 || projection->pt == 4){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nSheets + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nRows + threadsPerBlock.y - 1)/threadsPerBlock.y);
		sumImageCalc<<<blocksPerGrid, threadsPerBlock>>>(d_sumWorking, h_sum_maximum, d_sumImage, projection->nRows);
	}
	else if(projection->pt == 5 || projection->pt == 6){
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(
			(projection->nCols + threadsPerBlock.x - 1)/threadsPerBlock.x,
			(projection->nSheets + threadsPerBlock.y - 1)/threadsPerBlock.y);
		sumImageCalc<<<blocksPerGrid, threadsPerBlock>>>(d_sumWorking, h_sum_maximum, d_sumImage, projection->nSheets);
	}
	else {
		std::cout << "WRONG PT";
	}

	//Get the results of the sumImageCalc kernel
	unsigned char* h_sumImage = new unsigned char[projection->imageSize()];
	cudaMemcpy(h_sumImage, d_sumImage, projection->imageSize() * sizeof(char), cudaMemcpyDeviceToHost);
	const unsigned char* f_h_sumImage = h_sumImage;

	//Free the memory on device.
	cudaFree(d_maxImage);
	cudaFree(d_sumImage);
	cudaFree(d_buffer);
	cudaFree(d_sumWorking);
	cudaThreadSynchronize();


	//Print the images.
	std::string maxName = argv[6];
	std::string sumName = argv[6];
	maxName.append("MAX.png");
	sumName.append("SUM.png");

	if(projection->pt == 1 || projection->pt == 2){
		projection->writeTheFile(maxName, projection->nCols, projection->nRows, test);
		projection->writeTheFile(sumName, projection->nCols, projection->nRows, f_h_sumImage);
	}
	else if(projection->pt == 3 || projection->pt == 4){
		projection->writeTheFile(maxName, projection->nSheets, projection->nRows, test);
		projection->writeTheFile(sumName, projection->nSheets, projection->nRows, f_h_sumImage);
	}
	else if(projection->pt == 5 || projection->pt == 6){
		projection->writeTheFile(maxName, projection->nCols, projection->nSheets, test);
		projection->writeTheFile(sumName, projection->nCols, projection->nSheets, f_h_sumImage);
	}
	else {
		std::cout << "WRONG PT 2\n";
	}
	

	return 0;
}

