//#include "jbutil.h"
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <math.h>
#include <random>

//#include "vec2.h"
#include "vec2f.h"
#include "jbutil.h"
#include "nbodyConfig.h"

#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw std::runtime_error( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )

//#define CHUNK_SIZE 8
#define MAX_RUNS 1
//#define GRAV_CONSTANT = 6.67408e-11f;

/*
 * Definitions for field dimensions
 */
int fieldWidth;
int doubleFieldWidth;// = fieldWidth << 1;
//const int fieldHalfWidth = fieldWidth >> 1;
int fieldHeight;// = 2000;
//const int fieldHalfHeight = fieldHeight >> 1;
int doubleFieldHeight;// = fieldHeight << 1;

__constant__ __device__ float GRAV_CONSTANT = 6.67408e-11f;

/*
 * Particle structure
 */
struct __align__(64) Particle {
	Vec2f Position;
	Vec2f Velocity;
	float Mass;
	float Radius;

	CUDA_CALLABLE_MEMBER Particle(void) :
			Position(Vec2f(0, 0)), Velocity(0.f, 0.f), Mass(0.f) {
	}

	CUDA_CALLABLE_MEMBER Particle(Vec2f p, double m) :
			Position(p), Velocity(0.0, 0.0), Mass(m) {
	}

	CUDA_CALLABLE_MEMBER Particle(Vec2f p, Vec2f v, float m, float radius) :
			Position(p), Velocity(v), Mass(m), Radius(radius) {
	}

	CUDA_CALLABLE_MEMBER bool intersect(const Particle& p) {
		Vec2f direction = p.Position - Position;
		float distance = (direction.X * direction.X)
				+ (direction.Y * direction.Y);
		return distance < (Radius + p.Radius) * (Radius + p.Radius);
	}
};

typedef struct rgb_data {
	int r;
	int g;
	int b;
} RGB;

//Device global variables to prevent copying to host every iteration
__device__ Particle* deviceBodies;
__device__ float* deviceUpdatedMasses;
__device__ int deviceNumBodies;

/*
 * Compute forces of particles exerted on one another
 */
__global__ void ComputeForces(Particle* d_bodies, float* updatedMasses, int size, float timestep, int fieldWidth, int fieldHeight)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
//	if(j == 0) {
//		//printf("Entered\n");
//		//printf("Size: %d\n", deviceNumBodies);
//	}

	Particle* bodies = d_bodies;//deviceBodies;
	float* updated_masses = updatedMasses;//deviceUpdatedMasses;//*updated_massesAddr;
	int numBodies = size;//*numBodiesAddr;
	Vec2f direction, force, acceleration;
	float distance;
	Particle &p1 = bodies[j];
	force = 0.f, acceleration = 0.f;
	for (size_t k = 0; k < numBodies; ++k) {
		if (k == j)
			continue;

		Particle &p2 = bodies[k];
		if (p2.Mass == 0.f)
			continue; //check if deleted

		bool intersect = p1.intersect(p2);

		//Testing for body - body collision
		if ((p1.Mass >= p2.Mass) && intersect) {
			updated_masses[j] = p1.Mass + p2.Mass;
			p1.Radius += p2.Radius;
			continue;
		} else if ((p1.Mass < p2.Mass) && intersect) {
			updated_masses[j] = 0.f;
			continue;
		}

		// Compute direction vector
		direction = p2.Position - p1.Position;
		distance = direction.length();

#ifndef NDEBUG
		assert(distance != 0);
#endif
		// Accumulate force
		force += (direction * p2.Mass) / (distance * distance * distance);
		/*printf("Mass (p%d) : %.5f\n", (int)j, p1.Mass);
		 printf("Mass (p%d) : %.5f\n", (int)k, p2.Mass);
		 printf("Direction (p%d to p%d) : %.2f, %.2f\n", (int)j, (int)k,direction.X, direction.Y);
		 printf("Distance  (p%d to p%d): %.4f\n", (int)j, (int)k, distance);
		 printf("Force (p%d)  (running sum): %.2f, %.2f\n", (int)j, force.X, force.Y);*/

	};

	// Compute acceleration for body
	acceleration = force * GRAV_CONSTANT;
	p1.Velocity += acceleration * timestep;

	//Border collision
	if (p1.Position.X + (acceleration.X * timestep) > fieldWidth - p1.Radius
			|| p1.Position.X + (acceleration.X * timestep)
					< -fieldWidth + p1.Radius)
		p1.Velocity.X *= -1;
	if (p1.Position.Y + (acceleration.Y * timestep) > fieldHeight - p1.Radius
			|| p1.Position.Y + (acceleration.Y * timestep)
					< -fieldHeight + p1.Radius)
		p1.Velocity.Y *= -1;
}

/*
 * Update particle positions
 */

__global__ void MoveBodies(Particle* bodies, float* updatedMasses, float p_deltaT)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;//threadIdx.x;
	Particle* p_bodies = bodies;//deviceBodies;//*bodiesAddr;
	float* updated_masses = updatedMasses;//deviceUpdatedMasses;
	//p_bodies[j].Mass = updated_masses[j];
	if (updated_masses[j] != 0.f) {
		//printf("UPDATED MASS (p%d): %.2f\n", (int)j, updated_masses[j]);
		p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
		//updatedBodies.push_back(p_bodies.at(j));
	}
}

__global__ void generateImage(Particle* d_bodies, char* imgData, int width,
		int height, int fieldWidth, int fieldHeight)
{
	Particle* bodies = d_bodies;//deviceBodies;
	const int img_width = width;
	const int img_height = height;
	const int doubleFieldWidth = fieldWidth << 1;
	const int doubleFieldHeight = fieldHeight << 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;//threadIdx.x;

	//Since positions can be negative, we add the field width and height to determine the new position in the image ( no negative coords)
	int x_centre_pixel = (int) (((bodies[i].Position.X + fieldWidth) / doubleFieldWidth) * img_width);
	int y_centre_pixel = (int) (((bodies[i].Position.Y + fieldHeight) / doubleFieldHeight) * img_height);

	int x_sq;
	int y_sq;
	//int x, y;
	int y_min = y_centre_pixel - bodies[i].Radius < 0 ?	0 : y_centre_pixel - bodies[i].Radius;
	int y_max = y_centre_pixel + bodies[i].Radius >= img_height ? img_height : y_centre_pixel + bodies[i].Radius;
	int x_min =	x_centre_pixel - bodies[i].Radius < 0 ? 0 : x_centre_pixel - bodies[i].Radius;
	int x_max = x_centre_pixel + bodies[i].Radius > img_width ?	img_width : x_centre_pixel + bodies[i].Radius;

	//int N = 2 * bodies[i].Radius + 1;

	for (int y = y_min; y < y_max; ++y)
	{
		if (y < 0) y = 0;
		if (y > img_height) y = img_height;
		for (int x = x_min; x < x_max; ++x)
		{
			if (x < 0) x = 0;
			if (x > img_width) x = img_width;
			x_sq = (x - x_centre_pixel) * (x - x_centre_pixel);
			y_sq = (y - y_centre_pixel) * (y - y_centre_pixel);
			if (x_sq + y_sq <= (int)(bodies[i].Radius * bodies[i].Radius) /*roundf(bodies[i].Radius * bodies[i].Radius)*/)
			{
#ifndef NDEBUG
				assert((x >= 0) && (x <= img_width));
				assert((y >= 0) && (y <= img_height));
#endif
				//img_data(y, x) = 0;
				imgData[(img_width * y) + x] = (char) 0;//static_cast<char>(0);
			};
		}
	}

	//Brute forcing every pixel and checking whether it is in body's radius.
//	for (int k = 0; i < N; ++k)
//	{
//		for (int h = 0; h < N; ++h)
//		{
//			x = x_centre_pixel - (k - bodies[i].Radius);
//			y = y_centre_pixel - (h - bodies[i].Radius);
//			if (y < 0) y = 0;
//			if (y > img_height) y = img_height;
//			if (x < 0) x = 0;
//			if (x > img_width) x = img_width;
//			//x_sq = (x - x_centre_pixel) * (x - x_centre_pixel);
//			//y_sq = (y - y_centre_pixel) * (y - y_centre_pixel);
//			if ( (x*x + y*y) <= (int)(bodies[i].Radius * bodies[i].Radius) + 1)
//			{
//#ifndef NDEBUG
//				assert((x >= 0) && (x <= img_width));
//				assert((y >= 0) && (y <= img_height));
//#endif
//				//img_data(y, x) = 0;
//				imgData[(img_width * y) + x] = (char) 0;//static_cast<char>(0);
//			};
//		}
//	}

}

void saveImageToDisk(const std::string &filename, char* imgData, int imgWidth,
		int imgHeight) {
	size_t imageSize = imgWidth * imgHeight;
	std::ofstream outImg;

	outImg.open(filename, std::ofstream::out);
	std::cout << "Saving (" << imgWidth << "x" << imgHeight << ") to disk"
			<< std::endl;
	if (outImg.is_open()) {
		outImg << "P5\n" << imgWidth << " " << imgHeight << "\n255\n";
		for (int i = 0; i < imageSize; ++i) {
			//printf("%d\n", imgData[i]);
			outImg << imgData[i];
		}
		outImg.close();
	}
	else
	{
		std::cerr << "Error writing image to file:" << filename << std::endl
				<< "Ensure the the folder exists" << std::endl;
		exit(1);
	}
}

void SaveIterationImage(const std::string &filename,
		const std::vector<Particle> &bodies) {
	const int img_width = 1024;
	const int img_height = 900;

	//jbutil::matrix<int> img_data;
	//img_data.resize(img_height, img_width);

	size_t imageSize = img_width * img_height;
	//int* imgData = new int[imageSize];
	char* imgData = new char[imageSize];

	//Initializing matrix
	// for(int j = 0; j < img_width; ++j)
	// 	for(int i = 0; i < img_height; ++i)
	// 		img_data(i, j) = 254;
	//printf("hey\n");

	for (int i = 0; i < imageSize; ++i) {
		imgData[i] = (char) 254;
	}

	//printf("hi\n");

	//Populating matrix with body positions
	int x_centre_pixel;
	int y_centre_pixel;
	for (int i = 0; i < bodies.size(); ++i) {
		if (bodies[i].Mass == 0.f)
			continue;

		//Since positions can be negative, we add the field width and height to determine the new position in the image ( no negative coords)
		x_centre_pixel = (int) (((bodies[i].Position.X + fieldWidth)
				/ doubleFieldWidth) * img_width);
		y_centre_pixel = (int) (((bodies[i].Position.Y + fieldHeight)
				/ doubleFieldHeight) * img_height);

		int x_sq;
		int y_sq;
		int y_min =
				y_centre_pixel - bodies[i].Radius < 0 ?
						0 : y_centre_pixel - bodies[i].Radius;
		int y_max =
				y_centre_pixel + bodies[i].Radius >= img_height ?
						img_height : y_centre_pixel + bodies[i].Radius;
		int x_min =
				x_centre_pixel - bodies[i].Radius < 0 ?
						0 : x_centre_pixel - bodies[i].Radius;
		int x_max =
				x_centre_pixel + bodies[i].Radius > img_width ?
						img_width : x_centre_pixel + bodies[i].Radius;
		//Brute forcing every pixel and checking whether it is in body's radius.
		for (int y = y_min; y < y_max; ++y) {
			if (y < 0)
				y = 0;
			if (y > img_height)
				y = img_height;
			for (int x = x_min; x < x_max; ++x) {
				if (x < 0)
					x = 0;
				if (x > img_width)
					x = img_width;
				x_sq = (x - x_centre_pixel) * (x - x_centre_pixel);
				y_sq = (y - y_centre_pixel) * (y - y_centre_pixel);
				//std::cout<<"x after: "<< x <<std::endl;
				if (x_sq + y_sq
						<= roundf(bodies[i].Radius * bodies[i].Radius)) {
#ifndef NDEBUG
					assert((x >= 0) && (x <= img_width));
					assert((y >= 0) && (y <= img_height));
#endif
					//img_data(y, x) = 0;
					imgData[(img_width * y) + x] = (char) 0;//static_cast<char>(0);
				};
			}
		}
	}
//	jbutil::image<int> img(img_height, img_width,1, 255);
//	img.set_channel(0, img_data);

	std::ofstream outImg;
	outImg.open(filename, std::ofstream::out);
	std::cout << "Saving iteration to disk" << std::endl;
	if (outImg.is_open()) {
		outImg << "P5\n" << img_width << " " << img_height << "\n255\n";//<< "\n255\n";
		for (int i = 0; i < imageSize; ++i) {
			//printf("%d\n", imgData[i]);
			outImg << imgData[i];

		}
		//img.save(img_out);
		outImg.close();
	} else
		std::cerr << "Error writing image to file:" << filename << std::endl
				<< "Ensure the the folder exists" << std::endl;

	delete[] imgData;
	// std::ofstream img_out(filename.c_str(), std::ofstream::binary);
	// if (img_out.is_open())
	// {
	// 	img.save(img_out);
	// 	img_out.close();
	// }
	// else
	// 	std::cerr << "Error writing image to file:" << filename << std::endl <<"Ensure the the folder exists" <<std::endl;
}



__global__ void setupDeviceVariables(/*Particle** bodiesAddr, float** massesAddr, int* numBodiesAddr,*/ Particle* bodies, int size)
{
	//printf("bodies: %p\n", &bodies);
	//bodiesAddr = &bodies;
//	float* updatedMasses = new float[size];
//	for (int i = 0; i < size; ++i) {
//		updatedMasses[i] = bodies[i].Mass;
//		//printf("Mass: %.4f \nRadius: %.4f\n", body.Mass, body.Radius);
//	}
	//deviceUpdatedMasses = updatedMasses;
	deviceBodies = bodies;
	//printf("hey\n");
	deviceNumBodies = size;
	//massesAddr = &updatedMasses;
	//*numBodiesAddr = size;
}

__global__ void resetUpdatedMasses(/*Particle** bodiesAddr, float** massesAddr, int* numBodiesAddr*/)
{
	const Particle* bodies = deviceBodies;//*bodiesAddr;
	int numBodies = deviceNumBodies;
	//int size = deviceNumBodies;
	//float* updatedMasses = *massesAddr;
	float* newMasses = new float[numBodies];

	for (int i = 0; i < numBodies; ++i) {
		//printf("entered\n");
		newMasses[i] = bodies[i].Mass;
		//printf("Mass: %.4f \nRadius: %.4f\n", bodies[i].Mass, bodies[i].Radius);
	}
	//delete[] *massesAddr;
	if(deviceUpdatedMasses != nullptr){
		delete[] deviceUpdatedMasses;
		//printf("deleted masses\n");
	}
	deviceUpdatedMasses = newMasses;
	//printf("resetting\n");
//	massesAddr = &newMasses;
}

__global__ void DeleteMassesAndUpdateBodies(/*Particle** bodiesAddr, float** massesAddr, int* numBodies*/)
{
	Particle* bodies = deviceBodies;//*bodiesAddr;
	Particle* newBodies;
	float* updatedMasses = deviceUpdatedMasses;//*massesAddr;
	int size = deviceNumBodies;

	//Using this as a buffer to keep track of which particles will be deleted (sort of like a stack) - may waste some memory but doesnt use dynamic allocation
	int* indicesToDelete = new int[size];
	int deleteSize = 0;
	int newSize;

	for (size_t i = 0; i < size; ++i) {
		if (updatedMasses[i] == 0.f) {
			indicesToDelete[deleteSize] = i;
			deleteSize++;
			//newBodies.push_back(bodies[i]);
		}
	}

	newSize = size - deleteSize;
	newBodies = new Particle[newSize];
	//deviceNumBodies = newSize;

	int deleteIdx = 0;
	int newBodyIdx = 0;
	for (size_t i = 0; i < size; ++i)
	{
		if(i == indicesToDelete[deleteIdx])
		{
			deleteIdx++;

		}
		else
		{
			newBodies[newBodyIdx] = bodies[i];
			newBodyIdx++;
		}
	}

	delete[] deviceBodies;
	delete[] indicesToDelete;

	deviceNumBodies = newSize;
	deviceBodies = newBodies;
//	bodiesAddr = &newBodies;
//	delete[] bodies;
//	delete[] updatedMasses;
	//delete[] indicesToDelete;
	//bodies = newBodies;
}

int main(int argc, char **argv) {
	/*if(argc < 5){
	 std::cerr<<"Incorrect arguments. <particle count> <iterations> <save-image-every-x-iteration> <image-path>"<<std::endl;
	 exit(0);
	 }*/

	std::cout<<"Running simulation with the following settings:\n";
	ConfigData config = parseConfigFile("nbodyConfig.txt");
	std::cout<<"=====================\n";
	//exit(0);

	const int particleCount = config.particleCount;//std::stoi(argv[1]);
	const int maxIteration = config.totalIterations;//std::stoi(argv[2]);
	const int imageEveryIteration = config.save_Image_Every_Xth_Iteration;//std::stoi(argv[3]);
	const float timestep = config.timestep;
	const float minBodyMass = config.minRandBodyMass;
	const float maxBodyMass = config.maxRandBodyMass;
	fieldWidth = config.fieldWidth;
	doubleFieldWidth = fieldWidth << 1;
	fieldHeight = config.fieldHeight;
	doubleFieldHeight = fieldHeight << 1;


	std::stringstream fileOutput;
	std::stringstream imgOut;
	std::vector<Particle> bodies;
	//std::vector<float> updatedMasses;

	//jbutil::randgen gen;
	//gen.seed(42);

	Particle p;
	float x, y, m, r;

	std::mt19937 generator;
	std::uniform_real_distribution<float> distribution(0.f, 1.f);
	std::uniform_real_distribution<float> massDist(minBodyMass, maxBodyMass);
	std::uniform_real_distribution<float> radiusDist(config.minRadius, config.maxRadius);
	// distribution(generator);

	//Randomly generating bodies
	 for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	 {
	 	x = (distribution(generator) * doubleFieldWidth) - fieldWidth; //gen.fval(0, doubleFieldWidth) - fieldWidth;
	 	y = (distribution(generator) * doubleFieldHeight) - fieldHeight;
	 	m = massDist(generator);
	 	r = radiusDist(generator);//gen.fval(5, 50);
	 	//printf("Generated Particle:\nPos: (%.4f,%.4f)\nMass: %.4f\nRadius: %.4f\n", x, y, m, r);
	 	p = Particle(Vec2f(x,y), Vec2f(0.0,0.0), m, r);
	 	bodies.push_back(p);
	 	// updatedMasses.push_back(p.Mass);
	 }

	//Utilising jbutil
	// x = gen.fval(0, doubleFieldWidth) - fieldWidth;
	// y = gen.fval(0, doubleFieldHeight) - fieldHeight;
	// m = gen.fval(minBodyMass, maxBodyMass);
	// r = gen.fval(5, 50);
	// p = Particle(Vec2f(x,y), Vec2f(0.0,0.0), m, r);
	// bodies.push_back(p);
	// updatedMasses.push_back(p.Mass);

	//Particle p3(Vec2f(0,0), Vec2f(0,0), 1.989e20, 5);
	//bodies.push_back(p3);

	// ========================
	/*Particle p1(Vec2f(-500, 0), Vec2f(0, 10), 50, 10);
	Particle p2(Vec2f(500, 0), Vec2f(0, -10), 50, 10);
	Particle p3(Vec2f(0, 0), Vec2f(0, 0), 1e15f, 18);
	bodies.push_back(p1);
	bodies.push_back(p2);
	bodies.push_back(p3);*/
	//=======================
	//updatedMasses.push_back(p1.Mass);
	//updatedMasses.push_back(p2.Mass);
	//updatedMasses.push_back(p3.Mass);
	// 	updatedMasses.push_back(p.Mass);

	// bodies.push_back(Particle(Vec2f(0.f,0.f), 50));
	// updatedMasses.push_back(50.f);
	const int imgWidth = config.imgWidth;
	const int imgHeight = config.imgHeight;
	size_t imageSize = imgWidth * imgHeight;

	//printf("SIZE: %d\n", sizeof(Particle));
	float* d_updatedMasses;
	Particle* d_bodies;
	char* imgData;
	char* d_imgData;
	int threadsPerBlock = 64;
	int blocks;

	//Particle** newBodiesAddress;
	//float** newUpdatedMassesAddress;
	//int* numBodiesAddress;

	//cudaMalloc((void**) &newBodiesAddress, sizeof(Particle*));
	//cudaMalloc((void**) &newUpdatedMassesAddress, sizeof(float*));
	//cudaMalloc((void**) &numBodiesAddress, sizeof(int));

//	cudaMalloc((void **) &d_bodies, bodies.size() * sizeof(Particle));
//	cudaMemcpy(d_bodies, bodies.data(), bodies.size() * sizeof(Particle), cudaMemcpyHostToDevice);
//	setupDeviceVariables<<<1,1>>>(d_bodies, bodies.size());
//	CUDA_SYNC_CHECK();
 	//printf("passed\n");
	for (int run = 0; run < MAX_RUNS; ++run) {
		for (int iteration = 0; iteration < maxIteration; ++iteration)
		{
			//printf("Iteration: %d\n", iteration);
			std::vector<Particle> newBodies;
			std::vector<float> updatedMasses(bodies.size());

			cudaMalloc((void **) &d_bodies, bodies.size() * sizeof(Particle));
			cudaMalloc((void **) &d_updatedMasses,
					bodies.size() * sizeof(float));



			for (int i = 0; i < bodies.size(); ++i) {
				updatedMasses[i] = bodies[i].Mass;
				//printf("Mass: %.4f \nRadius: %.4f\n", body.Mass, body.Radius);
			}



			//Since bodies can decrease, we need to ensure that at least 1 block is always present
			blocks = bodies.size() < threadsPerBlock ? 1 : bodies.size() / threadsPerBlock;

			//Copying data over to device
			cudaMemcpy(d_bodies, bodies.data(),
					bodies.size() * sizeof(Particle), cudaMemcpyHostToDevice);
			cudaMemcpy(d_updatedMasses, updatedMasses.data(),
					updatedMasses.size() * sizeof(float),
					cudaMemcpyHostToDevice);

			//resetUpdatedMasses<<<1,1>>>();
			//CUDA_SYNC_CHECK();
			//Calculating movement
			//printf("Bodies: %ld\n", bodies.size());
			//printf("Blocks: %d\nThreads/Block: %d\nTotal threads: %d\n", blocks, threadsPerBlock, blocks * threadsPerBlock);
			ComputeForces<<<blocks, threadsPerBlock>>>(d_bodies, d_updatedMasses, bodies.size(),timestep, fieldWidth, fieldHeight);
			CUDA_SYNC_CHECK();
			MoveBodies<<<blocks, threadsPerBlock>>>(d_bodies, d_updatedMasses, timestep);
			CUDA_SYNC_CHECK();

			//DeleteMassesAndUpdateBodies<<<1,1>>>();

			//Copying data back to host
			cudaMemcpy(bodies.data(), d_bodies,bodies.size() * sizeof(Particle), cudaMemcpyDeviceToHost);
			cudaMemcpy(updatedMasses.data(), d_updatedMasses, updatedMasses.size() * sizeof(float), cudaMemcpyDeviceToHost);

			for (size_t i = 0; i < bodies.size(); ++i) {
				if (updatedMasses[i] != 0.f) {
					newBodies.push_back(bodies[i]);
				}
			}
			bodies = newBodies;

			if (iteration % imageEveryIteration == 0) {
				imgData = new char[imageSize];
				for (int i = 0; i < imageSize; ++i) {
					imgData[i] = (char) 254;
				}
				cudaMalloc((void**) &d_imgData, imageSize);
				cudaMemcpy(d_imgData, imgData, imageSize,
						cudaMemcpyHostToDevice);
				generateImage<<<blocks, threadsPerBlock>>>(d_bodies, d_imgData, imgWidth, imgHeight, fieldWidth, fieldHeight);
				CUDA_SYNC_CHECK()
				;
				cudaMemcpy(imgData, d_imgData, imageSize,
						cudaMemcpyDeviceToHost);
				imgOut.str(std::string());
				imgOut << config.imagePath << "/iteration_" << iteration << ".ppm";
				printf("Saving Iteration %d\n", iteration);
				saveImageToDisk(imgOut.str(), imgData, imgWidth, imgHeight);

				cudaFree(d_imgData);
				delete[] imgData;
			}
			cudaFree(d_bodies);
			cudaFree(d_updatedMasses);
		}
	}
	return 0;
}
