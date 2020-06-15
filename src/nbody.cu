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
#define THREADS_PER_BLOCK 128
#define GRAV_CONSTANT 6.67408e-11f
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

//__constant__ __device__ float GRAV_CONSTANT = 6.67408e-11f;


/*
 * Particle structure
 */
//struct __align__(32)  Particle {
//	Vec2f Position;
//	Vec2f Velocity;
//	float Mass;
//	float Radius;
//
//	CUDA_CALLABLE_MEMBER Particle(void) :
//			Position(Vec2f(0, 0)), Velocity(0.f, 0.f), Mass(0.f) {
//	}
//
//	CUDA_CALLABLE_MEMBER Particle(Vec2f p, double m) :
//			Position(p), Velocity(0.0, 0.0), Mass(m) {
//	}
//
//	CUDA_CALLABLE_MEMBER Particle(Vec2f p, Vec2f v, float m, float radius) :
//			Position(p), Velocity(v), Mass(m), Radius(radius) {
//	}
//
//	CUDA_CALLABLE_MEMBER bool intersect(const Particle& p) {
//		Vec2f direction = p.Position - Position;
//		float distance = (direction.X * direction.X)
//				+ (direction.Y * direction.Y);
//		return distance < (Radius + p.Radius) * (Radius + p.Radius);
//	}
//};

struct BodiesData
{
	int numBodies;
	size_t size;
	void* contiguousData;
	Vec2f* Positions;
	Vec2f* Velocities;
	float* Masses;
	float* Radii;

	void* d_contiguousData;

	BodiesData() : numBodies(), size(), contiguousData(), Positions(), Velocities(), Masses(), Radii(), d_contiguousData()
	{
	}

	void alloc(int bodies)
	{
		numBodies = bodies;
		size = numBodies * (sizeof(Vec2f) + sizeof(Vec2f) + sizeof(float) + sizeof(float));
		contiguousData = (void*)malloc(size);
		if(contiguousData == NULL)
		{
			printf("Failed to allocate body data");
			exit(0); //fail since program can't continue
		}
		//Splitting the contiguous data between the required arrays
		Positions = (Vec2f*) contiguousData;
		//printf("cont: %p\nPositions: %p\n", &contiguousData, );
		Velocities = (Vec2f*) &Positions[numBodies];
		Masses = (float*) &Velocities[numBodies];
		Radii = (float* ) &Masses[numBodies];
		d_contiguousData = nullptr;
	}

	void freeData()
	{
		free(contiguousData);
		cudaFree(d_contiguousData);
		d_contiguousData = nullptr;
	}

	void uploadToDevice()
	{
		//prevent reallocating memory
		if(d_contiguousData == nullptr)
		{
			cudaMalloc((void**)&d_contiguousData, size);
			cudaMemcpy(d_contiguousData, contiguousData, size, cudaMemcpyHostToDevice);
		}
	}

	//Make sure to delete previous data before calling this
	BodiesData& operator= (const BodiesData& newData)
	{
		contiguousData = newData.contiguousData;
		numBodies = newData.numBodies;
		size = newData.size;
		Positions = newData.Positions;
		Velocities = newData.Velocities;
		Masses = newData.Masses;
		Radii = newData.Radii;
		d_contiguousData = newData.d_contiguousData;
		return *this;
	}
};

//typedef struct rgb_data {
//	int r;
//	int g;
//	int b;
//} RGB;

//Device global variables to prevent copying to host every iteration
//__device__ Particle* deviceBodies;
//__device__ float* deviceUpdatedMasses;
//__device__ int deviceNumBodies;

__device__ inline bool areParticlesColliding(const Vec2f& p0, const float r0, const Vec2f& p1, const float r1)
{
	Vec2f direction = p1 - p0;
	float distance = (direction.X * direction.X) + (direction.Y * direction.Y);
	return distance < (r0 + r1) * (r0 + r1);
}

/*
 * Compute forces of particles exerted on one another
 */
//Particle* d_bodies, float* updatedMasses, Vec2f* updatedVelocities, float* updatedRadii,
__global__ void /*__launch_bounds__(THREADS_PER_BLOCK, 2)*/ ComputeForces(void* bodyData, float* updatedMasses, Vec2f* updatedVelocities,
		float* updatedRadii, int bodiesNum, float timestep, int fieldWidth, int fieldHeight)
{
	//Particle* bodies = d_bodies;//deviceBodies;
	int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

	//Subdividing the contiguous array
	int numBodies = bodiesNum;
	Vec2f* positions = (Vec2f*) bodyData;
	Vec2f* velocities = (Vec2f*) &positions[numBodies];
	float* masses = (float*) &velocities[numBodies];
	float* radii = (float*) &masses[numBodies];

	//float* updated_masses = updatedMasses;
	Vec2f direction, force, acceleration;
	force = 0.f, acceleration = 0.f;
	float distance;

	extern __shared__ Vec2f sharedMem[];
	Vec2f* shrdPositions = (Vec2f* )&sharedMem;
	float* shrdMasses = (float*) &shrdPositions[THREADS_PER_BLOCK];
	float* shrdRadii = (float*) &shrdMasses[THREADS_PER_BLOCK];

	Vec2f* blockThreadsPositions =(Vec2f*) &shrdRadii[THREADS_PER_BLOCK];
	float* blockThreadsMasses = (float*) &blockThreadsPositions[THREADS_PER_BLOCK];
	float* blockThreadsRadii = (float*) &blockThreadsMasses[THREADS_PER_BLOCK];

	Vec2f* shrdUpdatedVelocities = (Vec2f* ) &blockThreadsRadii[THREADS_PER_BLOCK];

	//Particle &p1 = bodies[j];
	//Loading data for the body that the current thread is handling
//	const Vec2f& threadBodyPosition = positions[globalThreadIdx];
//	const float threadBodyMass = masses[globalThreadIdx] ;
//	const float threadBodyRadius = radii[globalThreadIdx];

	blockThreadsPositions[threadIdx.x] = positions[globalThreadIdx];
	blockThreadsMasses[threadIdx.x] = masses[globalThreadIdx] ;
	blockThreadsRadii[threadIdx.x] = radii[globalThreadIdx];

	float updatedBodyMass = blockThreadsMasses[threadIdx.x];//threadBodyMass;
	float updatedBodyRadius = blockThreadsRadii[threadIdx.x];
	int globalBodyIdx;
	int shIdx;
	for (int k = 0; k < (int)(numBodies / THREADS_PER_BLOCK); ++k)
	{
		//Loading the next p bodies (p = threads per block)
		globalBodyIdx = (globalThreadIdx + (THREADS_PER_BLOCK * k)) % numBodies;
		//if(blockIdx.x == 0) printf("threadIdx: %d\nglobalId: %d\n\n", threadIdx.x, globalBodyIdx);
		shrdPositions[threadIdx.x] = positions[globalBodyIdx];
		shrdMasses[threadIdx.x] = masses[globalBodyIdx];
		shrdRadii[threadIdx.x] = radii[globalBodyIdx];
		__syncthreads();
		for (int shrdOffset = 0; shrdOffset < THREADS_PER_BLOCK ; ++shrdOffset)
		{
			if (globalBodyIdx == globalThreadIdx)
				continue;

			//Ensuring threads loop and read every shared location concurrently
			shIdx = (threadIdx.x + shrdOffset) % THREADS_PER_BLOCK;

//			bool intersect = areParticlesColliding(threadBodyPosition, threadBodyRadius,
//					shrdPositions[shIdx], shrdRadii[shIdx]);

			bool intersect = areParticlesColliding(blockThreadsPositions[threadIdx.x], blockThreadsRadii[threadIdx.x],
								shrdPositions[shIdx], shrdRadii[shIdx]);

			//Testing for body - body collision
			if ((blockThreadsMasses[threadIdx.x] >= shrdMasses[shIdx]) && intersect)
			{
				updatedBodyMass = /*threadBodyMass*/ blockThreadsMasses[threadIdx.x] + shrdMasses[shIdx];
				updatedBodyRadius = /*threadBodyRadius*/ blockThreadsRadii[threadIdx.x] + shrdRadii[shIdx];
//				shrdUpdatedMasses[threadIdx.x] = threadBodyMass + shrdMasses[shIdx];
//				shrdUpdatedRadii[threadIdx.x] = threadBodyRadius + shrdRadii[shIdx];;//p1.Radius += p2.Radius;
				continue;
			} else if ((blockThreadsRadii[threadIdx.x] < shrdMasses[shIdx]) && intersect) {
				//shrdUpdatedMasses[threadIdx.x] = 0.f;
				updatedBodyMass = 0.f;
				continue;
			}

			// Compute direction vector
			direction = shrdPositions[shIdx] - blockThreadsPositions[threadIdx.x];/*threadBodyPosition*/;
			distance = direction.length();

#ifndef NDEBUG
			assert(distance != 0);
#endif
			// Accumulate force
			force += (direction * shrdMasses[shIdx]) / (distance * distance * distance);
		}
		__syncthreads();
	};
	__syncthreads();
	updatedMasses[globalThreadIdx] = updatedBodyMass;//shrdUpdatedMasses[threadIdx.x];
	updatedRadii[globalThreadIdx] = updatedBodyRadius;//shrdUpdatedRadii[threadIdx.x];

	// Compute acceleration for body
	acceleration = force * GRAV_CONSTANT;
	shrdUpdatedVelocities[threadIdx.x] = acceleration * timestep;

	//Border collision
	if (blockThreadsPositions[threadIdx.x].X + (acceleration.X * timestep) > fieldWidth - blockThreadsRadii[threadIdx.x]
			|| blockThreadsPositions[threadIdx.x].X + (acceleration.X * timestep) < -fieldWidth + blockThreadsRadii[threadIdx.x])
		shrdUpdatedVelocities[threadIdx.x].X *= -1;
		//updatedVelocities[globalThreadIdx].X *= -1;
	if (blockThreadsPositions[threadIdx.x].Y + (acceleration.Y * timestep) > fieldHeight - blockThreadsRadii[threadIdx.x]
			|| blockThreadsPositions[threadIdx.x].Y + (acceleration.Y * timestep) < -fieldHeight + blockThreadsRadii[threadIdx.x])
		shrdUpdatedVelocities[threadIdx.x].Y *= -1;
		//updatedVelocities[globalThreadIdx].Y *= -1;

	updatedVelocities[globalThreadIdx] = shrdUpdatedVelocities[threadIdx.x];
	/*printf("Mass (p%d) : %.5f\n", (int)j, p1.Mass);
	 printf("Mass (p%d) : %.5f\n", (int)k, p2.Mass);
	 printf("Direction (p%d to p%d) : %.2f, %.2f\n", (int)j, (int)k,direction.X, direction.Y);
	 printf("Distance  (p%d to p%d): %.4f\n", (int)j, (int)k, distance);
	 printf("Force (p%d)  (running sum): %.2f, %.2f\n", (int)j, force.X, force.Y);*/
}

/*
 * Update particle positions
 */

__global__ void MoveBodies(void* bodyData, float* updatedMasses, Vec2f* updatedVelocities,float* updatedRadii,
		int numBodies, float p_deltaT)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	Vec2f* positions = (Vec2f*) bodyData;
	Vec2f* velocities = (Vec2f*) &positions[numBodies];
	float* masses = (float*) &velocities[numBodies];
	float* radii = (float*) &masses[numBodies];
	//Particle* p_bodies = bodies;//deviceBodies;//*bodiesAddr;
	float* updated_masses = updatedMasses;//deviceUpdatedMasses;
	if (updated_masses[j] != 0.f) {
		//printf("UPDATED MASS (p%d): %.2f\n", (int)j, updated_masses[j]);
		positions[j] += updatedVelocities[j] * p_deltaT;//p_bodies[j].Velocity * p_deltaT;
		masses[j] = updated_masses[j];
		radii[j] = updatedRadii[j];
	}
}

__global__ void generateImage(void* bodyData, int numBodies, char* imgData, int width,
		int height, int fieldWidth, int fieldHeight)
{
	//Particle* bodies = d_bodies;
	Vec2f* positions = (Vec2f*) bodyData;
	Vec2f* velocities = (Vec2f*) &positions[numBodies];
	float* masses = (float*) &velocities[numBodies];
	float* radii = (float*) &masses[numBodies];
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ Vec2f sharedMem[];
	Vec2f* shrdPositions = (Vec2f* )&sharedMem;
	float* shrdRadii = (float*) &shrdPositions[THREADS_PER_BLOCK];
	//float* shrdMasses = (float*) &shrdPositions[THREADS_PER_BLOCK];

	shrdPositions[threadIdx.x] = positions[i];
	shrdRadii[threadIdx.x] = radii[i];

	//printf("hi\n");

	const int img_width = width;
	const int img_height = height;
	const int doubleFieldWidth = fieldWidth << 1;
	const int doubleFieldHeight = fieldHeight << 1;

	//Since positions can be negative, we add the field width and height to determine the new position in the image ( no negative coords)
	int x_centre_pixel = (int) (((shrdPositions[threadIdx.x].X + fieldWidth) / doubleFieldWidth) * img_width);
	int y_centre_pixel = (int) (((shrdPositions[threadIdx.x].Y + fieldHeight) / doubleFieldHeight) * img_height);

	int x_sq;
	int y_sq;
	int y_min = y_centre_pixel - shrdRadii[threadIdx.x] < 0 ?	0 : y_centre_pixel - shrdRadii[threadIdx.x];
	int y_max = y_centre_pixel + shrdRadii[threadIdx.x] >= img_height ? img_height : y_centre_pixel + shrdRadii[threadIdx.x];
	int x_min =	x_centre_pixel - shrdRadii[threadIdx.x] < 0 ? 0 : x_centre_pixel - shrdRadii[threadIdx.x];
	int x_max = x_centre_pixel + shrdRadii[threadIdx.x] > img_width ?	img_width : x_centre_pixel + shrdRadii[threadIdx.x];

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
			if (x_sq + y_sq <= (int)(shrdRadii[threadIdx.x]* shrdRadii[threadIdx.x]))
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

int main(int argc, char **argv) {
	/*if(argc < 5){
	 std::cerr<<"Incorrect arguments. <particle count> <iterations> <save-image-every-x-iteration> <image-path>"<<std::endl;
	 exit(0);
	 }*/

	//printf("Size: %d\n", sizeof(Vec2f));
	//exit(0);

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
	//std::vector<Particle> bodies;
	//std::vector<float> updatedMasses;


//	std::mt19937 generator;
//	std::uniform_real_distribution<float> distribution(0.f, 1.f);
//	std::uniform_real_distribution<float> massDist(minBodyMass, maxBodyMass);
//	std::uniform_real_distribution<float> radiusDist(config.minRadius, config.maxRadius);
//	// distribution(generator);
//
//	//Randomly generating bodies
//	 for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
//	 {
//	 	x = (distribution(generator) * doubleFieldWidth) - fieldWidth; //gen.fval(0, doubleFieldWidth) - fieldWidth;
//	 	y = (distribution(generator) * doubleFieldHeight) - fieldHeight;
//	 	m = massDist(generator);
//	 	r = radiusDist(generator);//gen.fval(5, 50);
//	 	//printf("Generated Particle:\nPos: (%.4f,%.4f)\nMass: %.4f\nRadius: %.4f\n", x, y, m, r);
//	 	p = Particle(Vec2f(x,y), Vec2f(0.0,0.0), m, r);
//	 	bodies.push_back(p);
//	 	// updatedMasses.push_back(p.Mass);
//	 }


	//Particle p;
	float x, y, m, r;
	BodiesData bData;
	bData.alloc(particleCount);
	printf("Bodies: %d\n", bData.numBodies);

	jbutil::randgen gen;
	gen.seed(42);

	//Randomly generating body data
	for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	{
		 x = gen.fval(0, doubleFieldWidth) - fieldWidth;
		 y = gen.fval(0, doubleFieldHeight) - fieldHeight;
		 m = gen.fval(minBodyMass, maxBodyMass);
		 r = gen.fval(config.minRadius, config.maxRadius);
		 //printf("Base: %p\n Offset: %p\n", &bData.Positions, &bData.Velocities);
		 bData.Positions[bodyIndex] = Vec2f(x, y);
		 bData.Velocities[bodyIndex] = Vec2f(0.f, 0.f);
		 bData.Masses[bodyIndex] = m;
		 bData.Radii[bodyIndex] = r;
		 //p = Particle(Vec2f(x,y), Vec2f(0.0,0.0), m, r);
		 //bodies.push_back(p);
		 //updatedMasses.push_back(p.Mass);
	}

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
	float* d_updatedRadii;
	Vec2f* d_updatedVelocities;
	//Particle* d_bodies;
	char* imgData;
	char* d_imgData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks;

	BodiesData newData;
	int numBodies = particleCount;
	int newNumBodies;

//	std::vector<float> updatedMasses(numBodies);
//	std::vector<float> updatedRadii(numBodies);
//
//
//	//cudaMalloc((void **) &d_bodies, bodies.size() * sizeof(Particle));
//	cudaMalloc((void **) &d_updatedMasses, numBodies * sizeof(float));
//	cudaMalloc((void **) &d_updatedRadii, numBodies * sizeof(float));
//	cudaMalloc((void **) &d_updatedVelocities, numBodies * sizeof(Vec2f));
//
//
//
//	for (int i = 0; i < numBodies; ++i) {
//		updatedMasses[i] = bData.Masses[i];
//		updatedRadii[i] = bData.Radii[i];
//		//printf("Mass: %.4f \nRadius: %.4f\n", body.Mass, body.Radius);
//	}
//
//	bData.uploadToDevice();
//	cudaMemcpy(d_updatedMasses, updatedMasses.data(), updatedMasses.size() * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_updatedRadii, updatedRadii.data(), updatedRadii.size() * sizeof(float), cudaMemcpyHostToDevice);
	for (int run = 0; run < MAX_RUNS; ++run) {
		for (int iteration = 0; iteration < maxIteration; ++iteration)
		{
			//printf("Iteration: %d\n", iteration);
			//std::vector<Particle> newBodies;
			std::vector<float> updatedMasses(numBodies);
			std::vector<float> updatedRadii(numBodies);


			//cudaMalloc((void **) &d_bodies, bodies.size() * sizeof(Particle));
			cudaMalloc((void **) &d_updatedMasses, numBodies * sizeof(float));
			cudaMalloc((void **) &d_updatedRadii, numBodies * sizeof(float));
			cudaMalloc((void **) &d_updatedVelocities, numBodies * sizeof(Vec2f));



			for (int i = 0; i < numBodies; ++i) {
				updatedMasses[i] = bData.Masses[i];
				updatedRadii[i] = bData.Radii[i];
				//printf("Mass: %.4f \nRadius: %.4f\n", body.Mass, body.Radius);
			}



			//Since bodies can decrease, we need to ensure that at least 1 block is always present
			blocks = numBodies < threadsPerBlock ? 1 : numBodies / threadsPerBlock;
			//blocks = numBodies / threadsPerBlock;

			//Copying data over to device
			//cudaMemcpy(d_bodies, bodies.data(), bodies.size() * sizeof(Particle), cudaMemcpyHostToDevice);
			bData.uploadToDevice();
			cudaMemcpy(d_updatedMasses, updatedMasses.data(), updatedMasses.size() * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_updatedRadii, updatedRadii.data(), updatedRadii.size() * sizeof(float), cudaMemcpyHostToDevice);
			//resetUpdatedMasses<<<1,1>>>();
			//CUDA_SYNC_CHECK();
			//Calculating movement
			ComputeForces<<<blocks, threadsPerBlock, threadsPerBlock * ((2 * (sizeof(Vec2f) + sizeof(float) + sizeof(float))) + sizeof(Vec2f))>>>
					(bData.d_contiguousData, d_updatedMasses, d_updatedVelocities, d_updatedRadii, numBodies,timestep, fieldWidth, fieldHeight);
			//CUDA_SYNC_CHECK();
			MoveBodies<<<blocks, threadsPerBlock>>>(bData.d_contiguousData ,d_updatedMasses, d_updatedVelocities, d_updatedRadii, bData.numBodies, timestep);
			//CUDA_SYNC_CHECK();

			//DeleteMassesAndUpdateBodies<<<1,1>>>();

			//Copying data back to host
			cudaMemcpy(bData.contiguousData, bData.d_contiguousData, bData.size, cudaMemcpyDeviceToHost);
			//cudaMemcpy(updatedMasses.data(), d_updatedMasses, updatedMasses.size() * sizeof(float), cudaMemcpyDeviceToHost);

			newNumBodies = 0;
			for (size_t i = 0; i < numBodies; ++i) {
				if (bData.Masses[i] != 0.f)
				{
					newNumBodies++;
					//newBodies.push_back(bodies[i]);
				}
			}

			newData.alloc(newNumBodies);

			//copying relevant data to new place in memory while deleting other masses
			int newIdx = 0;
			for (size_t i = 0; i < numBodies; ++i)
			{
				if (bData.Masses[i] != 0.f)
				{
					 newData.Positions[newIdx] = bData.Positions[i];
					 newData.Velocities[newIdx] = bData.Velocities[i];
					 newData.Masses[newIdx] = bData.Masses[i];
					 newData.Radii[newIdx] = bData.Radii[i];
					 newIdx++;
				}
			}

			numBodies = newNumBodies;
			bData.freeData();
			bData = newData;

			if (iteration % imageEveryIteration == 0) {
				imgData = new char[imageSize];
				for (int i = 0; i < imageSize; ++i) {
					imgData[i] = (char) 254;
				}

				bData.uploadToDevice();
				cudaMalloc((void**) &d_imgData, imageSize);
				cudaMemcpy(d_imgData, imgData, imageSize,
						cudaMemcpyHostToDevice);
				generateImage<<<blocks, threadsPerBlock, threadsPerBlock * (sizeof(Vec2f) + sizeof(float)) >>>(bData.d_contiguousData, bData.numBodies, d_imgData, imgWidth, imgHeight, fieldWidth, fieldHeight);
				CUDA_SYNC_CHECK();
				cudaMemcpy(imgData, d_imgData, imageSize,
						cudaMemcpyDeviceToHost);
				imgOut.str(std::string());
				imgOut << config.imagePath << "/iteration_" << iteration << ".ppm";
				printf("Saving Iteration %d\n", iteration);
				saveImageToDisk(imgOut.str(), imgData, imgWidth, imgHeight);

				cudaFree(d_imgData);
				delete[] imgData;
			}
		}
			CUDA_SYNC_CHECK();
			cudaFree(d_updatedMasses);
			cudaFree(d_updatedRadii);
			cudaFree(d_updatedVelocities);
			cudaDeviceReset();
	}
	return 0;
}
