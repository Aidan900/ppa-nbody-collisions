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

#define MAX_RUNS 1
#define THREADS_PER_BLOCK 128
#define GRAV_CONSTANT 6.67408e-11f

/*
 * Definitions for field dimensions
 */
int fieldWidth;
int doubleFieldWidth;// = fieldWidth << 1;
int fieldHeight;// = 2000;
int doubleFieldHeight;// = fieldHeight << 1;

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

	void uploadToDevice(cudaStream_t stream = 0)
	{
		//prevent reallocating memory
		if(d_contiguousData == nullptr)
		{
			cudaMalloc((void**)&d_contiguousData, size);
			cudaMemcpyAsync(d_contiguousData, contiguousData, size, cudaMemcpyHostToDevice, stream);
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

	void printData()
	{
		for(int i = 0 ; i < numBodies ; ++i)
		{
			printf("Body #%d\n", i);
			printf("Position: (%.4f, %.4f)\n", Positions[i].X, Positions[i].Y);
			printf("Velocity: (%.4f, %.4f)\n", Velocities[i].X, Velocities[i].Y);
			printf("Mass: %.4f\n", Masses[i]);
			printf("Radius: %.4f\n", Radii[i]);
			printf("--------------\n");
		}
	}
};

__device__ inline bool areParticlesColliding(const Vec2f& p0, const float r0, const Vec2f& p1, const float r1)
{
	/// 2 flops
	Vec2f direction = p1 - p0;
	/// 3 flops
	float distance = (direction.X * direction.X) + (direction.Y * direction.Y);
	/// 3 flops, 1 comp
	return distance <= (r0 + r1) * (r0 + r1);
}

/*
 * Compute forces of particles exerted on one another
 */
__global__ void ComputeForces(void* bodyData, float* updatedMasses, Vec2f* updatedVelocities,
		float* updatedRadii, int bodiesNum, float timestep, int fieldWidth, int fieldHeight, int numBlocks, float growthRate)
{
	int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if(globalThreadIdx < bodiesNum)
	{
		//Subdividing the contiguous array
		int numBodies = bodiesNum;
		Vec2f* positions = (Vec2f*) bodyData;
		Vec2f* velocities = (Vec2f*) &positions[numBodies];
		float* masses = (float*) &velocities[numBodies];
		float* radii = (float*) &masses[numBodies];

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
		Vec2f* blockThreadsVelocities = (Vec2f *) &blockThreadsRadii[THREADS_PER_BLOCK];

		Vec2f* shrdUpdatedVelocities = (Vec2f* ) &blockThreadsVelocities[THREADS_PER_BLOCK];

		//Data about the body this thread represents
		blockThreadsPositions[threadIdx.x] = positions[globalThreadIdx];
		blockThreadsMasses[threadIdx.x] = masses[globalThreadIdx] ;
		blockThreadsRadii[threadIdx.x] = radii[globalThreadIdx];
		blockThreadsVelocities[threadIdx.x] = velocities[globalThreadIdx];

		float updatedBodyMass = blockThreadsMasses[threadIdx.x];
		float updatedBodyRadius = blockThreadsRadii[threadIdx.x];
		int globalBodyIdx;
		int shIdx;
		bool skip = true;
		int innerLoopLimit;
		bool deleted = false;
		// 1 comp, 1 int add
		for (int k = 0; k < numBlocks; ++k)
		{
			//Loading the next p bodies (p = threads per block)
			/// 3 ops
			globalBodyIdx = (globalThreadIdx + (THREADS_PER_BLOCK * k)) % numBodies;
			shrdPositions[threadIdx.x] = positions[globalBodyIdx];
			shrdMasses[threadIdx.x] = masses[globalBodyIdx];
			shrdRadii[threadIdx.x] = radii[globalBodyIdx];
			__syncthreads();

			//if this is the last block and less bodies then max threads are present, shared mem access needs to be limited
			/// 1 comp, 1 op
			innerLoopLimit = k == numBlocks - 1 ? numBodies % (THREADS_PER_BLOCK + 1) : THREADS_PER_BLOCK;
			///1 comp, 1 int add
			for (int shrdOffset = 0; shrdOffset < innerLoopLimit ; ++shrdOffset)
			{
				//If globalBodyIdx and globalThreadIdx are equal, we want to ensure that bodies are still compared in shared mem, hence the skip
				//1 compare
				if (skip && globalBodyIdx == globalThreadIdx)
				{
					skip = false;
					continue;
				}
				//Ensuring threads loop and read every shared location concurrently
				/// 1 int add, 1 modulo
				shIdx = (threadIdx.x + shrdOffset) % innerLoopLimit;

				/// 8 flops, 1 compare
				bool intersect = areParticlesColliding(blockThreadsPositions[threadIdx.x], blockThreadsRadii[threadIdx.x],
									shrdPositions[shIdx], shrdRadii[shIdx]);

				//Testing for body - body collision
				/// 2 compares (including both cases)
				if (intersect && (blockThreadsMasses[threadIdx.x] >= shrdMasses[shIdx]))
				{
					/// 3 flops in this branch
					updatedBodyMass += shrdMasses[shIdx];
					updatedBodyRadius += shrdRadii[shIdx] * growthRate;
					continue;
				}
				else if ( intersect && (blockThreadsMasses[threadIdx.x] < shrdMasses[shIdx]))
				{
					deleted = true;
					continue;
				}

				// Compute direction vector
				/// 2 flops (vec2f made of 2 floats)
				direction = shrdPositions[shIdx] - blockThreadsPositions[threadIdx.x];
				// 3 flops plus 1 sqrt
				distance = direction.length();

	#ifndef NDEBUG
				assert(distance != 0);
	#endif
				// Accumulate force
				/// 5 flops ( 3 mults, 1 div, 1 add)
				force += (direction * shrdMasses[shIdx]) / (distance * distance * distance);
			}
			__syncthreads();
		};
		__syncthreads();
		///1 comp
		updatedMasses[globalThreadIdx] =  deleted ? 0 : updatedBodyMass;//shrdUpdatedMasses[threadIdx.x];
		updatedRadii[globalThreadIdx] = updatedBodyRadius;//shrdUpdatedRadii[threadIdx.x];

		// Compute acceleration for body
		///1 flop
		acceleration = force * GRAV_CONSTANT;
		//1 flop
		shrdUpdatedVelocities[threadIdx.x] = acceleration * timestep;

		//Border collision
		///9 ops each  = 18 ops
		if (blockThreadsPositions[threadIdx.x].X + (acceleration.X * timestep) > fieldWidth - blockThreadsRadii[threadIdx.x]
				|| blockThreadsPositions[threadIdx.x].X + (acceleration.X * timestep) < -fieldWidth + blockThreadsRadii[threadIdx.x])
			blockThreadsVelocities[threadIdx.x].X *= -1;
		if (blockThreadsPositions[threadIdx.x].Y + (acceleration.Y * timestep) > fieldHeight - blockThreadsRadii[threadIdx.x]
				|| blockThreadsPositions[threadIdx.x].Y + (acceleration.Y * timestep) < -fieldHeight + blockThreadsRadii[threadIdx.x])
			blockThreadsVelocities[threadIdx.x].Y *= -1;

		/// 1 flop
		velocities[globalThreadIdx] = blockThreadsVelocities[threadIdx.x] + shrdUpdatedVelocities[threadIdx.x];
	}
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
	if ( j < numBodies)
	{
		Vec2f* positions = (Vec2f*) bodyData;
		Vec2f* velocities = (Vec2f*) &positions[numBodies];
		float* masses = (float*) &velocities[numBodies];
		float* radii = (float*) &masses[numBodies];
		float* updated_masses = updatedMasses;
		positions[j] += velocities[j] * p_deltaT;
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

	//Loading into shared mem
	shrdPositions[threadIdx.x] = positions[i];
	shrdRadii[threadIdx.x] = (radii[i] * width)/fieldWidth;

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
	std::cout << "Saving (" << imgWidth << "x" << imgHeight << ") to disk" << std::endl;
	if (outImg.is_open())
	{
		outImg << "P5\n" << imgWidth << " " << imgHeight << "\n255\n";
		for (int i = 0; i < imageSize; ++i) {
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
	double startTime = jbutil::gettime();

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

	//Particle p;
	float x, y, m, r;
	BodiesData bData;
	bData.alloc(particleCount);
	printf("Bodies: %d\n", bData.numBodies);

	jbutil::randgen gen;
//	gen.seed(jbutil::gettime());
	gen.seed(1024);

	//Randomly generating body data
	for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	{
		 x = gen.fval(0, doubleFieldWidth) - fieldWidth;
		 y = gen.fval(0, doubleFieldHeight) - fieldHeight;
		 m = gen.fval(minBodyMass, maxBodyMass);
		 r = gen.fval(config.minRadius, config.maxRadius);
		 bData.Positions[bodyIndex] = Vec2f(x, y);
		 bData.Velocities[bodyIndex] = Vec2f(0.f, 0.f);
		 bData.Masses[bodyIndex] = m;
		 bData.Radii[bodyIndex] = r;
	}

//	 bData.Positions[0] = Vec2f(-500, 0);
//	 bData.Positions[1] = Vec2f(500.f, 0.f);
//	 bData.Positions[2] = Vec2f(-600.f, -150.f);
//	 bData.Velocities[0] = Vec2f(10.f, 0); //3701.491
//	 bData.Velocities[1] = Vec2f(-10.f, 0.f);
//	 bData.Velocities[2] = Vec2f(0.f, 0.f);
//	 bData.Masses[0] = 1e10f;
//	 bData.Masses[1] = 1e14f;
//	 bData.Masses[2] = 1e3f;
//	 bData.Radii[0] = 10.f;
//	 bData.Radii[1] = 20.f;
//	 bData.Radii[2] = 7.f;

	 //bData.printData();


	// updatedMasses.push_back(50.f);
	const int imgWidth = config.imgWidth;
	const int imgHeight = config.imgHeight;
	size_t imageSize = imgWidth * imgHeight;

	float* d_updatedMasses;
	float* d_updatedRadii;
	Vec2f* d_updatedVelocities;

	char* imgData;
	char* d_imgData;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocks;

	BodiesData newData;
	int numBodies = particleCount;
	int newNumBodies;
	size_t sharedMemSize = threadsPerBlock * ((2 * (sizeof(Vec2f) + sizeof(float) + sizeof(float))) + 2 * sizeof(Vec2f));

	cudaStream_t calculationStream;
	cudaStream_t imageStream;

	cudaStreamCreate(&calculationStream);
	cudaStreamCreate(&imageStream);
	std::vector<float> updatedMasses(numBodies);
	std::vector<float> updatedRadii(numBodies);
	for (int run = 0; run < MAX_RUNS; ++run) {
		for (int iteration = 0; iteration < maxIteration; ++iteration)
		{
			cudaMalloc((void **) &d_updatedMasses, numBodies * sizeof(float));
			cudaMalloc((void **) &d_updatedRadii, numBodies * sizeof(float));

			//Filling up the new arrays
			for (int i = 0; i < numBodies; ++i) {
				updatedMasses[i] = bData.Masses[i];
				updatedRadii[i] = bData.Radii[i];
			}

			//Since bodies can decrease, we need to ensure that at least 1 block is always present
			blocks = numBodies < threadsPerBlock ? 1 : numBodies / threadsPerBlock;

			//Copying data over to device
			bData.uploadToDevice(calculationStream);
			cudaMemcpyAsync(d_updatedMasses, updatedMasses.data(), updatedMasses.size() * sizeof(float), cudaMemcpyHostToDevice, calculationStream);
			cudaMemcpyAsync(d_updatedRadii, updatedRadii.data(), updatedRadii.size() * sizeof(float), cudaMemcpyHostToDevice, calculationStream);

			//Calculating movement
			ComputeForces<<<blocks, threadsPerBlock, sharedMemSize, calculationStream>>>
					(bData.d_contiguousData, d_updatedMasses, d_updatedVelocities, d_updatedRadii, numBodies, timestep, fieldWidth, fieldHeight, blocks, config.growthRate);
			MoveBodies<<<blocks, threadsPerBlock, 0, calculationStream>>>(bData.d_contiguousData ,d_updatedMasses, d_updatedVelocities, d_updatedRadii, bData.numBodies, timestep);

			//Copying data back to host
			cudaMemcpyAsync(bData.contiguousData, bData.d_contiguousData, bData.size, cudaMemcpyDeviceToHost, calculationStream);

			newNumBodies = 0;
			for (size_t i = 0; i < numBodies; ++i) {
				if (bData.Masses[i] != 0.f)
				{
					newNumBodies++;
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

			//Saving the image generated asynchronously in the previous iteration
			if ((iteration - 1) % imageEveryIteration == 0)
			{
				//synchronizing to make sure image is ready
				cudaStreamSynchronize(imageStream);
				imgOut.str(std::string());
				imgOut << config.imagePath << "/iteration_" << iteration - 1 << ".ppm";
				saveImageToDisk(imgOut.str(), imgData, imgWidth, imgHeight);
				cudaFree(d_imgData);
				delete[] imgData;
			}

			numBodies = newNumBodies;
			bData.freeData();
			bData = newData;

			//Starting generateImage kernel
			if (iteration % imageEveryIteration == 0)
			{
				imgData = new char[imageSize];
				bData.uploadToDevice();
				cudaMalloc((void**) &d_imgData, imageSize);
				cudaMemsetAsync(d_imgData, 254, imageSize, imageStream);
				generateImage<<<blocks, threadsPerBlock, threadsPerBlock * (sizeof(Vec2f) + sizeof(float)), imageStream>>>
						(bData.d_contiguousData, bData.numBodies, d_imgData, imgWidth, imgHeight, fieldWidth, fieldHeight);
				cudaMemcpyAsync(imgData, d_imgData, imageSize, cudaMemcpyDeviceToHost, imageStream);

			}
			//Cleanup
			cudaFree(d_updatedMasses);
			cudaFree(d_updatedRadii);
			updatedMasses.clear();
			updatedRadii.clear();
		}
			CUDA_SYNC_CHECK();
			cudaDeviceReset();
			printf("Time taken: %.4f\n", jbutil::gettime() - startTime);
	}
	return 0;
}
