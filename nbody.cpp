//#include "jbutil.h"
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <math.h>
#include <random>


//#include "vec2.h"
#include "vec2f.h"


//#define CHUNK_SIZE 8
#define MAX_RUNS 1

/*
 * Constant definitions for field dimensions, and particle masses
 */
const int fieldWidth = 	2000;
const int doubleFieldWidth = fieldWidth << 1;
//const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 2000;
//const int fieldHalfHeight = fieldHeight >> 1;
const int doubleFieldHeight = fieldHeight << 1;

const float Grav_constant = 6.67408e-11;//20.f;

typedef struct rgb_data{
	int r;
	int g;
	int b;
} RGB;

/*
 * Particle structure
 */
struct Particle
{
	Vec2f Position;
	Vec2f Velocity;
	double	Mass;
	float Radius;

	Particle(void) 
		: Position(Vec2f(0,0))
		, Velocity( 0.f, 0.f )
		, Mass ( 0.f )
	{ }

	Particle(Vec2f p, double m) 
	: Position(p)
	, Velocity( 0.0, 0.0 )
	, Mass (m)
	{ }

	Particle(Vec2f p, Vec2f v, float m, float radius) 
	: Position(p)
	, Velocity(v)
	, Mass (m)
	, Radius(radius)
	{ }

	bool intersect(const Particle& p)
	{
		Vec2f direction = p.Position - Position;
		float distance = (direction.X * direction.X) + (direction.Y * direction.Y);
		return distance < (Radius + p.Radius) *  (Radius + p.Radius);
	}
};

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForces(std::vector<Particle> &bodies, std::vector<float> &updated_masses, float timestep)
{
	Vec2f direction, force, acceleration;
	float distance;

	for (size_t j = 0; j < bodies.size(); ++j)
	{
		Particle &p1 = bodies[j];
		if(p1.Mass == 0.f) continue; //deleted
		force = 0.f, acceleration = 0.f;
		for (size_t k = 0; k < bodies.size(); ++k)
		{
			if (k == j) continue;		

			Particle &p2 = bodies[k];
			if(p2.Mass == 0.f) continue; //check if deleted

			bool intersect = p1.intersect(p2);

			//Testing for body - body collision
			if( (p1.Radius >= p2.Radius) && intersect)
			{
				updated_masses[j] = p1.Mass + p2.Mass;
				p1.Radius += p2.Radius;
				//printf("INCREASED (p%d)\n", (int)j);
				continue;
			}
			else if((p1.Radius < p2.Radius) && intersect)
			{
				updated_masses[j] = 0.f;
				//printf("DELETED (p%d)\n", (int)j);
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

		}
		//printf("Force (p%d) (total) : %.2f, %.2f\n", (int)j, force.X, force.Y);
		// Compute acceleration for body 
		acceleration = force * Grav_constant;
		//printf("Acceleration (p%d): %0.20f, %0.20f\n", (int)j, acceleration.X, acceleration.Y);
		p1.Velocity += acceleration * timestep;
		//printf("Velocity (p%d): %.2f, %.2f\n", (int)j, p1.Velocity.X, p1.Velocity.Y);
		//Border collision
		if(p1.Position.X + (acceleration.X * timestep) > fieldWidth - p1.Radius || p1.Position.X + (acceleration.X * timestep) < -fieldWidth + p1.Radius)
			p1.Velocity.X *= -1;
		if(p1.Position.Y + (acceleration.Y * timestep) > fieldHeight - p1.Radius || p1.Position.Y + (acceleration.Y * timestep) < -fieldHeight + p1.Radius)
			p1.Velocity.Y *= -1;
		//printf("=======================\n");
	}
}

/*
 * Update particle positions
 */

void MoveBodies(std::vector<Particle>& p_bodies, std::vector<Particle>& updatedBodies, std::vector<float> &updated_masses, float p_deltaT)
{
	for (size_t j = 0; j < p_bodies.size(); ++j)
	{
		//p_bodies[j].Mass = updated_masses[j];
		if(updated_masses[j] != 0.f) 
		{
			//printf("UPDATED MASS (p%d): %.2f\n", (int)j, updated_masses[j]);
			p_bodies.at(j).Position += p_bodies.at(j).Velocity * p_deltaT;
			updatedBodies.push_back(p_bodies.at(j));
		}
	}
}

	// std::ofstream ofs;
	// vec3 col;

	// std::cout << "Saving Atlas to disk" << std::endl;
	// ofs.open("./atlas_out.ppm", std::ofstream::out);
	// ofs << "P3\n" << width << " " << height << "\n255\n";
	// for (size_t i = 0; i < (size_t)height*width; ++i) {
	// 	col = data[i];
	// 	int ir = int(255.99 * std::max<float>(0.f, std::min<float>(1.f, col[0])));
	// 	int ig = int(255.99 * std::max<float>(0.f, std::min<float>(1.f, col[1])));
	// 	int ib = int(255.99 * std::max<float>(0.f, std::min<float>(1.f, col[2])));
	// 	ofs << ir << " " << ig << " " << ib << "\n";
	// }
	// ofs.close();
	// std::cout << "Atlas saved on disk" << std::endl;

void SaveIterationImage(const std::string &filename, const std::vector<Particle> &bodies)
{
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

	for(int i=0; i < imageSize	; ++i){
		imgData[i] = (char)254;
	}

	//printf("hi\n");

	//Populating matrix with body positions
	int x_centre_pixel;
	int y_centre_pixel;
	for(int i = 0; i< bodies.size(); ++i)
	{
		if(bodies[i].Mass == 0.f) continue;

		//Since positions can be negative, we add the field width and height to determine the new position in the image ( no negative coords)
		x_centre_pixel = (int)(((bodies[i].Position.X + fieldWidth) / doubleFieldWidth) * img_width);
		y_centre_pixel = (int)(((bodies[i].Position.Y + fieldHeight) / doubleFieldHeight) * img_height);

		int x_sq;
		int y_sq;
		int y_min = y_centre_pixel-bodies[i].Radius < 0 ? 0  : y_centre_pixel-bodies[i].Radius; 
		int y_max = y_centre_pixel+bodies[i].Radius >= img_height ? img_height  : y_centre_pixel + bodies[i].Radius; 
		int x_min = x_centre_pixel-bodies[i].Radius < 0 ? 0  : x_centre_pixel - bodies[i].Radius; 
		int x_max = x_centre_pixel+bodies[i].Radius > img_width ? img_width : x_centre_pixel + bodies[i].Radius; 
		//Brute forcing every pixel and checking whether it is in body's radius. 
		for(int y= y_min; y < y_max; ++y)
		{
			if(y < 0)y = 0;
			if(y > img_height) y = img_height;
    		for(int x=x_min; x < x_max; ++x){
				if(x < 0)x=0;
				if(x > img_width) x = img_width;
				x_sq = (x-x_centre_pixel)*(x-x_centre_pixel);
				y_sq = (y-y_centre_pixel)*(y-y_centre_pixel);
				//std::cout<<"x after: "<< x <<std::endl;
        		if(x_sq + y_sq <= roundf(bodies[i].Radius * bodies[i].Radius))
				{
#ifndef NDEBUG
					assert((x >= 0) && (x <= img_width));
					assert((y >= 0) && (y <= img_height));
#endif
					//img_data(y, x) = 0;
					imgData[(img_width * y) + x] = (char)0;//static_cast<char>(0);
				};
			}
		}
	}
//	jbutil::image<int> img(img_height, img_width,1, 255);
//	img.set_channel(0, img_data);


	std::ofstream outImg;
	outImg.open(filename, std::ofstream::out);
	std::cout << "Saving iteration to disk" << std::endl;
	if (outImg.is_open())
	{	
		outImg << "P5\n" << img_width << " " << img_height << "\n255\n";//<< "\n255\n";
		for(int i= 0; i < imageSize ; ++i)
		{
			//printf("%d\n", imgData[i]);
			outImg << imgData[i];
			
		}
		//img.save(img_out);
		outImg.close();
	}
	else
		std::cerr << "Error writing image to file:" << filename << std::endl <<"Ensure the the folder exists" <<std::endl;

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

int main(int argc, char **argv)
{
	if(argc < 5){
		std::cerr<<"Incorrect arguments. <particle count> <interations> <save-image-every-x-iteration> <image-path>"<<std::endl;
		exit(0);
	}

	const int particleCount = std::stoi(argv[1]);
	const int maxIteration = std::stoi(argv[2]);;
	const int imageEveryIteration = std::stoi(argv[3]);
	const float timestep = 0.1f;

	const float minBodyMass = 1e3f;
	const float maxBodyMass = 1e7f;

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
	std::uniform_real_distribution<float> radiusDist(5, 15);
    distribution(generator);

	//Randomly generating bodies
	for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	{
		x = (distribution(generator) * doubleFieldWidth) - fieldWidth; //gen.fval(0, doubleFieldWidth) - fieldWidth;
		y = (distribution(generator) * doubleFieldHeight) - fieldHeight;
		m = massDist(generator);
		r = radiusDist(generator);//gen.fval(5, 50);
		printf("Generated Particle:\nPos: (%.4f,%.4f)\nMass: %.4f\nRadius: %.4f\n", x, y, m, r);
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
	// Particle p1(Vec2f(-500,0), Vec2f(0,10), 50, 10);
	// Particle p2(Vec2f(500,0), Vec2f(0,-10), 50, 10);
	// Particle p3(Vec2f(0,0), Vec2f(0,0), 1e15, 18);
	// bodies.push_back(p1);
	// bodies.push_back(p2);
	// bodies.push_back(p3);
	//=======================
	//updatedMasses.push_back(p1.Mass);
	//updatedMasses.push_back(p2.Mass);
	//updatedMasses.push_back(p3.Mass);
	// 	updatedMasses.push_back(p.Mass);


	// bodies.push_back(Particle(Vec2f(0.f,0.f), 50));
	// updatedMasses.push_back(50.f);

	for(int run = 0; run < MAX_RUNS; ++run)
	{
		for (int iteration = 0; iteration < maxIteration; ++iteration)
		{
			std::vector<Particle> newBodies;
			std::vector<float> updatedMasses(bodies.size());
			for(int i = 0; i < bodies.size(); ++i)
			{
				updatedMasses[i] = bodies[i].Mass;
				//printf("Mass: %.4f \nRadius: %.4f\n", body.Mass, body.Radius);
			}		
			ComputeForces(bodies, updatedMasses, timestep);
			MoveBodies(bodies, newBodies, updatedMasses, timestep);
			bodies = newBodies;
			if(iteration % imageEveryIteration == 0)
			{
				imgOut.str(std::string());

				imgOut << argv[4]<<"/iteration_" << iteration << ".ppm";
				printf("Saving Iteration %d\n", iteration);
				SaveIterationImage(imgOut.str(), bodies);
				}
			//printf("Iteration: %d\n", iteration);
			//printf("++++++++++++\n");
		}
	}
	return 0;
}