#include "jbutil.h"
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>
#include <math.h>


#include "vec2f.h"

//#define CHUNK_SIZE 8
#define MAX_RUNS 1

/*
 * Constant definitions for field dimensions, and particle masses
 */
const int fieldWidth = 	5000;
const int doubleFieldWidth = fieldWidth << 1;
//const int fieldHalfWidth = fieldWidth >> 1;
const int fieldHeight = 5000;
//const int fieldHalfHeight = fieldHeight >> 1;
const int doubleFieldHeight = fieldHeight << 1;

const float Grav_constant = 20.f;

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
	double	Mass; //the mass of the particle will be treated as radius

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

	Particle(Vec2f p, Vec2f v, float m) 
	: Position(p)
	, Velocity(v)
	, Mass (m)
	{ }

	bool intersect(const Particle& p)
	{
		Vec2f direction = p.Position - Position;
		float distance = (direction.X * direction.X) + (direction.Y * direction.Y);
		return distance < (Mass + p.Mass) *  (Mass + p.Mass);
	}
};

/*
 * Compute forces of particles exerted on one another
 */
void ComputeForces(std::vector<Particle> &bodies, std::vector<float> &updated_masses ,float timestep)
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

			//Testing for body - body collision
			if( (p1.Mass >= p2.Mass) && p1.intersect(p2))
			{
				updated_masses[j] = p1.Mass + p2.Mass;
				continue;
			}
			else if((p1.Mass < p2.Mass) && p1.intersect(p2))
			{
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
		}
		// Compute acceleration for body 
		acceleration = force * Grav_constant;
		p1.Velocity += acceleration * timestep;

		//Border collision
		if(p1.Position.X + (acceleration.X * timestep) > fieldWidth - p1.Mass || p1.Position.X + (acceleration.X * timestep) < -fieldWidth + p1.Mass)
			p1.Velocity.X *= -1;
		if(p1.Position.Y + (acceleration.Y * timestep) > fieldHeight - p1.Mass || p1.Position.Y + (acceleration.Y * timestep) < -fieldHeight + p1.Mass)
			p1.Velocity.Y *= -1;
	}
}

/*
 * Update particle positions
 */

void MoveBodies(std::vector<Particle> &p_bodies, std::vector<float> &updated_masses, float p_deltaT)
{
	for (size_t j = 0; j < p_bodies.size(); ++j)
	{
		p_bodies[j].Mass = updated_masses[j];
		if(p_bodies[j].Mass > 100.f) p_bodies[j].Mass = 100.f; //capping the maximum mass for display purposes
		p_bodies[j].Position += p_bodies[j].Velocity * p_deltaT;
	}
}

void SaveIterationImage(const std::string &filename, const std::vector<Particle> &bodies)
{
	const int img_width = 1024;
	const int img_height = 900;
	jbutil::matrix<int> img_data;
	img_data.resize(img_height, img_width);
	// img_data_g.resize(img_height, img_width);
	// img_data_b.resize(img_height, img_width);

	//Initializing matrix
	for(int j = 0; j < img_width; ++j)
		for(int i = 0; i < img_height; ++i)
		{
			img_data(i, j) = 254;
			//img_data_g(i, j) = 255;
			//img_data_b(i, j) = 255;
		}

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
		int y_min = y_centre_pixel-bodies[i].Mass < 0 ? 0  : y_centre_pixel-bodies[i].Mass; 
		int y_max = y_centre_pixel+bodies[i].Mass >= img_height ? img_height  : y_centre_pixel + bodies[i].Mass; 
		int x_min = x_centre_pixel-bodies[i].Mass < 0 ? 0  : x_centre_pixel - bodies[i].Mass; 
		int x_max = x_centre_pixel+bodies[i].Mass > img_width ? img_width : x_centre_pixel + bodies[i].Mass; 
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
        		if(x_sq + y_sq <= roundf(bodies[i].Mass * bodies[i].Mass))
				{
#ifndef NDEBUG
					assert((x >= 0) && (x <= img_width));
					assert((y >= 0) && (y <= img_height));
#endif
					img_data(y, x) = 0;
					//img_data_g(y, x) = 0;
					//img_data_b(y, x) = 0;
				};
			}
		}
	}
	jbutil::image<int> img(img_height, img_width,1, 255);
	img.set_channel(0, img_data);
	//img.set_channel(1, img_data_g);
	//img.set_channel(2, img_data_b);

	std::ofstream img_out(filename.c_str(), std::ofstream::binary);
	if (img_out.is_open())
	{	
		img.save(img_out);
		img_out.close();
	}
	else
		std::cerr << "Error writing image to file:" << filename << std::endl <<"Ensure the the folder exists" <<std::endl;
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

	const float minBodyMass = 2.5f;
	const float maxBodyMass = 5.f;

	std::stringstream fileOutput;
	std::stringstream imgOut;
	std::vector<Particle> bodies;
	std::vector<float> updatedMasses;

	jbutil::randgen gen;
	gen.seed(42);

	Particle p;
	float x, y, m;

	//Randomly generating bodies
	for (int bodyIndex = 0; bodyIndex < particleCount; ++bodyIndex)
	{
		x = gen.fval(0, doubleFieldWidth) - fieldWidth;
		y = gen.fval(0, doubleFieldHeight) - fieldHeight;
		m = gen.fval(minBodyMass, maxBodyMass);
		p = Particle(Vec2f(x,y), m);
		bodies.push_back(p);
		updatedMasses.push_back(p.Mass);
	}

	// bodies.push_back(Particle(Vec2f(0.f,0.f), 50));
	// updatedMasses.push_back(50.f);

	for(int run = 0; run < MAX_RUNS; ++run)
	{		
		for (int iteration = 0; iteration < maxIteration; ++iteration)
		{
			ComputeForces(bodies, updatedMasses, timestep);
			MoveBodies(bodies, updatedMasses, timestep);
			if(iteration % imageEveryIteration == 0)
			{
				imgOut.str(std::string());
				imgOut << argv[4]<<"/iteration_" << iteration << ".ppm";
				SaveIterationImage(imgOut.str(), bodies);
			}
		}
	}
	return 0;
}