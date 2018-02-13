/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//
	//	Similar to what we have done in initialization lesson of the PF
	//

	// Initialize number of particles
	num_particles = 1000;

	// RND-Engine
	default_random_engine gen;

	// Normal distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles);
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		// Init weight
		weights[i] = 1.0;
		// Init particles
		Particle &p = particles[i];
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = weights[i];
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// RND-Engine
	default_random_engine gen;

	// Normal distribution for x, y and theta
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	if (abs(yaw_rate) < 0.00000001)
	{
		for (int i = 0; i<num_particles; i++)
		{
			// Process model
			Particle &p = particles[i];
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
			// Process noise
			p.x += dist_x(gen);
			p.y += dist_y(gen);
			p.theta += dist_theta(gen);
		}
	}
	else
	{
		for (int i = 0; i<num_particles; i++)
		{
			// Process model
			Particle &p = particles[i];
			p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
			// Process noise
			p.x += dist_x(gen);
			p.y += dist_y(gen);
			p.theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, const std::vector<LandmarkObs>& observations, std::vector<int>& association_index) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	//
	// NOTE: Method is called during updateWeights
	//

	// parameter association_index
	// carries the observation index which is associated with the i - th predicted. 
	// Initialized with - 1, which denotes missing association
	association_index.clear();
	association_index.resize(predicted.size(), -1);

	for (unsigned p = 0; p < predicted.size(); p++)
	{
		// Init minimum distance
		double min_dist = 10000000.0;

		for (unsigned o = 0; o < observations.size(); o++)
		{
			double dist =	(predicted[p].x - observations[o].x) * (predicted[p].x - observations[o].x) +
							(predicted[p].y - observations[o].y) * (predicted[p].y - observations[o].y);

			if (dist < min_dist)
			{
				min_dist = dist;
				association_index[p] = o;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Iterate over particles
	for (int i = 0; i < num_particles; i++)
	{
		// Predict measurements
		Particle &p = particles[i];
		std::vector<LandmarkObs> predicted;
		unsigned num_landmarks = map_landmarks.landmark_list.size();
		predicted.reserve(num_landmarks);

		for (unsigned n = 0; n < num_landmarks; n++)
		{
			const Map::single_landmark_s &landmark = map_landmarks.landmark_list[n];

			// Translation in world coordinates
			double dx_world = landmark.x_f - p.x;
			double dy_world = landmark.y_f - p.y;
			// Landmarks in sensor range
			if (sqrt(dx_world * dx_world + dy_world * dy_world) < sensor_range)
			{
				LandmarkObs pred;
				pred.id = landmark.id_i;
				// Rotate translation
				pred.x = cos(p.theta) * dx_world + sin(p.theta) * dy_world;
				pred.y = -sin(p.theta) * dx_world + cos(p.theta) * dy_world;

				predicted.push_back(pred);
			}
		}

		// Data association
		std::vector<int> association_index;
		dataAssociation(predicted, observations, association_index);

		// Update weightss
		p.weight = 1.0;
		for (unsigned n = 0; n<predicted.size(); n++)
		{
			const LandmarkObs &pred = predicted[n];
			double dx(sensor_range), dy(sensor_range);
			if (association_index[n] != -1)
			{
				const LandmarkObs &obs = observations[association_index[n]];
				dx = pred.x - obs.x;
				dy = pred.y - obs.y;
			}
			else
			{
				cout << "No Landmark association for [ " << pred.id << " ]" << endl;
			}

			double arg = (dx * dx) / (std_landmark[0] * std_landmark[0]) + (dy * dy) / (std_landmark[1] * std_landmark[1]);
			arg *= -0.5;
			double probab = exp(arg) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
			p.weight *= probab;
		}
		weights[i] = p.weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// RND-Engine
	default_random_engine gen;
	// Distribution according to weights
	discrete_distribution<int> dist(weights.begin(), weights.end());

	std::vector<Particle> res_particles;
	res_particles.reserve(num_particles);
	// Resampling
	for (int i = 0; i < num_particles; i++)
	{
		int res_idx = dist(gen);
		res_particles.push_back(particles[res_idx]);
	}
	// Use resampled particles 
	particles = res_particles;
	for (int i = 0; i < num_particles; i++)
	{
		weights[i] = particles[i].weight;
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
