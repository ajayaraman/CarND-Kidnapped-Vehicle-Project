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

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 120;

	// This line creates a normal (Gaussian) distribution 
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.x = dist_x(gen);
		// where "gen" is the random engine initialized earlier (line 18).
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);

		weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < num_particles; i++)
	{
		double x, y, theta;

		if ( fabs(yaw_rate) < 0.0001){
			x = particles[i].x + velocity * delta_t * cos( particles[i].theta );
			y = particles[i].y + velocity * delta_t * sin( particles[i].theta );
			theta = particles[i].theta;
		}
		else{
			x = particles[i].x + (velocity/yaw_rate) * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			y = particles[i].y + (velocity/yaw_rate) * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			theta = particles[i].theta + yaw_rate * delta_t;
		}
		//create a normal (Gaussian) distribution for x.
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i =0; i< observations.size(); i++)
	{
		LandmarkObs o = observations[i];

		double mindist = 1e20; //some large value outside of map range

		// id of association
		int map_id = -1;

		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs p = predicted[j];

			double d = dist(o.x,o.y,p.x, p.y);

			if ( d < mindist)
			{
				mindist = d;
				map_id = p.id;
			}
		}

		// now set the observation id to be the id of the nearest prediction landmark
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	//At the end we will recompute the weights
	weights.clear();

	//For each particle do
	for (int p = 0; p < num_particles; p++)
	{
		double px = particles[p].x;
		double py = particles[p].y;
		double ptheta = particles[p].theta;

		// create a vector of predicted landmark locations within sensor range
		vector<LandmarkObs> predictions;
		
		// Filter landmarks that must be put into the predictions list based on feasibility from sensor range
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double lx = map_landmarks.landmark_list[j].x_f;
			double ly = map_landmarks.landmark_list[j].y_f;
			int lid = map_landmarks.landmark_list[j].id_i;

			if ( dist(lx, ly, px, py) <= sensor_range )
			{
				LandmarkObs o = {lid, lx, ly};
				predictions.push_back(o);
			}
		}

		//Transform observed landmarks to map coordinates
		vector<LandmarkObs> translated_obs;
		LandmarkObs obs;
		for (int i = 0; i < observations.size(); i++)
		{
			LandmarkObs tobs;
			obs = observations[i];

			tobs.x = px + (obs.x * cos( ptheta) -  obs.y * sin( ptheta));
			tobs.y = py + (obs.x * sin( ptheta) +  obs.y * cos( ptheta));
			translated_obs.push_back(tobs);
		}

		// Do data association
		dataAssociation(predictions, translated_obs);

		// compute particle's new weights
		particles[p].weight = 1.0;

		for( int t = 0; t < translated_obs.size(); t++)
		{
			double x, y, ux, uy;
			x = translated_obs[t].x;
			y = translated_obs[t].y;
			int map_tid = translated_obs[t].id;

			//find associated map location ux, uy by using the map id of the translated observations
			//from our dataAssociation step
			for(int pred = 0; pred < predictions.size(); pred++)
			{
				if(predictions[pred].id == map_tid)
				{
					ux = predictions[pred].x;
					uy = predictions[pred].y;
					break;
				}
			}

			//Use multivariate gaussian to compute weight associated with this particle
			double sx = std_landmark[0];
			double sy = std_landmark[1];
			double tobs_w = (1 / (2 * M_PI * sx * sy)) * exp( -0.5 * ( pow(x - ux, 2)/(sx * sx) + pow(y-uy,2)/(sy * sy) ) );

			//Multiply all such observation weights

			particles[p].weight *= tobs_w;
		}

		//keep track of and update the weights array because this is what is used in the resampling step
		weights.push_back(particles[p].weight);
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	discrete_distribution<int> dist(weights.begin(), weights.end());

	vector<Particle> resampled_particles;

	for(int i = 0; i < num_particles; i++)
	{
		//Sample a particle from the discrete distribution
		resampled_particles.push_back(particles[dist(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
