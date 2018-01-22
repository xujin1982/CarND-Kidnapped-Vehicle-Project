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

// Create random engine
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    Particle temp_particle;

    // Set the number of particles
    num_particles = 100;

    // Create normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Initialize all particles
    for(int i=0;i<num_particles;i++){
        temp_particle.id = i;
        temp_particle.x = dist_x(gen);
        temp_particle.y = dist_y(gen);
        temp_particle.theta = dist_theta(gen);
        temp_particle.weight = 1.0;
        particles.push_back(temp_particle);

        //cout<<temp_particle.id<<"\t"<<temp_particle.x<<"\t"<<temp_particle.y<<"\t"<<temp_particle.theta<<"\t"<<temp_particle.weight<<"\t"<<endl;
    }

    // Finish Initialization
    is_initialized = true;

    cout<<"init done!"<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    double new_theta, temp_variable;

    // add random Gaussian noise to x, y and theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i=0;i<num_particles;i++){
        if(fabs(yaw_rate) < 0.000001){
            temp_variable = velocity * delta_t;
            particles[i].x += temp_variable * cos(particles[i].theta);
            particles[i].y += temp_variable * sin(particles[i].theta);
        }
        else{
            new_theta = particles[i].theta + yaw_rate * delta_t;
            temp_variable = velocity / yaw_rate;
            particles[i].x += temp_variable * (sin(new_theta) - sin(particles[i].theta));
            particles[i].y += temp_variable * (cos(particles[i].theta) - cos(new_theta));
            particles[i].theta = new_theta;

            // add noise
            particles[i].x += dist_x(gen);
            particles[i].y += dist_y(gen);
            particles[i].theta += dist_theta(gen);
        }

        // reset particle id
        particles[i].id = i;

        //cout<<particles[i].id<<"\t"<<particles[i].x<<"\t"<<particles[i].y<<"\t"<<particles[i].theta<<"\t"<<particles[i].weight<<"\t"<<endl;

    }
    //cout<<"prediction: "<<i<<" done!"<<endl;


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    double min_dist, x_obs, y_obs, x_landmark, y_landmark, temp_dist;
	int land_id;

    for(int i=0;i<observations.size();i++){
        // init minimum distance
        min_dist = numeric_limits<double>::max();

        // init landmark id
        land_id = -1;

        x_obs = observations[i].x;
        y_obs = observations[i].y;

        for(int j=0;j<predicted.size();j++){
            x_landmark = predicted[j].x;
            y_landmark = predicted[j].y;

            temp_dist = dist(x_obs, y_obs, x_landmark, y_landmark);

            if(temp_dist < min_dist){
                min_dist = temp_dist;
                land_id = predicted[j].id;
            }
        }

        // set the observation's id by using a nearest-neighbors data association
        observations[i].id = land_id;
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

	double x_part, y_part, theta_part, x_landmark, y_landmark, temp_dist, x_obs, y_obs, x_map, y_map, sum_weight;
    int id_map;
    //int k, size_predicted;
    //bool flag;

    for(int i=0;i<num_particles;i++){
        x_part = particles[i].x;
        y_part = particles[i].y;
        theta_part = particles[i].theta;

        // create a vector for landmarks within sensor range
        vector<LandmarkObs> predicted;

        for(int j=0;j<map_landmarks.landmark_list.size();j++){
            x_landmark = map_landmarks.landmark_list[j].x_f;
            y_landmark = map_landmarks.landmark_list[j].y_f;
            temp_dist = dist(x_part, y_part, x_landmark, y_landmark);

            if(temp_dist <= sensor_range){
                predicted.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, x_landmark, y_landmark});
            }
        }

        // create a vector for observations from vehicle coordinate to map coordinate
        vector<LandmarkObs> transform_ob;

        for(int j=0;j<observations.size();j++){
            x_obs = observations[j].x;
            y_obs = observations[j].y;

            // Transformation of observations
            x_map = x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs);
            y_map = y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs);

            transform_ob.push_back(LandmarkObs{observations[i].id, x_map, y_map});
        }

        dataAssociation(predicted,transform_ob);

        // Calculate particle weight
        particles[i].weight = 1.0;

        for(int j=0;j<transform_ob.size();j++){
            x_map = transform_ob[j].x;
            y_map = transform_ob[j].y;
            id_map = transform_ob[j].id;

            ///*
            for(int k=0;k<predicted.size();k++){
                if(predicted[k].id == id_map){
                    x_landmark = predicted[k].x;
                    y_landmark = predicted[k].y;
                    break;
                }
            }
            //*/
            /*
            flag = false;
            k=0;
            size_predicted = predicted.size();
            while(!flag && k<size_predicted){
                if(predicted[k].id == id_map){
                    flag = true;
                    x_landmark = predicted[k].x;
                    y_landmark = predicted[k].y;
                }
                k++;
            }
            */

            particles[i].weight *= exp(-(pow((x_map-x_landmark),2.0)/(2*pow(std_landmark[0],2.0))+pow((y_map-y_landmark),2.0)/(2*pow(std_landmark[1],2.0))))/(2*M_PI*std_landmark[0]*std_landmark[1]);
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	vector<double> weights;
	int index;
	double w_max, beta;


	w_max = numeric_limits<double>::min();
	for(int i=0;i<num_particles;i++){
        if(w_max < particles[i].weight){
            w_max = particles[i].weight;
        }
        weights.push_back(particles[i].weight);
	}

	uniform_int_distribution<int> int_dist(0, num_particles-1);
    index = int_dist(gen);

    uniform_real_distribution<double> real_dist(0, 2*w_max);
    beta = 0.0;

    for(int i=0;i<num_particles;i++){
        beta += real_dist(gen);
        while(weights[index] < beta){
            beta -= weights[index];
            index = (index +1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

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
