// This is the main DLL file.
/*
*  This is a PSO algorithm adapted from an ES
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <SharkDefs.h>
#include <EALib/PopulationT.h>
#include <Array/Array.h>
#define Dimension 20

//=======================================================================
//
// fitness function: sphere model
//
double sphere(const std::vector< double >& x)
{
	unsigned i;
	double   sum;
	for (sum = 0., i = 0; i < x.size(); i++)
		sum += Shark::sqr(x[ i ]);
	return sum;
}

Array<double> getDesignVariable(const std::vector<double>& x)
{
	unsigned i;
	Array<double> y(Dimension);
	for (i = 0; i < x.size(); i++)
		y(i) = x [i];
	return y;
}

int rand_num(int n)
{
  int r = rand();
  if(r<0)r=-r;
  return r%n;
}
//=======================================================================
//
// main progra
//
//=======================================================================
//
// main program
//
int main(int argc, char **argv)
{
	//
	// constants
	//
	const unsigned PopulationSize       = 50;
	const unsigned Iterations   = 400;
	const unsigned Interval     = 10;

	const double   MinInit      = -3;
	const double   MaxInit      = + 5;
	const double   VMaxInit    = 1.5;
	const double   VMinInit    = 0.5;

	const bool     PlusStrategy = false;

	unsigned       i, j, t;

	double gbestF;
	Array<double> gbestX(Dimension), gbestY(Dimension);
	Array<double> lbestF(PopulationSize);
	Array<double> lbestX(PopulationSize,Dimension), lbestY(PopulationSize, Dimension);

	Array<double> tempX(Dimension), tempY(Dimension);

	//fp = fopen("result_PSO.dat","w"); 

	// initialize random number generator
	//
	Rng::seed(argc > 1 ? atoi(argv[ 1 ]) : 1234);

	//
	// define populations
	//
	PopulationT<double> parents(PopulationSize,     ChromosomeT< double >(Dimension), // position of the particles
					   ChromosomeT< double >(Dimension)); // velocity of the particles
	PopulationT<double> offsprings(PopulationSize, ChromosomeT< double >(Dimension),
						  ChromosomeT< double >(Dimension));

	//
	// minimization task
	//
	parents   .setMinimize();
	offsprings.setMinimize();

	//
	// initialize parent population
	//
	for (i = 0; i < parents.size(); ++i) {
		parents[ i ][ 0 ].initialize(MinInit,   MaxInit);
		parents[ i ][ 1 ].initialize(VMinInit, VMaxInit);
	}
	//
	// selection parameters (number of elitists)
	//
	for (i = 0; i < parents.size(); ++i)
			parents[ i ].setFitness(sphere(parents[ i ][ 0 ]));
	
	//Get the best fitness

	gbestF = parents.best().fitnessValue();
	
	gbestX = getDesignVariable(parents.best()[0]);
	gbestY = getDesignVariable(parents.best()[1]);
	//fprintf(fp,"Best fitness = %f\n", gbestF);

	for(i = 0; i < parents.size(); ++i){
		lbestF(i) = parents[i].getFitness();
		
		tempX = getDesignVariable(parents[i][0]);
		tempY = getDesignVariable(parents[i][1]);
		for(j=0; j<Dimension;j++){
			lbestX(i,j) = tempX(j);
			lbestY(i,j) = tempY(j);
		}
		
	}
	
	// Parameter setup
	//
	double     wmax = 0.9; // Wmax
	double	   wmin = 0.4; // Wmin
	double c1 = 2;
	double c2 = 2;
	double r1;
	double r2; 

	//
	// iterate
	double gtbestF;
	for (t = 0; t < Iterations; ++t) {
		//
		double w = wmax - (wmax-wmin)*t/(double)Iterations;
	    //fprintf(fp, "weight %f",w);
		// update the global best
        gtbestF = parents.best().fitnessValue();
		if(gtbestF < gbestF){
			gbestF = gtbestF;
			gbestX = getDesignVariable(parents.best()[0]);
			gbestY = getDesignVariable(parents.best()[1]);
		}
		//Update the personal best
		for(i = 0; i < parents.size(); ++i){
			
			double ltF = parents[i].getFitness();
			if(ltF < lbestF(i)){
				lbestF(i) = ltF;
				tempX = getDesignVariable(parents[i][0]);
				tempY = getDesignVariable(parents[i][1]);

				for(j=0; j<Dimension;j++){	
					lbestX(i,j) = tempX(j);
					lbestY(i,j) = tempY(j);
				}
			}
		}


		for (i = 0; i < parents.size(); ++i) {
			//		
			// Update the position and velocity of the particles
			tempX = getDesignVariable(parents[i][0]);
			tempY = getDesignVariable(parents[i][1]);
	
			for(j=0; j<Dimension; j++){
				r1 = rand_num(100)/100.;
				r2 = rand_num(100)/100.;
				//fprintf(fp, "Random numbers %f %f \n", r1, r2); 
				tempY(j) = 0.7*(w*tempY(j) + c1*r1*(lbestX(i,j)-tempX(j)) + c2*r2*(gbestX(j)-tempX(j)));
				tempX(j) = tempX(j) + tempY(j);
			}
			
			for(j=0; j<Dimension; j++){
				offsprings[i][0][j] = tempX(j);
				offsprings[i][1][j] = tempY(j);
			}
		}
		//
		// evaluate objective function (parameters in chromosome #0)
		//
		for (i = 0; i < offsprings.size(); ++i)
			offsprings[ i ].setFitness(sphere(offsprings[ i ][ 0 ]));

		//Pass all offspring to parents (no selection in PSO!!)
		parents = offsprings;
		//
		// print out best value found so far
		//
		if (t % Interval == 0)
			std::cout << t << "\tbest value = " << parents.best().fitnessValue() << std::endl;
		//fprintf(fp,"%d %f %f\n", t, parents.best().fitnessValue(), parents.worst().fitnessValue());
	}

	// lines below are for self-testing this example, please ignore
	if(parents.best().fitnessValue()<1.e-14) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);

}