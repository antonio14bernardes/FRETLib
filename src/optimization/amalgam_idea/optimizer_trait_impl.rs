use nalgebra::DVector;
use rand::thread_rng;


use crate::optimization::optimizer::{FitnessFunction, OptimizationFitness};

use super::{AmalgamIdea, AmalgamIdeaError};

use super::super::optimizer::Optimizer;
use super::super::set_of_var_subsets::*;
use super::amalgam_parameters::*;
use super::super::multivariate_gaussian::CovMatrixType;
use super::super::tools::select_top_n;


impl<F, Fitness> Optimizer<f64, Fitness> for AmalgamIdea<F, Fitness> 
where 
F: FitnessFunction<f64, Fitness>,
Fitness: OptimizationFitness
{

    type Error = AmalgamIdeaError;

    // Evaluate the fitness of a single solution
    fn evaluate(&self, solution: &Vec<f64>) -> Result<Fitness, AmalgamIdeaError> {
        let fitness_function = self
            .fitness_function
            .as_ref()
            .ok_or(AmalgamIdeaError::FitnessFunctionNotSet)?;
        
        Ok(fitness_function.evaluate(&solution))

    }

    // Initialize the population
    fn initialize(&mut self) -> Result<(), AmalgamIdeaError>{

        // Check if fitness function has been set
        let _ = self.fitness_function.as_ref().ok_or(AmalgamIdeaError::FitnessFunctionNotSet)?;

        
        // Check if subsets have been initialized. If not, assume no subset division.
        if self.subsets.is_none() {
            let mut subsets = 
            SetVarSubsets::<f64>::new_empty(vec![(0..self.problem_size).collect()])
            .map_err(|err| AmalgamIdeaError::SubsetError {err})?;

            if let Some(population) = &self.initial_population {
                subsets.set_population(population.clone()).map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            }

            self.subsets = Some(subsets);
        }
        
        // Check if parameters have been manually set. If not, set them up with auto
        if self.parameters.is_none() {
            let factorized = self.subsets.as_ref().unwrap().get_subset_indices().len() == 1;
            let prob_size = self.problem_size;
            let memory = self.iter_memory;

            // If there is any full subset cov matrix, compute parameters for full cov matrix
            let mut cov_mat_type = CovMatrixType::Diagonal;
            for subset in self.subsets.as_ref().unwrap().get_subsets() {
                if subset.get_covariance_matrix_type() == Some(&CovMatrixType::Full) {
                    cov_mat_type = CovMatrixType::Full;
                    break;
                }
            }

            let new_params = AmalgamIdeaParameters::new_auto(prob_size, &cov_mat_type, factorized, memory);

            self.parameters = Some(new_params);
        }

        // Check if population size has been manually set
        if let Some(pop_size) = self.manual_pop_size {
            let params = self.parameters.as_mut().unwrap();
            params.population_size = pop_size;
        }
        
        // Check if population has been initialized.
        // If not, sample from manually set distribution or do random init from the subsets
        if self.initial_population.is_none() {

            let pop_size = self.get_pop_size().unwrap();

            let subsets = self.subsets.as_mut().unwrap();

            let mut rng = thread_rng(); // Use this function for random number generation

            if self.init_with_manual_distribution {
                subsets.initialize_population(pop_size, &mut rng)
                .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            } else {
                subsets.initialize_random_population(pop_size, &mut rng, None)
                .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            }

            self.initial_population = Some(subsets.get_population());
        }

        // Initialize the distributions in the subsets
        self.subsets.as_mut().unwrap().compute_distributions()
        .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;

        // Set the current pop and fitness fields
        self.current_population = self.initial_population.as_ref().unwrap().clone();
        self.current_fitnesses =  self.evaluate_population()?;

        // Allocate space for the prev mean shifts
        let mut init_mean_shift = Vec::<DVector<f64>>::new();
        for indices in self.get_subsets().unwrap().get_subset_indices() {
            let num_vars = indices.len();
            init_mean_shift.push(DVector::zeros(num_vars));
        }
        self.current_mean_shifts = init_mean_shift;

        // Set best current individual
        let best_solution = select_top_n(self.initial_population.as_ref().unwrap(), &self.current_fitnesses, 1);
        self.best_solution = Some((best_solution.0[0].clone(), best_solution.1[0].clone()));

        // Initialize c_mult
        self.c_mult = Some(1.0);

        Ok(())
    }

    // Execute a single optimization step
    fn step(&mut self) -> Result<(), AmalgamIdeaError> {
        // Perform selection
        let selection = self.selection()?;

        // Update distribution based on selection
        let (new_means, _new_covs) = self.update_distribution(selection)?;

        // Sample population and return the improved individuals for the the parameter update method
        let improved_individuals = self.update_population()?;

        // Update cmult and number of stagnant iterations
        self.update_parameters(improved_individuals, new_means)?;

        Ok(())
    }

    // Run the optimization process
    fn run(&mut self) -> Result<(), AmalgamIdeaError>  {
        
        self.initialize()?;

        let c_mult_min = self.parameters.as_ref().unwrap().c_mult_min;

        let mut iters = 0;

        let termination = 
        |max_iters_option:Option<usize>, iters: &mut usize, c_mult_min:f64, c_mult_option:Option<f64>| {
            *iters = *iters + 1;
            let max_iter_reach = 
            if let Some(max_iter) = max_iters_option {
                *iters > max_iter
            } else {false};

            let c_mult_min_reach = c_mult_option.unwrap() < c_mult_min;

            max_iter_reach || c_mult_min_reach
        };

        while !termination(self.max_iterations, &mut iters, c_mult_min, self.c_mult) {
            
            self.step()?;

            let new_fit = self.best_solution.as_ref().unwrap().1.clone();

            self.best_fitnesses.push(new_fit);

            let (best_sol, fit) = self.get_best_solution().unwrap();
            if iters % 20 == 0 {
                println!("In iteration {}", iters);
                println!("Best solution: {:?}. With fitness: {:?}", best_sol, fit);
            }            
        }

        Ok(())
    }

    // Retrieve the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&(Vec<f64>, Fitness)> {
        self.best_solution.as_ref() 
    }
}