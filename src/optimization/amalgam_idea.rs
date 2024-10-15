use std::vec;

use rand::thread_rng;

use super::constraints::OptimizationConstraint;
use super::multivariate_gaussian::CovMatrixType;
use super::optimizor::{Optimizor, OptimizorError};
use super::set_of_var_subsets::*;
use super::variable_subsets::*;

pub struct AmalgamIdea<F> 
where F: Fn(&[f64]) -> f64 
{
    problem_size: usize,
    memory: bool,
    parameters: Option<Vec<AmalgamIdeaParameters>>,

    fitness_function: Option<F>,

    subsets: Option<SetVarSubsets<f64>>,

    initial_population: Option<Vec<Vec<f64>>>,
    current_population: Vec<Vec<f64>>,
    best_solution: Option<Vec<f64>>,

    fitnesses: Vec<f64>
}


impl<F> Optimizor<f64> for AmalgamIdea<F>
where
    F: Fn(&[f64]) -> f64,
{
    // Evaluate the fitness of a single solution
    fn evaluate(&self, solution: &Vec<f64>) -> Result<f64, OptimizorError> {
        let fitness_function = self
            .fitness_function
            .as_ref()
            .ok_or(OptimizorError::FitnessFunctionNotSet)?;
        
        Ok(fitness_function(&solution))

    }

    // Initialize the population
    fn initialize(&mut self) -> Result<(), OptimizorError>{
        // Check if subsets have been initialized. If not, assume no subset division.
        if self.subsets.is_none() {
            let subsets = 
            SetVarSubsets::<f64>::new_empty(vec![(0..self.problem_size).collect()])
            .map_err(|err| OptimizorError::SubsetError {err})?;

            self.subsets = Some(subsets);
        }

        // Check if parameters have been manually set. If not, set them up with auto
        let factorized = self.subsets.as_ref().unwrap().get_subset_indices().len() == 1;
        let prob_size = self.problem_size;
        let memory = self.memory;

        let mut parameters: Vec<AmalgamIdeaParameters> = Vec::new();

        for subset in self.subsets.as_ref().unwrap().get_subsets() {
            let cov_mat_type = subset.get_covariance_matrix_type().unwrap_or(&CovMatrixType::Full);

            let new_params = AmalgamIdeaParameters::new_auto(prob_size, cov_mat_type, factorized, memory);

            parameters.push(new_params);
        }



        // Check if population has been initialized. If not, do random init from the subsets
        let pop_size = self.get_pop_size().unwrap();

        if self.initial_population.is_none() {
            let subsets = self.subsets.as_mut().unwrap();
            let mut rng = thread_rng(); // Use thread_rng() for random number generation
            subsets.initialize_random_population(pop_size, &mut rng, None)
            .map_err(|err| OptimizorError::VariableSubsetError { err })?;
        }

        Ok(())
    }

    // Execute a single optimization step
    fn step(&mut self) {
        // Implement logic for a single optimization step
        unimplemented!()
    }

    // Run the optimization process
    fn run(&mut self) {
        // Implement the main optimization loop
        // This will likely call `step` and handle convergence
        unimplemented!()
    }

    // Retrieve the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&Vec<f64>> {
        // Return the current best solution
        // This could either be a single solution or best from the population
        self.best_solution.as_ref() // Assuming `current_values` holds the best so far
    }
}



impl<F> AmalgamIdea<F> where F: Fn(&[f64]) -> f64 {
    pub fn new(problem_size: usize, memory: bool) -> Self {
        Self {
            problem_size,
            memory,

            parameters: None,

            fitness_function: None,

            subsets: None,

            initial_population: None,
            current_population: Vec::new(),
            best_solution: None,

            fitnesses: Vec::new(),
        }
    } 

    pub fn set_parameters(&mut self, parameters: Vec<AmalgamIdeaParameters>) -> Result<(), OptimizorError> {
        let subsets = self.subsets.as_ref()
        .ok_or(OptimizorError::SubsetError { err: VariableSubsetError::SubsetsNotDefined})?;

        if parameters.len() == 1 {
            let repeated_parameters = vec![parameters[0].clone(); subsets.get_subsets().len()];
            self.parameters = Some(repeated_parameters);

        } else if parameters.len() == subsets.get_subsets().len() {
            self.parameters = Some(parameters);

        } else {
            return Err(OptimizorError::IncompatibleParameterSet);
        }
        
        Ok(())
    }
    
    pub fn set_fitness_function(&mut self, fitness_function: F) {
        self.fitness_function = Some(fitness_function);
    }

    pub fn set_initial_population(&mut self, initial_population: Vec<Vec<f64>>) -> Result<(), OptimizorError> {

        if initial_population.iter().any(|ind| ind.len() != self.problem_size) {
            return Err(OptimizorError::PopulationIncompatibleWithProblemSize)
        }

        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_population(initial_population.clone())
            .map_err(|err| OptimizorError::VariableSubsetError { err })?;
        }
        self.initial_population = Some(initial_population);

        Ok(())
    }

    pub fn set_dependency_subsets(&mut self, dependency_subsets: Vec<Vec<usize>>) -> Result<(), OptimizorError> {

        // Check if max index in the subsets is compatible with the defined problem size
        let max_idx = dependency_subsets
        .iter()
        .flat_map(|s| s.iter().cloned())
        .max()
        .ok_or(OptimizorError::SubsetsIncompatibleWithProblemSize)?;

        if max_idx != self.problem_size - 1 {
            return Err(OptimizorError::SubsetsIncompatibleWithProblemSize);
        }


        if let Some(ref init_pop) = self.initial_population {

            let set = SetVarSubsets::new(dependency_subsets, init_pop.clone(), None)
            .map_err(|err| OptimizorError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        } else {

            let set = SetVarSubsets::new_empty(dependency_subsets)
            .map_err(|err| OptimizorError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        }

        Ok(())
    }

    pub fn set_constraints(&mut self, constraints: Vec<OptimizationConstraint<f64>>) -> Result<(), OptimizorError> {
        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_constraints(constraints).map_err(|err| OptimizorError::VariableSubsetError { err })?;
        
        } else {

            if constraints.len() == 1 { // Assume we are dealing with only one subset - all vars related with eachother
                let subset_indices = vec![(0..self.problem_size).collect::<Vec<usize>>()];
                
                self.set_dependency_subsets(subset_indices)?;

                let subsets = self.subsets.as_mut().unwrap();

                subsets.set_constraints(constraints).map_err(|err| OptimizorError::VariableSubsetError { err })?;

            } else {
                return Err(OptimizorError::VariableSubsetsNotDefined);
            }
        }

        Ok(())
    }

    pub fn get_pop_size(&self) -> Option<usize> {
        if let Some(params) = &self.parameters {
            return Some(params[0].population_size)
        }
        None
    }

    pub fn check_readiness(&self) -> Result<(), OptimizorError> {
        let _ = self.fitness_function.as_ref().ok_or(OptimizorError::FitnessFunctionNotSet);
        let _ = self.fitness_function.as_ref().ok_or(OptimizorError::FitnessFunctionNotSet);

        Ok(())
    }

    pub fn evaluate_population(&self, initial_population: Vec<Vec<f64>>) -> Result<Vec<f64>, OptimizorError> {

        let evals: Vec<f64> = initial_population
            .iter()
            .map(|individual| self.evaluate(individual))
            .collect::<Result<Vec<f64>, OptimizorError>>()?;

        Ok(evals)
    }
}


#[derive(Debug, Clone)]
pub struct AmalgamIdeaParameters {

    population_size: usize,

    // Selection parameter
    pub tau: f64, // Fraction of population to be selected for new distribution computation

    // c_mult parameters
    pub c_mult_inc: f64, // Factor to increase c_mult
    pub c_mult_dec: f64, // Factor to decrease c_mult 

    // Memory parameters
    pub eta_cov: f64, // Memory in updating the cov matrix
    pub eta_shift: f64, // Memory in updating the mean

    // Mean shift parameters
    pub alpha_shift: f64, // Fraction of selected individuals to shift
    pub gamma_shift: f64, // Factor for shift amount

    // Stagnant Iterations Threshold
    pub stagnant_iterations_threshold: usize,
}

impl AmalgamIdeaParameters {
    pub fn new
    (
    population_size: usize,
    tau: f64,
    c_mult_inc: f64, 
    c_mult_dec: f64,
    eta_cov: f64,
    eta_shift: f64,
    alpha_shift: f64,
    gamma_shift: f64,
    stagnant_iterations_threshold: usize,
    ) -> Self {

        Self{ population_size, tau, c_mult_inc, c_mult_dec, eta_cov, eta_shift, alpha_shift, gamma_shift, stagnant_iterations_threshold }
    }

    pub fn new_auto(prob_size: usize, cov_mat_type: &CovMatrixType, factorized: bool, memory: bool) -> Self {

        let tau = 3.5; // Independent of everything else
        let gamma_shift = 2.0; // Independent of everything else

        // If subset indices only has one entry, then the optimal subset division hasnt been performed
        // As such, check if cov_mat_type is Full or Diagonal and compute parameters accordingly
        // Also, check if there is memory - if we're using baseline Amalgam or iAmalgam

        let mut alphas_cov = [0.0f64; 3];
        let mut alphas_shift = [0.0_f64; 3];
        let alphas_population_size: [f64; 3];

        if !factorized {
            match cov_mat_type {
                CovMatrixType::Full => {
                    if memory {
                        alphas_cov = [-1.1, 1.2, 1.6];
                        alphas_shift = [-1.2, 0.31, 0.50];

                        alphas_population_size = [0.0, 10.0, 0.5];

                    } else {
                        alphas_population_size = [17.0, 3.0, 1.5];
                    }
                    
                }
                CovMatrixType::Diagonal => {
                    if memory {
                        alphas_cov = [-0.40, 0.15, -0.034];
                        alphas_shift = [-0.31, 0.70, 0.65];

                        alphas_population_size = [0.0, 4.0, 0.5];
                    } else {
                        alphas_population_size = [0.0, 10.0, 0.5];
                    }
                    
                }
            }
        } else {
            if memory {
                alphas_cov = [-0.33, 1.5,1.1];
                alphas_shift = [-0.52, 0.70, 0.65];

                alphas_population_size = [0.0, 7.0, 0.5];
            } else {
                alphas_population_size = [12.0, 8.0, 0.7];
            }
            
        }


        let population_size = (alphas_population_size[0] + alphas_population_size[1] * (prob_size as f64).powf(alphas_population_size[2])) as usize;

        let eta_cov = 1.0 - (alphas_cov[0] * (population_size as f64).powf(alphas_cov[1]) / (prob_size as f64).powf(alphas_cov[2])).exp();
        let eta_shift = 1.0 - (alphas_shift[0] * (population_size as f64).powf(alphas_shift[1]) / (prob_size as f64).powf(alphas_shift[2])).exp();




        let alpha_shift = 0.5 * tau * population_size as f64 / (population_size + 1) as f64;
        
        let c_mult_dec = 0.9;
        let c_mult_inc = 1.0 / c_mult_dec;

        let stagnant_iterations_threshold: usize = 25 + prob_size;


        Self {
            population_size,
            tau,
            c_mult_inc,
            c_mult_dec,
            eta_cov,
            eta_shift,
            alpha_shift,
            gamma_shift,
            stagnant_iterations_threshold,
        }

    }
}



pub enum AmalgamIdeaError {
    IncompatibleParameterSet
}

