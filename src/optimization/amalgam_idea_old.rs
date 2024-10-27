use std::vec;

use nalgebra::{Cholesky, DMatrix, DVector};
use rand::{random, thread_rng, Rng};

use super::constraints::OptimizationConstraint;
use super::multivariate_gaussian::CovMatrixType;
use super::optimizer::{Optimizer, OptimizerError, select_top_n};
use super::set_of_var_subsets::*;
use super::variable_subsets::*;
use super::amalgam_parameters::*;

pub struct AmalgamIdea<'a> {
    problem_size: usize,
    iter_memory: bool,
    parameters: Option<AmalgamIdeaParameters>,
    fitness_function: Option<Box<dyn Fn(&[f64]) -> f64 + 'a>>,
    subsets: Option<SetVarSubsets<f64>>,
    initial_population: Option<Vec<Vec<f64>>>,
    current_population: Vec<Vec<f64>>,
    latest_selection: Option<Vec<Vec<f64>>>,
    best_solution: Option<(Vec<f64>, f64)>,
    current_fitnesses: Vec<f64>,
    best_fitnesses: Vec<f64>,
    stagnant_iterations: usize,

    current_mean_shifts: Vec<DVector<f64>>,

    c_mult: Option<f64>,
}


impl<'a> Optimizer<f64> for AmalgamIdea<'a> {

    // Evaluate the fitness of a single solution
    fn evaluate(&self, solution: &Vec<f64>) -> Result<f64, OptimizerError> {
        let fitness_function = self
            .fitness_function
            .as_ref()
            .ok_or(OptimizerError::FitnessFunctionNotSet)?;
        
        Ok(fitness_function(&solution))

    }

    // Initialize the population
    fn initialize(&mut self) -> Result<(), OptimizerError>{

        // Check if fitness function has been set
        let _ = self.fitness_function.as_ref().ok_or(OptimizerError::FitnessFunctionNotSet)?;

        // Check if subsets have been initialized. If not, assume no subset division.
        if self.subsets.is_none() {
            let mut subsets = 
            SetVarSubsets::<f64>::new_empty(vec![(0..self.problem_size).collect()])
            .map_err(|err| OptimizerError::SubsetError {err})?;

            if let Some(population) = &self.initial_population {
                subsets.set_population(population.clone()).map_err(|err| OptimizerError::VariableSubsetError { err })?;
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
        

        // Check if population has been initialized. If not, do random init from the subsets
        if self.initial_population.is_none() {

            let pop_size = self.get_pop_size().unwrap();

            let subsets = self.subsets.as_mut().unwrap();

            let mut rng = thread_rng(); // Use this function for random number generation

            subsets.initialize_random_population(pop_size, &mut rng, None)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;

            self.initial_population = Some(subsets.get_population());
        }


        // Initialize the distributions in the subsets
        self.subsets.as_mut().unwrap().compute_distributions()
        .map_err(|err| OptimizerError::VariableSubsetError { err })?;


        // Set the current pop and fitness fields
        self.current_population = self.initial_population.as_ref().unwrap().clone();
        self.current_fitnesses = self.evaluate_population()?;

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
    fn step(&mut self) -> Result<(), OptimizerError> {
        // Perform selection
        self.selection()?;

        // Update distribution based on selection
        let (new_means, _new_covs) = self.update_distribution()?;

        // Sample population and return the improved individuals for the the parameter update method
        let improved_individuals = self.update_population()?;

        // Update cmult and number of stagnant iterations
        self.update_parameters(improved_individuals, new_means)?;

        Ok(())
    }

    // Run the optimization process
    fn run(&mut self, max_iterations: usize) -> Result<(), OptimizerError>  {
        
        let min_cmult = 1e-5;
        
        self.initialize();

        let mut iters = 0_usize;

        while iters < max_iterations &&  self.c_mult.unwrap() > min_cmult {
            self.step()?;

            let new_fit = self.best_solution.as_ref().unwrap().1;

            self.best_fitnesses.push(new_fit);

            iters += 1;
        }

        Ok(())
    }

    // Retrieve the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&(Vec<f64>, f64)> {
        self.best_solution.as_ref() 
    }
}


impl<'a> AmalgamIdea<'a> {
    pub fn selection(&mut self) -> Result<(), OptimizerError> {
        // Check if initialization has been done:
        let _ = self.check_initialization()?;

        let params = self.parameters.as_ref().unwrap();
        let num_to_select = (params.tau * params.population_size as f64) as usize;

        let (selection, _fitnesses) = 
        select_top_n(&self.current_population, &self.current_fitnesses, num_to_select);

        self.latest_selection = Some(selection);

        Ok(())
    }

    pub fn update_distribution(&mut self) -> Result<(Vec<DVector<f64>>, Vec<DMatrix<f64>>), OptimizerError> {

        // Check if initialized
        let _ = self.check_initialization()?;

        // Check if selection has been performed and retrieve latest selection
        let latest_selection = self.latest_selection.as_ref().ok_or(OptimizerError::SelectionNotPerformed)?;


        ////// Update distribution per subset

        // Get current dist
        let mut prev_means: Vec<DVector<f64>> = Vec::new();
        let mut prev_covs: Vec<DMatrix<f64>> = Vec::new();

        {
            let subsets = self.subsets.as_mut().unwrap();
            let subsets_vec = subsets.get_subsets_mut();

            for subset in subsets_vec.iter_mut() {
                
                let (prev_mean_ref, prev_cov_ref) = subset.get_distribution().ok_or(OptimizerError::InitializationNotPerformed)?;
                let (prev_mean, prev_cov) = (prev_mean_ref.clone(), prev_cov_ref.clone());
                prev_means.push(prev_mean);
                prev_covs.push(prev_cov);
            }

            // Update population and distribution based on the selected individuals
            let _ = subsets.set_population(latest_selection.clone())
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;

            let _ = subsets.compute_distributions().
            map_err(|err| OptimizerError::VariableSubsetError { err })?;
        }

        // Separate prev_means from prev_covs
        

        // Update the means and cov matrices
        let new_means = self.get_updated_means()?;
        let new_covs = self.get_updated_covs(&prev_covs)?;
        let _ = self.get_updated_mean_shifts(&prev_means)?;


        // Set the new distribution
        let subsets = self.subsets.as_mut().unwrap();
        let subsets_vec = subsets.get_subsets_mut();

        for (i, subset) in subsets_vec.iter_mut().enumerate() {
            subset.set_distribution_manual(new_means[i].clone(), new_covs[i].clone())
            .map_err
            (
                |e| OptimizerError::VariableSubsetError
                { 
                    err: VariableSubsetError::MultivariateGaussianError { err: e }
                }

            )?;
        }

        Ok((new_means, new_covs))
    }

    pub fn update_population(&mut self)  -> Result<Vec<Vec<f64>>, OptimizerError> {
        let _ = self.check_initialization()?;

        // Get the current best solution and other parameters
        let (curr_best_solution, curr_best_fitness) = self
        .best_solution
        .as_ref()
        .unwrap();

        let params = self
            .parameters
            .as_ref()
            .unwrap();

        let subsets = self
            .subsets
            .as_ref()
            .unwrap();

        // Create a new random number generator using thread_rng
        let mut rng = thread_rng();

        // Start the new population with the best solution from the previous generation
        let mut new_population: Vec<Vec<f64>> = vec![curr_best_solution.clone()];

        // Calculate the number of individuals to sample
        let num_to_sample = params.population_size - 1;

        // Sample the remaining individuals
        let sampled = subsets
            .sample_individuals(num_to_sample, &mut rng)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;

        // Perform shifting
        let sampled_with_shifted = self.perform_shifting(sampled, &mut rng)?;

        // Add the sampled individuals to the new population
        new_population.extend(sampled_with_shifted);

        // Update the population with the newly formed population
        self.current_population = new_population;



        // Check for improvements

        // Compute fitnesses for new population
        self.current_fitnesses = self.evaluate_population()?;


        // Gather new solutions that are better than previous best and update best solution
        let mut improved_individuals = Vec::new();
        // let mut improved_fitnesses = Vec::new();

        let mut new_best_fitness = *curr_best_fitness;
        let mut new_best_individual = curr_best_solution;

        for (individual, fitness) in self.current_population.iter().zip(&self.current_fitnesses) {
            if fitness > curr_best_fitness {
                improved_individuals.push(individual.clone());
                //improved_fitnesses.push(fitness);
            }

            if fitness > &new_best_fitness {
                new_best_fitness = *fitness;
                new_best_individual = individual;
            }
        }

        self.best_solution = Some((new_best_individual.clone(), new_best_fitness));
        
        // Reset latest selection to None
        self.latest_selection = None;


        Ok(improved_individuals)
    }

    pub fn update_parameters(&mut self,
        improved_individuals: Vec<Vec<f64>>,
        curr_means: Vec<DVector<f64>>
        ) 
        -> Result<(), OptimizerError> {

        
        let _ = self.check_initialization()?;

        let subsets = self.subsets.as_ref().unwrap();
        let params = self.parameters.as_ref().unwrap();
        let mut c_mult: f64 = *self.c_mult.as_ref().unwrap();

        // If there has been an improvement
        if !improved_individuals.is_empty() {
            // Reset stagnant iterations counter
            self.stagnant_iterations = 0;

            // if c_mult is below 1 it gets set to 1
            if c_mult < 1.0 {
                c_mult = 1.0;
            }

            // Compute mean of improved solutions
            let mut means_vec = vec![0.0; self.problem_size];
            for ind in &improved_individuals {
                for (idx, value) in ind.iter().enumerate() {
                    means_vec[idx] += value;
                }
            }
            let num_improvements = improved_individuals.len();
            for i in 0..means_vec.len() {
                means_vec[i] /= num_improvements as f64;
            }

            let means_dvec = DVector::from(means_vec);

            let scrambled_improv_means_vec = scramble_means(subsets.get_subset_indices(), &means_dvec);

            // Check if the differences between means, when transformed back to the uniform dist, is larger than theta_sdr
            // in any dimension
            let means_diff: Vec<DVector<f64>> = curr_means
                .iter()
                .zip(scrambled_improv_means_vec)
                .map(|(curr, improved)| curr - improved).collect();

            for (i, subset) in subsets.get_subsets().iter().enumerate() {

                let mean_diff = &means_diff[i];

                // Get the inverse of the cholesky factor
                let subset_dist = subset.get_distribution_object().ok_or(OptimizerError::InitializationNotPerformed)?;
                let cholesky_inv = subset_dist.get_cholesky_inv();

                // Get the diff back in the standard gaussian space
                let z = cholesky_inv * mean_diff;

                // Check if z is larger than theta_sdr in any direction
                let theta_sdr = 1.0;
                let larger = z.iter().any(|v| v.abs() > theta_sdr);

                if larger {
                    c_mult = c_mult * params.c_mult_inc;
                }
            }
        } else {

            if c_mult <= 1.0 {
                self.stagnant_iterations += 1;
            }

            if c_mult > 1.0 || self.stagnant_iterations >= params.stagnant_iterations_threshold {
                c_mult = c_mult * params.c_mult_dec;
            }

            if c_mult < 1.0 && self.stagnant_iterations < params.stagnant_iterations_threshold {
                c_mult = 1.0;
            }
            
        }

        self.c_mult = Some(c_mult);

        Ok(())
    }

    pub fn get_updated_means(&self) -> Result<Vec<DVector<f64>>, OptimizerError>{

        let subsets = self.subsets.as_ref().ok_or(OptimizerError::VariableSubsetsNotDefined)?;
        let subsets_vec = subsets.get_subsets();

        let mut new_means = Vec::new();

        // Update the distribution
        for subset in subsets_vec.iter() {
            // Get the provisional distribution parameters
            let (new_mean_ref, _provisional_cov_ref) = 
            subset.get_distribution().ok_or(OptimizerError::InitializationNotPerformed)?;
        
            new_means.push(new_mean_ref.clone());
            
        }

        Ok(new_means)
    }

    pub fn get_updated_mean_shifts(&mut self, prev_means: &Vec<DVector<f64>>) -> Result<(), OptimizerError>{

        let mut new_mean_shifts = Vec::new();

        let subsets = self.subsets.as_ref().ok_or(OptimizerError::VariableSubsetsNotDefined)?;
        let subsets_vec = subsets.get_subsets();

        // Update the distribution
        for (i, subset) in subsets_vec.iter().enumerate() {
            // println!("Subset: {:?}", &subset);
            
            let prev_mean = &prev_means[i];
            let prev_mean_shift = &self.current_mean_shifts[i];
            let (new_mean, _new_cov_ref) = 
            subset.get_distribution().ok_or(OptimizerError::InitializationNotPerformed)?;

            // Compute the provisional mean shit
            let provisional_shift = new_mean - prev_mean;

            let eta_shift = self.parameters.as_ref().unwrap().eta_shift;

            let new_mean_shift = (1.0 - eta_shift) * prev_mean_shift + eta_shift * &provisional_shift;

            new_mean_shifts.push(new_mean_shift);

        }

        self.current_mean_shifts = new_mean_shifts;

        Ok(())
    }

    pub fn get_updated_covs(&mut self, prev_dists: &Vec<DMatrix<f64>>) -> Result<Vec<DMatrix<f64>>, OptimizerError> {
        
        let subsets = self.subsets.as_mut().unwrap();
        let subsets_vec = subsets.get_subsets_mut();

        let mut new_covs = Vec::new();

        // Update the distribution
        for (i, subset) in subsets_vec.iter_mut().enumerate() {

            // Get the provisional distribution parameters
            let (_provisional_mean_ref, provisional_cov_ref) = 
            subset.get_distribution().ok_or(OptimizerError::InitializationNotPerformed)?;

            let provisional_cov = provisional_cov_ref.clone();

            // Get the previous dist params
            let prev_cov = &prev_dists[i];


            // Compute new cov matrix
            let eta_cov = self.parameters.as_ref().unwrap().eta_cov;
            let c_mult = self.c_mult.as_ref().ok_or(OptimizerError::InitializationNotPerformed)?;

            let new_cov_pre_mult = (1.0 - &eta_cov) * prev_cov + eta_cov * provisional_cov;

            // Perform adaptative variance scaling
            let new_cov = *c_mult * new_cov_pre_mult;

            new_covs.push(new_cov);
        }

        Ok(new_covs)
    }

    pub fn perform_shifting(
        &self,
        sampled: Vec<Vec<f64>>,
        rng: &mut impl Rng
    ) -> Result<Vec<Vec<f64>>, OptimizerError> {

        let params = self
            .parameters
            .as_ref()
            .ok_or(OptimizerError::ParametersNoSet)?;

        let subsets = self
            .subsets
            .as_ref()
            .ok_or(OptimizerError::VariableSubsetsNotDefined)?;

        // Compute number of individuals to shift
        let num_sampled = sampled.len();
        let num_to_shift = (params.alpha_shift * (num_sampled as f64)) as usize;

        // Select individuals to shift
        let num_subsets = subsets.get_num_subsets();
        let random_indices: Vec<usize> = (0..num_to_shift)
            .map(|_| rng.gen_range(0..num_subsets))
            .collect();

        // Init shifted vector
        let mut sampled_with_shifted = sampled;

        for i in random_indices {
            let individual = &sampled_with_shifted[i];

            let ind_dvec = DVector::<f64>::from_vec(individual.clone());
            let mean_shift = unscramble_means(subsets.get_subset_indices(), &self.current_mean_shifts);
            let gamma = params.gamma_shift;
            let c_mult = self.c_mult.as_ref().ok_or(OptimizerError::InitializationNotPerformed)?;

            let shifted_ind_dvec = ind_dvec + c_mult * gamma * mean_shift;
            let shifted_ind: Vec<f64> = shifted_ind_dvec.iter().copied().collect();

            sampled_with_shifted[i] = shifted_ind;
        }

        Ok(sampled_with_shifted)
    }
}



// Basic non algo related methods
impl<'a> AmalgamIdea<'a> {
    pub fn new(problem_size: usize, iter_memory: bool) -> Self {
        Self {
            problem_size,
            iter_memory,

            parameters: None,

            fitness_function: None,

            subsets: None,

            initial_population: None,
            current_population: Vec::new(),
            latest_selection: None,
            best_solution: None,

            current_fitnesses: Vec::new(),
            best_fitnesses: Vec::new(),

            current_mean_shifts: Vec::new(),

            stagnant_iterations: 0,

            c_mult: None,
        }
    } 

    pub fn set_parameters(&mut self, parameters: AmalgamIdeaParameters) -> Result<(), OptimizerError> {
        // Check population size compatibility with parameters
        if let Some(population) = self.initial_population.as_ref() {
            if parameters.population_size != population.len() {
                return Err(OptimizerError::PopulationIncompatibleWithParameters);
            }
        }

        self.parameters = Some(parameters); // Set the single parameters object

        Ok(())
    }
    
    pub fn set_fitness_function<F>(&mut self, fitness_function: F)
    where F: Fn(&[f64]) -> f64 + 'a,
    {
        self.fitness_function = Some(Box::new(fitness_function));
    }

    pub fn set_initial_population(&mut self, initial_population: Vec<Vec<f64>>) -> Result<(), OptimizerError> {

        if initial_population.iter().any(|ind| ind.len() != self.problem_size) {
            return Err(OptimizerError::PopulationIncompatibleWithProblemSize)
        }

        if let Some(params) = self.parameters.as_ref() {
            if params.population_size != initial_population.len() {
                return Err(OptimizerError::PopulationIncompatibleWithParameters);
            }
        }
        

        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_population(initial_population.clone())
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
        }
        self.initial_population = Some(initial_population);

        Ok(())
    }

    pub fn set_dependency_subsets(&mut self, dependency_subsets: Vec<Vec<usize>>) -> Result<(), OptimizerError> {

        if self.subsets.is_some() {
            return Err(OptimizerError::SubsetError { err: VariableSubsetError::SubsetsAlreadyDefined });
        }

        // Check if max index in the subsets is compatible with the defined problem size
        let max_idx = dependency_subsets
        .iter()
        .flat_map(|s| s.iter().cloned())
        .max()
        .ok_or(OptimizerError::SubsetsIncompatibleWithProblemSize)?;

        if max_idx != self.problem_size - 1 {
            return Err(OptimizerError::SubsetsIncompatibleWithProblemSize);
        }


        if let Some(ref init_pop) = self.initial_population {

            let set = SetVarSubsets::new(dependency_subsets, init_pop.clone(), None)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        } else {

            let set = SetVarSubsets::new_empty(dependency_subsets)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        }

        Ok(())
    }

    pub fn set_constraints(&mut self, constraints: Vec<OptimizationConstraint<f64>>) -> Result<(), OptimizerError> {
        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_constraints(constraints).map_err(|err| OptimizerError::VariableSubsetError { err })?;
        
        } else {

            if constraints.len() == 1 { // Assume we are dealing with only one subset - all vars related with eachother
                let subset_indices = vec![(0..self.problem_size).collect::<Vec<usize>>()];
                
                self.set_dependency_subsets(subset_indices)?;

                let subsets = self.subsets.as_mut().unwrap();

                subsets.set_constraints(constraints).map_err(|err| OptimizerError::VariableSubsetError { err })?;

            } else {
                return Err(OptimizerError::VariableSubsetsNotDefined);
            }
        }

        Ok(())
    }

    pub fn get_pop_size(&self) -> Option<usize> {
        if let Some(params) = &self.parameters {
            return Some(params.population_size)
        }
        None
    }

    pub fn get_parameters(&self) -> Option<&AmalgamIdeaParameters> {
        self.parameters.as_ref()
    }

    pub fn get_initial_population(&self) -> Option<&Vec<Vec<f64>>> {
        self.initial_population.as_ref()
    }

    pub fn get_current_population(&self) -> &Vec<Vec<f64>> {
        &self.current_population
    }

    pub fn get_current_fitnesses(&self) -> &Vec<f64> {
        &self.current_fitnesses
    }

    pub fn get_subsets(&self) -> Option<&SetVarSubsets<f64>> {
        self.subsets.as_ref()
    }

    pub fn get_latest_selection(&self) -> Option<&Vec<Vec<f64>>> {
        self.latest_selection.as_ref()
    }

    pub fn get_cmult(&self) -> Option<&f64> {
        self.c_mult.as_ref()
    }

    pub fn get_stagnant_iterations(&self) -> usize {
        self.stagnant_iterations
    }

    pub fn get_fitnesses(&self) -> &Vec<f64> {
        &self.best_fitnesses
    }

    pub fn check_initialization(&self) -> Result<(), OptimizerError> {
        if self.parameters.is_none() || 
        self.initial_population.is_none() || 
        self.subsets.is_none() ||
        self.current_population.is_empty() ||
        self.best_solution.is_none() {
            return Err(OptimizerError::InitializationNotPerformed);
        }

        Ok(())
    }

    pub fn evaluate_population(&self) -> Result<Vec<f64>, OptimizerError> {
        let population = &self.current_population;
        let evals: Vec<f64> = population
            .iter()
            .map(|individual| self.evaluate(individual))
            .collect::<Result<Vec<f64>, OptimizerError>>()?;

        Ok(evals)
    }
}



pub enum AmalgamIdeaError {
    IncompatibleParameterSet
}

