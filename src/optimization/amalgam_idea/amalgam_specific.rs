use nalgebra::{DVector, DMatrix};
use rand::{thread_rng, Rng};


use super::super::optimizer::{Optimizer, OptimizationError};
use super::super::set_of_var_subsets::*;
use super::super::variable_subsets::*;
use super::super::tools::select_top_n;


use super::{AmalgamIdea, AmalgamIdeaError};


impl<'a> AmalgamIdea<'a> {
    pub fn selection(&mut self) -> Result<Vec<Vec<f64>>, AmalgamIdeaError> {
        // Check if initialization has been done:
        let _ = self.check_initialization()?;

        let params = self.parameters.as_ref().unwrap();
        let num_to_select = (params.tau * params.population_size as f64) as usize;

        let (selection, _fitnesses) = 
        select_top_n(&self.current_population, &self.current_fitnesses, num_to_select);

        // self.latest_selection = Some(selection);

        Ok(selection)
    }

    pub fn update_distribution(&mut self, latest_selection: Vec<Vec<f64>>) -> Result<(Vec<DVector<f64>>, Vec<DMatrix<f64>>), AmalgamIdeaError> {

        // Check if initialized
        let _ = self.check_initialization()?;

        // Check if selection has been performed and retrieve latest selection
        // let latest_selection = self.latest_selection.as_ref().ok_or(AmalgamIdeaError::SelectionNotPerformed)?;


        ////// Update distribution per subset

        // Get current dist
        let mut prev_means: Vec<DVector<f64>> = Vec::new();
        let mut prev_covs: Vec<DMatrix<f64>> = Vec::new();

        {
            let subsets = self.subsets.as_mut().unwrap();
            let subsets_vec = subsets.get_subsets_mut();

            for subset in subsets_vec.iter() {
                
                let (prev_mean_ref, prev_cov_ref) = subset.get_distribution().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;
                let (prev_mean, prev_cov) = (prev_mean_ref.clone(), prev_cov_ref.clone());
                prev_means.push(prev_mean);
                prev_covs.push(prev_cov);
            }

            // Update population and distribution based on the selected individuals
            let _ = subsets.set_population(latest_selection.clone())
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;

            // let _ = subsets.compute_distributions().
            // map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            let _ = subsets.compute_distributions()
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;

        }

        // Separate prev_means from prev_covs
        

        // Update the means and cov matrices
        let _ = self.get_updated_mean_shifts(&prev_means)?;
        let new_means = self.get_updated_means()?;
        let new_covs = self.get_updated_covs(&prev_covs)?;
        


        // Set the new distribution
        let subsets = self.subsets.as_mut().unwrap();
        let subsets_vec = subsets.get_subsets_mut();

        // for (i, subset) in subsets_vec.iter_mut().enumerate() {
        //     subset.set_distribution_manual(new_means[i].clone(), new_covs[i].clone())
        //     .map_err
        //     (
        //         |e| AmalgamIdeaError::VariableSubsetError
        //         { 
        //             err: VariableSubsetError::MultivariateGaussianError { err: e }
        //         }

        //     )?;
        // }

        for (i, subset) in subsets_vec.iter_mut().enumerate() {
            subset.set_distribution_manual(new_means[i].clone(), new_covs[i].clone())
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
        }


        Ok((new_means, new_covs))
    }

    pub fn update_population(&mut self)  -> Result<Vec<Vec<f64>>, AmalgamIdeaError> {
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
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;

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
        // self.latest_selection = None;


        Ok(improved_individuals)
    }

    pub fn update_parameters(&mut self,
        improved_individuals: Vec<Vec<f64>>,
        curr_means: Vec<DVector<f64>>
        ) 
        -> Result<(), AmalgamIdeaError> {

        
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
                let subset_dist = subset.get_distribution_object().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;
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

    pub fn get_updated_means(&self) -> Result<Vec<DVector<f64>>, AmalgamIdeaError>{

        let subsets = self.subsets.as_ref().ok_or(AmalgamIdeaError::VariableSubsetsNotDefined)?;
        let subsets_vec = subsets.get_subsets();

        let mut new_means = Vec::new();

        // Update the distribution
        for subset in subsets_vec.iter() {
            // Get the provisional distribution parameters
            let (new_mean_ref, _provisional_cov_ref) = 
            subset.get_distribution().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;
        
            new_means.push(new_mean_ref.clone());
            
        }

        Ok(new_means)
    }

    pub fn get_updated_mean_shifts(&mut self, prev_means: &Vec<DVector<f64>>) -> Result<(), AmalgamIdeaError>{

        let mut new_mean_shifts = Vec::new();

        let subsets = self.subsets.as_ref().ok_or(AmalgamIdeaError::VariableSubsetsNotDefined)?;
        let subsets_vec = subsets.get_subsets();

        // Update the distribution
        for (i, subset) in subsets_vec.iter().enumerate() {
            // println!("Subset: {:?}", &subset);
            
            let prev_mean = &prev_means[i];
            let prev_mean_shift = &self.current_mean_shifts[i];
            let (new_mean, _new_cov_ref) = 
            subset.get_distribution().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;

            // Compute the provisional mean shit
            let provisional_shift = new_mean - prev_mean;

            let eta_shift = self.parameters.as_ref().unwrap().eta_shift;

            let new_mean_shift = (1.0 - eta_shift) * prev_mean_shift + eta_shift * &provisional_shift;

            new_mean_shifts.push(new_mean_shift);

        }

        self.current_mean_shifts = new_mean_shifts;

        Ok(())
    }

    pub fn get_updated_covs(&mut self, prev_dists: &Vec<DMatrix<f64>>) -> Result<Vec<DMatrix<f64>>, AmalgamIdeaError> {
        
        let subsets = self.subsets.as_mut().unwrap();
        let subsets_vec = subsets.get_subsets_mut();

        let mut new_covs = Vec::new();

        // Update the distribution
        for (i, subset) in subsets_vec.iter_mut().enumerate() {

            // Get the provisional distribution parameters
            let (_provisional_mean_ref, provisional_cov_ref) = 
            subset.get_distribution().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;

            let provisional_cov = provisional_cov_ref.clone();

            // Get the previous dist params
            let prev_cov = &prev_dists[i];


            // Compute new cov matrix
            let eta_cov = self.parameters.as_ref().unwrap().eta_cov;
            let c_mult = self.c_mult.as_ref().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;

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
    ) -> Result<Vec<Vec<f64>>, AmalgamIdeaError> {

        let params = self
            .parameters
            .as_ref()
            .ok_or(AmalgamIdeaError::ParametersNoSet)?;

        let subsets = self
            .subsets
            .as_ref()
            .ok_or(AmalgamIdeaError::VariableSubsetsNotDefined)?;

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
            let c_mult = self.c_mult.as_ref().ok_or(AmalgamIdeaError::InitializationNotPerformed)?;

            let shifted_ind_dvec = ind_dvec + c_mult * gamma * mean_shift;
            let shifted_ind: Vec<f64> = shifted_ind_dvec.iter().copied().collect();

            sampled_with_shifted[i] = shifted_ind;
        }

        // Ensure that shifted values respect constraints
        let subsets = self.subsets.as_ref().unwrap();

        let corrected_shifted = subsets.enforce_constraint_external(&sampled_with_shifted)
        .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;


        Ok(corrected_shifted)
    }
}

