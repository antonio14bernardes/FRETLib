use super::multivariate_gaussian::*;
use super::constraints::*;
use nalgebra::constraint;
use nalgebra::{DMatrix, DVector};
use rand::distributions;
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use core::num;
use std::collections::HashSet;
use std::ops::{Add, Sub, Div, Mul};
use std::fmt::Debug;


#[derive(Debug)]
pub struct VariableSubset<T>
where
    T: Clone,
{
    indices: Vec<usize>,
    population: Option<Vec<Vec<T>>>,

    distribution: Option<MultivariateGaussian>,
    cov_mat_type: Option<CovMatrixType>,

    constraint: Option<OptimizationConstraint<T>>

}

impl<T> VariableSubset<T> 
where T: Clone,
{
    pub fn new(indices: &[usize]) -> Self {
        Self {
            indices: indices.to_vec(),
            population: None,
            distribution: None,
            cov_mat_type: None,
            constraint: None,
        }
    }

    pub fn set_population(&mut self, population: Vec<Vec<T>>) {
        self.population = Some(population);
    }

    pub fn get_indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn get_population(&self) -> Option<&Vec<Vec<T>>> {
        self.population.as_ref()
    }

    pub fn get_constraint(&self) -> Option<&OptimizationConstraint<T>> {
        self.constraint.as_ref()
    }

    pub fn get_covariance_matrix_type(&self) -> Option<&CovMatrixType> {
        self.cov_mat_type.as_ref()
    }

    pub fn check_ready(&self) -> bool {
        self.population.is_some()
    }
}




impl<T> VariableSubset<T>
where
    T: Default + Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + From<f64> + Into<f64>,
{
    pub fn set_constraint(&mut self, constraint: OptimizationConstraint<T>) -> Result<(), VariableSubsetError> {
        match &constraint {
            OptimizationConstraint::MaxValue { max } => {
                let len = max.len();
                if len != 1 && len != self.indices.len() {
                    return Err(VariableSubsetError::InvalidMinMaxVecLen)
                }
            }
            OptimizationConstraint::MinValue { min } => {
                let len = min.len();
                if len != 1 && len != self.indices.len() {
                    return Err(VariableSubsetError::InvalidMinMaxVecLen)
                }
            }
            OptimizationConstraint::MaxMinValue {max, min } => {
                let min_len = min.len();
                let max_len = max.len();

                if min_len != 1 && min_len != self.indices.len() || max_len != 1 && max_len != self.indices.len(){
                    return Err(VariableSubsetError::InvalidMinMaxVecLen)
                }
            }

            _ => {}
        }
        self.constraint = Some(constraint);

        Ok(())
    }

    pub fn enforce_constraint(&mut self) -> Result<(), VariableSubsetError> {
        let og_population = self.population.as_ref().ok_or(VariableSubsetError::PopulationNotFound)?;
    
        if let Some(constraint) = self.constraint.as_ref() {

            // Correct population according to constraint
            let corrected_population = enforce_constraint_external(og_population, constraint);

            // Update the population with the corrected values
            self.population = Some(corrected_population);

        }
    
        Ok(())
    }

    /******
     * WARNING. At this time, the distribution for the current struct can only be a 
     * Multivariate Dist with f64s
    ******/

    pub fn set_covariance_matrix_type(&mut self, cov_type: CovMatrixType) -> Result<(), VariableSubsetError> {
        self.cov_mat_type = Some(cov_type);

        self.compute_distribution()
    }

    pub fn compute_distribution(&mut self) -> Result<(), VariableSubsetError> {
        let values = self.population.as_ref().ok_or(VariableSubsetError::PopulationEmpty)?;

        // Ensure the observations are valid - the number of variables matches the indices size
        if values[0].len() != self.indices.len() {
            return Err(VariableSubsetError::IncompatiblePopulationShape);
        }

        // Convert `Vec<Vec<T>>` to `Vec<Vec<f64>>` by mapping each element to f64
        let converted_values: Vec<Vec<f64>> = values
            .iter()
            .map(|row| row.iter().map(|val| (*val).into()).collect())
            .collect();

        // Convert `Vec<Vec<f64>>` to `&[&[f64]]` for the function call
        let values_slices: Vec<&[f64]> = converted_values.iter().map(|v| v.as_slice()).collect();

        // Check the desired type of covariance matrix
        let cov_type = self.cov_mat_type.as_ref().unwrap_or(&CovMatrixType::Full); // Default to Full cov matrix

        // Compute the multivariate Gaussian distribution based on these observations
        let distribution = MultivariateGaussian::from_observations(&values_slices, cov_type)
        .map_err(|e| VariableSubsetError::MultivariateGaussianError { err: e })?;
        self.distribution = Some(distribution);

        Ok(())
    }

    // Set the distribution manually using the mean and flattened covariance vector
    pub fn set_distribution_from_vecs(&mut self, mean: Vec<f64>, flat_cov: Vec<f64>) -> Result<(), VariableSubsetError> {
        // Check if the mean vector length matches the number of variables
        if mean.len() != self.indices.len() {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
        
        // Check if the flattened covariance vector has the correct length
        let expected_len = self.indices.len() * (self.indices.len() + 1) / 2;
        if flat_cov.len() != expected_len {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
        
        // Try creating the MultivariateGaussian, returning a wrapped error if it fails
        let distribution = MultivariateGaussian::new_from_vecs(mean, flat_cov)
            .map_err(|err| VariableSubsetError::MultivariateGaussianError { err })?;
        
        self.distribution = Some(distribution);
        Ok(())
    }

    // Set the distribution manually using DVector and DMatrix from nalgebra
    pub fn set_distribution_manual(&mut self, mean: DVector<f64>, cov: DMatrix<f64>) -> Result<(), VariableSubsetError> {
        // Check if the mean vector length matches the number of variables
        if mean.len() != self.indices.len() {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }

        // Check if the covariance matrix has the correct dimensions
        if cov.nrows() != self.indices.len() || cov.ncols() != self.indices.len() {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
        
        // Try creating the MultivariateGaussian, returning a wrapped error if it fails
        let distribution = MultivariateGaussian::new(mean, cov)
            .map_err(|err| VariableSubsetError::MultivariateGaussianError { err })?;
        self.distribution = Some(distribution);
        Ok(())
    }

    // Get a reference to the distribution
    pub fn get_distribution(&self) -> Option<(&DVector<f64>, &DMatrix<f64>)> {
        if let Some(dist) = &self.distribution {
            return Some(dist.get_mean_cov())
        }
        None
    } 

    pub fn get_distribution_object(&self) -> Option<&MultivariateGaussian> {
        self.distribution.as_ref()
    }

    // Sample new set of individuals from distribution
    pub fn sample_individuals(&self, n: usize, rng: &mut impl Rng) -> Result<Vec<Vec<T>>, VariableSubsetError> {


        if let Some(ref dist) = self.distribution {
            let mut new_inds: Vec<Vec<T>> = Vec::with_capacity(n);

            for _ in 0..n {
                let out_dvec= dist.sample(rng);

                // Convert DVector<f64> to Vec<f64>
                let new_ind: Vec<T> = out_dvec.iter().map(|a| T::from(*a)).collect();

                new_inds.push(new_ind);
            }
    
            if let Some(constraint) = self.constraint.as_ref() {
                let corrected_inds = enforce_constraint_external(&new_inds, constraint);

                new_inds = corrected_inds;
            }

            return Ok(new_inds);
        } else {
            return Err(VariableSubsetError::DistributionNotYetComputed);
        }
    }

    // Init population based on set distribution
    pub fn initialize_population(&mut self, pop_size: usize, rng: &mut impl Rng) -> Result<(), VariableSubsetError> {
        let new_pop: Vec<Vec<T>> = self.sample_individuals(pop_size, rng)?;

        self.population = Some(new_pop);

        Ok(())
    }

    pub fn initialize_random_population(
        &mut self,
        pop_size: usize,
        rng: &mut impl Rng,
        random_range: Option<(T, T)>,  // Optional range for random generation
    ) -> Result<(), VariableSubsetError> {
        let mut population: Vec<Vec<T>> = Vec::with_capacity(pop_size);
        let num_values = self.indices.len();
    
        // Default min and max values
        let mut min_values = vec![T::default(); num_values];
        let mut max_values = vec![T::default() + T::from(1.0); num_values];
    
        // Apply constraints
        if let Some(constraint) = self.constraint.as_ref() {
            match constraint {
                OptimizationConstraint::MaxValue { max } => {
                    let constraint_max = if max.len() == 1 { vec![max[0]; num_values] } else { max.clone() };
                    for i in 0..num_values {

                        max_values[i] = constraint_max[i];
    
                        // Ensure range is valid
                        if min_values[i] > max_values[i] {
                            min_values[i] = max_values[i] - T::from(0.1); // Small buffer to avoid empty range
                        }
                    }
                }
                OptimizationConstraint::MinValue { min } => {
                    let constraint_min = if min.len() == 1 { vec![min[0]; num_values] } else { min.clone() };
                    for i in 0..num_values {
                        min_values[i] = constraint_min[i];
    
                        // Ensure range is valid
                        if min_values[i] > max_values[i] {
                            max_values[i] = min_values[i] + T::from(0.1); // Small buffer to avoid empty range
                        }
                    }
                }
                OptimizationConstraint::MaxMinValue { max, min } => {
                    let constraint_max = if max.len() == 1 { vec![max[0]; num_values] } else { max.clone() };
                    let constraint_min = if min.len() == 1 { vec![min[0]; num_values] } else { min.clone() };
                    
                    for i in 0..num_values {
                        max_values[i] = constraint_max[i];
                        min_values[i] = constraint_min[i];
    
                        // Ensure range is valid
                        if min_values[i] > max_values[i] {
                            return Err(VariableSubsetError::InvalidMinMaxValues);
                        }
                    }
                }
                _ => {}
            }
        } else {
            // Apply random range if provided
            if let Some((min_val, max_val)) = random_range {
                if min_val > max_val {
                    return Err(VariableSubsetError::InvalidMinMaxValues);
                }
                min_values = vec![min_val; num_values];
                max_values = vec![max_val; num_values];
            }
        }
    
        // Generate random values for population
        for _ in 0..pop_size {
            let mut individual: Vec<T> = Vec::with_capacity(num_values);
            for i in 0..num_values {
                let min_f64 = min_values[i].into();
                let max_f64 = max_values[i].into();
                
                // Generate a random value in the range [min, max] without SampleUniform
                let random_factor = rng.gen::<f64>(); // Between 0.0 and 1.0
                let random_val_f64 = min_f64 + random_factor * (max_f64 - min_f64);
                
                // Convert back to T
                individual.push(T::from(random_val_f64));
            }
    
            // Apply the constraint to repair the values
            if let Some(ref constraint) = self.constraint {
                constraint.repair(&mut individual);
            }
    
            population.push(individual);
        }
    
        self.population = Some(population);
        Ok(())
    }

}


pub fn enforce_constraint_external<T>(population: &Vec<Vec<T>>, constraint: &OptimizationConstraint<T>) -> Vec<Vec<T>>
where
    T: Default + Copy + Clone + PartialOrd + 
    Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + 
    From<f64> + Into<f64>,
{
    let mut corrected_population: Vec<Vec<T>> = Vec::new();
    
    for individual in population {
        let mut corrected_individual = individual.clone(); 
        constraint.repair(&mut corrected_individual); // Apply the constraint repair
        corrected_population.push(corrected_individual); // Add to the corrected population
    }

    corrected_population
}

#[derive(Debug)]
pub enum VariableSubsetError {
    IncompatiblePopulationShape,
    PopulationEmpty,
    PopulationNotFound,
    MultivariateGaussianError{err: MultivariateGaussianError},
    SubsetsAreNotIndependent,
    SubsetIndicesEmpty,
    InvalidIndices,
    IncompatibleNumberOfConstraints,
    DistributionNotYetComputed,
    InvalidMinMaxValues,
    InvalidMinMaxVecLen,
    InvalidCovMatrixTypeslen,
    SubsetsNotDefined,
    SubsetsAlreadyDefined,
    IncompatibleInputSizes,

}


#[cfg(test)]
mod tests_subsets {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_set_population() {
        let indices = vec![0, 1, 2];
        let mut subset = VariableSubset::new(&indices);

        let population = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        subset.set_population(population.clone());
        assert_eq!(subset.population.unwrap(), population, "Population was not set correctly.");
    }

    #[test]
    fn test_set_constraint() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let constraint = OptimizationConstraint::SumTo { sum: 1.0 };
        subset.set_constraint(constraint);

        if let Some(OptimizationConstraint::SumTo { sum }) = subset.constraint {
            assert_eq!(sum, 1.0, "Constraint sum was not set correctly.");
        } else {
            panic!("Constraint was not set correctly.");
        }
    }

    #[test]
    fn test_initialize_population_with_single_max_min_value_constraint() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let pop_size = 3;
        let mut rng = thread_rng();

        // Set a single MaxValue and MinValue for the subset with multiple indices
        let constraint = OptimizationConstraint::MaxMinValue {
            max: vec![5.0],  // Single max value
            min: vec![1.0],  // Single min value
        };
        subset.set_constraint(constraint).unwrap();

        // Initialize the population with the constraint
        subset.initialize_random_population(pop_size, &mut rng, Some((0.0, 10.0))).unwrap();
        let population = subset.population.as_ref().unwrap();

        // Check if each individual in the population respects the single min and max value constraints
        for individual in population {
            for value in individual {
                assert!(*value >= 1.0 && *value <= 5.0, "Value does not match the min/max constraint.");
            }
        }
    }

    #[test]
    fn test_compute_distribution() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let population = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        subset.set_population(population.clone());

        let result = subset.compute_distribution();
        assert!(result.is_ok(), "Failed to compute distribution.");

        assert!(subset.distribution.is_some(), "Distribution was not computed or set.");
    }

    #[test]
    fn test_sample_individuals() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        // Set a simple population
        let population = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        subset.set_population(population.clone());

        // Compute the distribution based on the population
        subset.compute_distribution().unwrap();

        // Sample new individuals
        let mut rng = thread_rng();
        let new_individuals = subset.sample_individuals(3, &mut rng);

        assert!(new_individuals.is_ok(), "Failed to sample individuals.");
        let sampled_population = new_individuals.unwrap();
        assert_eq!(sampled_population.len(), 3, "Did not sample the correct number of individuals.");
    }

    #[test]
    fn test_sample_individuals_no_distribution() {
        let indices = vec![0, 1];
        let subset = VariableSubset::<f64>::new(&indices);

        let mut rng = thread_rng();
        let result = subset.sample_individuals(3, &mut rng);
        assert!(result.is_err(), "Expected an error because the distribution was not computed.");
    }

    #[test]
    fn test_initialize_population_no_constraints() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::<f64>::new(&indices);

        let pop_size = 5;
        let mut rng = thread_rng();

        // Initialize without constraints and no custom range
        subset.initialize_random_population(pop_size, &mut rng, None).unwrap();
        let population = subset.population.as_ref().unwrap();

        // Ensure the population has been initialized with the correct size
        assert_eq!(population.len(), pop_size, "Population size is incorrect.");
        assert_eq!(population[0].len(), indices.len(), "Problem size is incorrect.");

        // Ensure values are within default range [0.0, 1.0]
        for individual in population {
            for value in individual {
                assert!(*value >= 0.0 && *value <= 1.0, "Value is out of default range [0.0, 1.0].");
            }
        }
    }

    #[test]
    fn test_initialize_population_with_sumto_constraint() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let pop_size = 3;
        let mut rng = thread_rng();

        // Set the SumTo constraint
        let constraint = OptimizationConstraint::SumTo { sum: 10.0 };
        subset.set_constraint(constraint);

        // Initialize the population with the constraint
        subset.initialize_random_population(pop_size, &mut rng, None).unwrap();
        let population = subset.population.as_ref().unwrap();

        // Check if each individual in the population sums to 10.0
        for individual in population {
            let sum: f64 = individual.iter().copied().sum();
            assert!((sum - 10.0).abs() < 1e-6, "Sum does not match the constraint.");
        }
    }

    #[test]
    fn test_initialize_population_with_min_max_constraints() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let pop_size = 4;
        let mut rng = thread_rng();

        // Set the MaxValue and MinValue constraints
        let constraint = OptimizationConstraint::MaxMinValue {
            max: vec![4.0, 5.0],
            min: vec![1.0, 2.0],
        };
        subset.set_constraint(constraint);

        // Initialize the population with the constraints
        subset.initialize_random_population(pop_size, &mut rng, None).unwrap();
        let population = subset.population.as_ref().unwrap();

        // Ensure all values are within the specified min and max ranges
        for individual in population {
            assert!(individual[0] >= 1.0 && individual[0] <= 4.0, "Value out of range for index 0.");
            assert!(individual[1] >= 2.0 && individual[1] <= 5.0, "Value out of range for index 1.");
        }
    }

    #[test]
    fn test_initialize_population_with_custom_range() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        let pop_size = 6;
        let prob_size = 2;
        let mut rng = thread_rng();

        // Custom range: values between [10.0, 20.0]
        let random_range = Some((10.0, 20.0));

        // Initialize the population with the custom range
        subset.initialize_random_population(pop_size, &mut rng, random_range).unwrap();
        let population = subset.population.as_ref().unwrap();

        // Ensure values are within the custom range [10.0, 20.0]
        for individual in population {
            for value in individual {
                println!("Value: {}", &value);
                assert!(*value >= 10.0 && *value <= 20.0, "Value is out of custom range [10.0, 20.0].");
            }
        }
    }

    #[test]
    fn test_cov_matrix_type_handling() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        // Set a simple population
        let population = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        subset.set_population(population.clone());

        // Set the covariance matrix type to Diagonal
        subset.set_covariance_matrix_type(CovMatrixType::Diagonal).unwrap();

        // Compute the distribution with Diagonal covariance matrix type
        subset.compute_distribution().unwrap();

        // Retrieve the distribution and check if it's diagonal
        if let Some((_, cov)) = subset.get_distribution() {
            // Ensure the off-diagonal elements are zero for diagonal covariance matrix
            for i in 0..cov.nrows() {
                for j in 0..cov.ncols() {
                    if i != j {
                        assert_eq!(cov[(i, j)], 0.0, "Off-diagonal element is not zero in diagonal covariance matrix.");
                    }
                }
            }
        } else {
            panic!("Failed to compute or retrieve distribution.");
        }

        // Now, set the covariance matrix type to Full
        subset.set_covariance_matrix_type(CovMatrixType::Full).unwrap();

        // Compute the distribution with Full covariance matrix type
        subset.compute_distribution().unwrap();

        // Retrieve the distribution and check if it's full (non-zero off-diagonal elements)
        if let Some((_, cov)) = subset.get_distribution() {
            // Ensure that at least one off-diagonal element is non-zero for full covariance matrix
            let mut has_non_zero_off_diag = false;
            for i in 0..cov.nrows() {
                for j in 0..cov.ncols() {
                    if i != j && cov[(i, j)] != 0.0 {
                        has_non_zero_off_diag = true;
                        break;
                    }
                }
            }
            assert!(has_non_zero_off_diag, "Full covariance matrix does not have non-zero off-diagonal elements.");
        } else {
            panic!("Failed to compute or retrieve distribution.");
        }
    }

    #[test]
    fn test_enforce_constraint() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        // Set a simple population
        let population = vec![
            vec![5.0, 7.0],
            vec![9.0, 2.0],
            vec![3.0, 8.0],
        ];
        subset.set_population(population.clone());

        // Set a constraint to repair individuals such that each value is within the range [1.0, 6.0]
        let constraint = OptimizationConstraint::MaxMinValue {
            max: vec![6.0, 6.0],
            min: vec![1.0, 1.0],
        };
        subset.set_constraint(constraint).unwrap();

        // Enforce the constraints
        let result = subset.enforce_constraint();
        assert!(result.is_ok(), "Failed to enforce constraints.");

        // Retrieve the corrected population
        let corrected_population = subset.get_population().unwrap();

        // Check if each value in the corrected population is within the range [1.0, 6.0]
        for individual in corrected_population {
            assert!(individual[0] >= 1.0 && individual[0] <= 6.0, "Value out of range for index 0.");
            assert!(individual[1] >= 1.0 && individual[1] <= 6.0, "Value out of range for index 1.");
        }
    }

    #[test]
    fn test_sample_individuals_with_max_min_constraints() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::new(&indices);

        // Set the MaxMinValue constraints for the subset
        let constraint = OptimizationConstraint::MaxMinValue {
            max: vec![4.0, 5.0],
            min: vec![1.0, 2.0],
        };
        subset.set_constraint(constraint).unwrap();

        // Set up a simple population to compute the initial distribution
        let population = vec![
            vec![1.5, 2.5],
            vec![2.5, 3.5],
            vec![3.5, 4.5],
        ];
        subset.set_population(population);
        subset.compute_distribution().unwrap();

        // Sample a large number of individuals
        let num_samples = 1000;
        let mut rng = thread_rng();
        let sampled_individuals = subset.sample_individuals(num_samples, &mut rng).unwrap();

        // Check if all sampled individuals satisfy the constraints
        for individual in sampled_individuals {
            assert!(individual[0] >= 1.0 && individual[0] <= 4.0, "Sampled value out of range for index 0.");
            assert!(individual[1] >= 2.0 && individual[1] <= 5.0, "Sampled value out of range for index 1.");
        }
    }

    #[test]
    fn test_set_distribution_manually() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::<f64>::new(&indices);

        // Define mean and covariance for the manual distribution setting
        let mean = vec![1.0, 2.0];
        let cov = vec![
            vec![1.0, 0.5],
            vec![0.5, 2.0],
        ];

        // Flatten the covariance matrix for `set_distribution_from_vecs`
        let flat_cov = MultivariateGaussian::flatten_cov(&cov);

        // Test `set_distribution_from_vecs`
        let result = subset.set_distribution_from_vecs(mean.clone(), flat_cov.clone());
        assert!(result.is_ok(), "Failed to set distribution from vecs.");

        // Check that distribution was set correctly
        if let Some((dist_mean, dist_cov)) = subset.get_distribution() {
            let mean_vec = DVector::from_vec(mean.clone());
            
            // Unflatten `flat_cov` and convert it to DMatrix for comparison
            let unflattened_cov = MultivariateGaussian::unflatten_cov(&flat_cov).expect("Failed to unflatten covariance matrix");
            let cov_matrix = DMatrix::from_vec(2, 2, unflattened_cov.into_iter().flatten().collect());

            assert_eq!(dist_mean, &mean_vec, "Mean vector does not match after setting from vecs.");
            assert_eq!(dist_cov, &cov_matrix, "Covariance matrix does not match after setting from vecs.");
        } else {
            panic!("Distribution was not set when using set_distribution_from_vecs.");
        }

        // Test `set_distribution_manual`
        let mut subset = VariableSubset::<f64>::new(&indices);
        let mean_dvector = DVector::from_vec(mean.clone());
        let cov_dmatrix = DMatrix::from_vec(2, 2, cov.iter().flatten().cloned().collect());

        let result = subset.set_distribution_manual(mean_dvector.clone(), cov_dmatrix.clone());
        assert!(result.is_ok(), "Failed to set distribution manually.");

        // Check that distribution was set correctly
        if let Some((dist_mean, dist_cov)) = subset.get_distribution() {
            assert_eq!(dist_mean, &mean_dvector, "Mean vector does not match after setting manually.");
            assert_eq!(dist_cov, &cov_dmatrix, "Covariance matrix does not match after setting manually.");
        } else {
            panic!("Distribution was not set when using set_distribution_manual.");
        }
    }

    #[test]
    fn test_initialize_population() {
        let indices = vec![0, 1];
        let mut subset = VariableSubset::<f64>::new(&indices);

        // Define mean and covariance for the distribution
        let mean = DVector::from_vec(vec![1.0, 2.0]);
        let cov = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 2.0]);

        // Set the distribution manually for the subset
        subset.set_distribution_manual(mean.clone(), cov.clone()).unwrap();

        // Set the population size
        let pop_size = 100000;
        let mut rng = thread_rng();

        // Initialize the population using the distribution
        let result = subset.initialize_population(pop_size, &mut rng);
        assert!(result.is_ok(), "Failed to initialize population from distribution.");

        // Retrieve the initialized population
        let population = subset.get_population().expect("Population should be initialized");

        // Check if the population has the correct number of individuals
        assert_eq!(population.len(), pop_size, "Population size does not match the expected size.");

        // Check if each individual has the correct number of dimensions
        for individual in population {
            assert_eq!(individual.len(), indices.len(), "Individual dimension does not match indices length.");
        }

        // Optional: Perform statistical checks on the population (e.g., mean and variance)
        // Check if the population is approximately centered around the set mean
        let mean_estimate: Vec<f64> = (0..indices.len())
            .map(|i| population.iter().map(|ind| ind[i]).sum::<f64>() / pop_size as f64)
            .collect();
        println!("mean estimate: {:?}", mean_estimate);
        for (i, mean_val) in mean.iter().enumerate() {
            assert!(
                (mean_estimate[i] - mean_val).abs() < 0.01,
                "Sampled mean for dimension {} is not close to the expected mean.",
                i
            );
        }
    }
}