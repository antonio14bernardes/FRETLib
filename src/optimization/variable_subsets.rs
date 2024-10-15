use super::multivariate_gaussian::*;
use super::constraints::*;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use core::num;
use std::collections::HashSet;
use std::ops::{Add, Sub, Div, Mul};


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
    T: Copy + Clone + Into<f64> + Default + PartialOrd
     + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> 
     + From<f64> + SampleUniform + std::fmt::Debug,
{

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
    pub fn set_distribution_from_vecs(&mut self, mean: Vec<f64>, flat_cov: Vec<f64>) -> Result<(), MultivariateGaussianError> {
        let distribution = MultivariateGaussian::new_from_vecs(mean, flat_cov)?;
        self.distribution = Some(distribution);
        Ok(())
    }

    // Set the distribution manually using `DVector` and `DMatrix` from nalgebra
    pub fn set_distribution_manual(&mut self, mean: DVector<f64>, cov: DMatrix<f64>) -> Result<(), MultivariateGaussianError> {
        let distribution = MultivariateGaussian::new(mean, cov)?;
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

    // Sample new set of individuals from distribution
    pub fn sample_individuals(&self, n: usize, rng: &mut impl Rng) -> Result<Vec<Vec<f64>>, VariableSubsetError> {

        let mut new_inds: Vec<Vec<f64>> = Vec::with_capacity(n);

        if let Some(ref dist) = self.distribution {
            for _ in 0..n {
                let out_dvec= dist.sample(rng);

                // Convert DVector<f64> to Vec<f64>
                let new_ind: Vec<f64> = out_dvec.iter().cloned().collect();

                new_inds.push(new_ind);
            }

            return Ok(new_inds);
        } else {
            return Err(VariableSubsetError::DistributionNotYetComputed);
        }
    }

    // New method: Initializes the population with random values considering constraints
    pub fn initialize_random_population(
        &mut self,
        pop_size: usize,
        rng: &mut impl Rng,
        random_range: Option<(T, T)>,  // Optional range for random generation
    ) -> Result<(), VariableSubsetError> {

        let mut population: Vec<Vec<T>> = Vec::with_capacity(pop_size);

        let (min_val, max_val) = if let Some((min, max)) = random_range {
            if min > max {return Err(VariableSubsetError::InvalidMinMaxValues)}
            (min, max)
        } else {
            // Default random range, (0.0, 1.0) for f64
            (T::default(), T::default() + T::from(1.0))
        };

        let num_values = self.indices.len();
        for _ in 0..pop_size {
            let mut individual: Vec<T> = Vec::with_capacity(num_values);

            for _ in 0..num_values {
                let random_val = rng.gen_range(min_val..max_val);
                individual.push(random_val);
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



#[derive(Debug)]
pub enum VariableSubsetError {
    IncompatiblePopulationShape,
    PopulationEmpty,
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
}

