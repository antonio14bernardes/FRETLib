use super::constraints::*;
use super::multivariate_gaussian::CovMatrixType;
use super::variable_subsets::*;
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use std::collections::HashSet;
use std::ops::{Add, Sub, Div, Mul};
use nalgebra::{DMatrix, DVector};



#[derive(Debug)]
pub struct SetVarSubsets<T>
where T: Clone,
{
    subsets: Vec<VariableSubset<T>>,
    indices:  Vec<Vec<usize>>,
}

impl<T> SetVarSubsets<T> 
where T: Copy + Clone + Default + std::fmt::Debug,
{
    pub fn new_empty(indices: Vec<Vec<usize>>) -> Result<SetVarSubsets<T>, VariableSubsetError> {
        // Check that indices isn't empty
        if indices.is_empty() || indices.iter().any(|s| s.is_empty()) {
            return Err(VariableSubsetError::SubsetIndicesEmpty);
        }

        // Check that each index appears in only one subset
        let mut all_indices = HashSet::new();
        for subset_indices in &indices {
            for &index in subset_indices {
                if !all_indices.insert(index) {
                    return Err(VariableSubsetError::SubsetsAreNotIndependent);
                }
            }
        }

        // Check that each index appears in one subset
        let max = all_indices.iter().max().unwrap();
        let num_vars = all_indices.len();
        if *max != (num_vars - 1) {
            return Err(VariableSubsetError::InvalidIndices);
        }

        // Create the subsets
        let mut subsets = Vec::new();
        for subset_indices in &indices {
            // Create a VariableSubset and store the indices it contains
            let variable_subset = VariableSubset::new(&subset_indices);

            subsets.push(variable_subset);
        }
      

        Ok(Self{subsets, indices})

    }


    pub fn new(indices: Vec<Vec<usize>>, population: Vec<Vec<T>>, constraints_option: Option<Vec<OptimizationConstraint<T>>>) -> Result<Self, VariableSubsetError> {
        
        // Get a set of subsets where each subset only has the indices stored
        let mut set_subsets = Self::new_empty(indices)?;

        // Set the population
        set_subsets.set_population(population)?;
        
        // Set the constraints
        if let Some(constraints) = constraints_option {
            set_subsets.set_constraints(constraints)?;
        }
      

        Ok(set_subsets)
    }

    pub fn set_population(&mut self, population: Vec<Vec<T>>) -> Result<(), VariableSubsetError> {
        // Check that population isn't empty
        if population.is_empty() || population.iter().any(|p| p.is_empty()) {
            return Err(VariableSubsetError::PopulationEmpty);
        }

        // Check that the number of indices matches the number of variables in each individual of the population
        let num_vars = self.get_num_variables();
        
        if population.iter().any(|ind| ind.len() != num_vars) {
            return Err(VariableSubsetError::IncompatiblePopulationShape);
        }

        // Create the subsets
        for subset in self.subsets.iter_mut() {

            // Extract values corresponding to the indices from each individual in the population
            let subset_population: Vec<Vec<T>> = population
                .iter()
                .map(|individual| {
                    subset.get_indices()
                        .iter()
                        .map(|&idx| individual[idx])
                        .collect()
                })
                .collect();

            // Create a VariableSubset and store the relevant values
            subset.set_population(subset_population);
        }

        Ok(())


    }

    pub fn set_constraints(&mut self, constraints: Vec<OptimizationConstraint<T>>) -> Result<(), VariableSubsetError> {
        // Check if number of constraints equals the number of subsets
        if constraints.len() != self.indices.len() {
            return Err(VariableSubsetError::IncompatibleNumberOfConstraints)
        }

        for (subset, constraint) in self.subsets.iter_mut().zip(constraints) {
            subset.set_constraint(constraint)?;
        }

        Ok(())
    }

    pub fn get_subset_indices(&self) -> &Vec<Vec<usize>> {
        &self.indices
    }

    pub fn get_population(&self) -> Vec<Vec<T>> {
        let scrambled_population: Vec<Vec<Vec<T>>> = self.subsets.iter().map(|s| s.get_population().unwrap().to_vec()).collect();
        unscramble_population(&self.indices, &scrambled_population)
    }

    pub fn get_num_variables(&self) -> usize {
        *self.indices.iter().map(|subset| subset.iter().max().unwrap()).max().unwrap() + 1
    }

    pub fn get_subsets(&self) -> &Vec<VariableSubset<T>> {&self.subsets}

    pub fn get_subsets_mut(&mut self) -> &mut Vec<VariableSubset<T>> {&mut self.subsets}

    pub fn check_ready(&self) -> bool {
        self.subsets.iter().all(|s| s.check_ready())
    }

}

impl<T> SetVarSubsets<T> 
where
    T: Copy + Clone + Into<f64> + Default + PartialOrd
     + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> 
     + From<f64> + SampleUniform  + std::fmt::Debug,
{

    // Set the covariance matrix type for each subset
    pub fn set_covariance_matrix_types(&mut self, cov_types: Vec<CovMatrixType>) -> Result<(), VariableSubsetError> {
        if cov_types.len() != 1 && cov_types.len() != self.subsets.len() {
            return Err(VariableSubsetError::InvalidCovMatrixTypeslen);
        }

        if cov_types.len() == 1 {
            for subset in &mut self.subsets {
                subset.set_covariance_matrix_type(cov_types[0].clone())?;
            }
        } else {
            for (subset, cov_type) in self.subsets.iter_mut().zip(&cov_types) {
                subset.set_covariance_matrix_type(cov_type.clone())?;
            }
        }

        Ok(())
    }

    // pub fn set_covariance_matrix_types(&mut self)
    pub fn compute_distributions(&mut self) -> Result<(), VariableSubsetError> {
        self.subsets
        .iter_mut()
        .try_for_each(|subset| subset.compute_distribution())?;
         
        Ok(())
    }

    pub fn sample_individuals(&self, n: usize, rng: &mut impl Rng) -> Result<Vec<Vec<f64>>, VariableSubsetError> {
        let scrambled_population: Result<Vec<Vec<Vec<f64>>>, VariableSubsetError> = self
        .subsets
        .iter()
        .map(|subset| subset.sample_individuals(n, rng)) // Sample individuals
        .collect();

        let scrambled_population = scrambled_population?;

        // Unscramble the output to have it be in the original population shape
        let unscrambled_population = unscramble_population(&self.indices, &scrambled_population);

        Ok(unscrambled_population)
    }

    pub fn initialize_random_population(
        &mut self, 
        pop_size: usize, 
        rng: &mut impl Rng, 
        random_range: Option<(T, T)>, // Optional min/max range
    ) -> Result<(), VariableSubsetError> {
        
        println!("Constraints: {:?}", self.subsets);
        // Initialize all subsets with random population
        for subset in &mut self.subsets {
            subset.initialize_random_population(pop_size, rng, random_range)?;
        }
        
        Ok(())
    }

}

fn unscramble_population<T: Clone + Copy + Default + std::fmt::Debug>(indices: &Vec<Vec<usize>>, population: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {

    println!("Scrambled population shape ({},{},{})", population.len(), population[0].len(), population[1].len());
    // Find the number of individuals and the total number of variables from indices
    let num_individuals = population[0].len();
    let total_vars = indices.iter().flatten().max().unwrap() + 1;

    // Create a new vector of vectors with the same structure as the original population
    let mut original_population: Vec<Vec<T>> = vec![vec![T::default(); total_vars]; num_individuals];

    // Iterate through each subset and place its values back in the original structure
    for (subset_idx, subset_indices) in indices.iter().enumerate() {
        let subset_population = &population[subset_idx];

        // Iterate over individuals in the population
        for (individual_idx, individual) in subset_population.iter().enumerate() {
            for (j, &value) in individual.iter().enumerate() {
                let original_index = subset_indices[j];
                original_population[individual_idx][original_index] = value;
            }
        }
    }

    original_population
}


#[cfg(test)]
mod tests_set_var_subsets {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_new_empty_set_var_subsets() {
        let indices = vec![
            vec![0, 1, 2],
            vec![3, 4],
        ];

        let result = SetVarSubsets::<f64>::new_empty(indices.clone());
        assert!(result.is_ok(), "Failed to create SetVarSubsets with valid indices");

        let subsets = result.unwrap();
        assert_eq!(subsets.get_subset_indices(), &indices, "Subset indices mismatch");
    }

    #[test]
    fn test_new_set_var_subsets_with_population() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let result = SetVarSubsets::<f64>::new(indices.clone(), population.clone(), None);
        assert!(result.is_ok(), "Failed to create SetVarSubsets with population");

        let set_var_subsets = result.unwrap();
        let retrieved_population = set_var_subsets.get_population();

        assert_eq!(retrieved_population, population, "Population mismatch");
    }

    #[test]
    fn test_initialize_random_population() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();
        println!("Num subsets: {}", set_var_subsets.subsets.len());
        let mut rng = thread_rng();
        let pop_size = 5;
        let prob_size = 4;
        let random_range = Some((0.0, 10.0));

        let result = set_var_subsets.initialize_random_population(pop_size, &mut rng, random_range);
        assert!(result.is_ok(), "Failed to initialize random population");

        let population = set_var_subsets.get_population();
        println!("Got pop");
        assert_eq!(population.len(), pop_size, "Incorrect population size");
        assert_eq!(population[0].len(), prob_size, "Incorrect individual size in population");

        // Check if values are within the random range
        for individual in &population {
            for &val in individual {
                assert!(val >= 0.0 && val <= 10.0, "Value out of random range");
            }
        }
    }

    #[test]
    fn test_set_population() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();

        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let result = set_var_subsets.set_population(population.clone());
        assert!(result.is_ok(), "Failed to set population");

        let retrieved_population = set_var_subsets.get_population();
        assert_eq!(retrieved_population, population, "Population mismatch after setting");
    }
    
    #[test]
    fn test_set_constraints() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        // Create an empty SetVarSubsets
        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();

        // Define constraints for each subset
        let constraints: Vec<OptimizationConstraint<f64>> = vec![
            OptimizationConstraint::SumTo { sum: 1.0 },
            OptimizationConstraint::MaxMinValue { max: vec![0.8, 0.8], min: vec![0.2, 0.2] }
        ];

        // Set the constraints
        let result = set_var_subsets.set_constraints(constraints.clone());
        assert!(result.is_ok(), "Failed to set constraints");

        // Verify that the constraints have been correctly applied to each subset
        for (subset, constraint) in set_var_subsets.get_subsets().iter().zip(constraints.iter()) {
            match (&subset.get_constraint(), constraint) {
                (Some(OptimizationConstraint::SumTo { sum: s1 }), OptimizationConstraint::SumTo { sum: s2 }) => {
                    assert_eq!(s1, s2, "SumTo constraint not set correctly");
                }
                (Some(OptimizationConstraint::MaxMinValue { max: max1, min: min1 }), OptimizationConstraint::MaxMinValue { max: max2, min: min2 }) => {
                    assert_eq!(max1, max2, "MaxMinValue max constraint not set correctly");
                    assert_eq!(min1, min2, "MaxMinValue min constraint not set correctly");
                }
                _ => panic!("Unexpected constraint type or constraint was not set"),
            }
        }
    }

    #[test]
    fn test_compute_distributions() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new(indices, population, None).unwrap();

        let result = set_var_subsets.compute_distributions();
        assert!(result.is_ok(), "Failed to compute distributions");
    }

    #[test]
    fn test_sample_individuals() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new(indices, population, None).unwrap();

        // First, compute the distributions
        set_var_subsets.compute_distributions().unwrap();

        // Sample individuals from the distributions
        let mut rng = thread_rng();
        let result = set_var_subsets.sample_individuals(3, &mut rng);
        assert!(result.is_ok(), "Failed to sample individuals");

        let sampled_individuals = result.unwrap();
        assert_eq!(sampled_individuals.len(), 3, "Incorrect number of sampled individuals");
    }

    #[test]
    fn test_constraints_applied_in_random_initialization() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let constraints = vec![
            OptimizationConstraint::MaxValue { max: vec![5.0, 5.0] },
            OptimizationConstraint::MinValue { min: vec![2.0, 2.0] },
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();
        set_var_subsets.set_constraints(constraints).unwrap();

        let mut rng = thread_rng();
        let pop_size = 5;
        let random_range = Some((0.0, 10.0));

        let result = set_var_subsets.initialize_random_population(pop_size, &mut rng, random_range);
        assert!(result.is_ok(), "Failed to initialize random population with constraints");

        let population = set_var_subsets.get_population();
        println!("Population: {:?}", &population);

        for individual in &population {
            assert!(individual[0] <= 5.0 && individual[1] <= 5.0, "Max constraint violated");
            assert!(individual[2] >= 2.0 && individual[3] >= 2.0, "Min constraint violated");
        }
    }

    #[test]
    fn test_update_covariance_matrix_by_multiplying_individually() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new(indices, population, None).unwrap();

        // Create means and covariance matrices for each subset
        let mean1 = DVector::from_vec(vec![1.0, 2.0]);
        let cov1 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        let mean2 = DVector::from_vec(vec![3.0, 4.0]);
        let cov2 = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);

        // Set initial distributions manually for each subset
        let subsets = set_var_subsets.get_subsets_mut();
        subsets[0].set_distribution_manual(mean1.clone(), cov1.clone()).unwrap();
        subsets[1].set_distribution_manual(mean2.clone(), cov2.clone()).unwrap();

        // Now, update the covariance matrices by multiplying them by 1.1
        let scaling_factor = 1.1;

        for subset in set_var_subsets.get_subsets_mut() {
            if let Some((mean, cov)) = subset.get_distribution() {
                // Scale the covariance matrix by 1.1
                let updated_cov = cov * scaling_factor;

                // Update the distribution with the new scaled covariance matrix
                subset.set_distribution_manual(mean.clone(), updated_cov).unwrap();
            } else {
                panic!("Distribution not set correctly for subset");
            }
        }

        // Verify that the covariance matrices were updated correctly by checking each subset individually
        let expected_cov1 = cov1 * scaling_factor;
        let expected_cov2 = cov2 * scaling_factor;

        // Use `get_subsets` to get an immutable reference to each subset
        let subsets = set_var_subsets.get_subsets();

        // Check subset 1
        if let Some((_, cov)) = subsets[0].get_distribution() {
            assert_eq!(
                cov.as_slice(),
                expected_cov1.as_slice(),
                "Covariance matrix for subset 1 was not updated correctly"
            );
        } else {
            panic!("Distribution for subset 1 not found");
        }

        // Check subset 2
        if let Some((_, cov)) = subsets[1].get_distribution() {
            assert_eq!(
                cov.as_slice(),
                expected_cov2.as_slice(),
                "Covariance matrix for subset 2 was not updated correctly"
            );
        } else {
            panic!("Distribution for subset 2 not found");
        }
    }

    #[test]
    fn test_set_covariance_matrix_types() {
        let indices = vec![
            vec![0, 1],
            vec![2, 3],
        ];

        // Initialize SetVarSubsets with some population
        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        let mut set_var_subsets = SetVarSubsets::<f64>::new(indices, population, None).unwrap();

        // Test case 1: Set a single covariance matrix type for all subsets
        let cov_types = vec![CovMatrixType::Diagonal];  // Use Diagonal for all subsets
        let result = set_var_subsets.set_covariance_matrix_types(cov_types.clone());
        assert!(result.is_ok(), "Failed to set a single covariance matrix type for all subsets");

        // Ensure that all subsets have been set to Diagonal covariance matrix type
        for subset in set_var_subsets.get_subsets() {
            let cov_type = subset.get_covariance_matrix_type().unwrap();
            assert_eq!(cov_type, &CovMatrixType::Diagonal, "Covariance matrix type not set to Diagonal");
        }

        // Test case 2: Set individual covariance matrix types for each subset
        let cov_types = vec![CovMatrixType::Full, CovMatrixType::Diagonal];  // Different type for each subset
        let result = set_var_subsets.set_covariance_matrix_types(cov_types.clone());
        assert!(result.is_ok(), "Failed to set individual covariance matrix types for subsets");

        // Ensure that the correct types are applied
        for (subset, expected_type) in set_var_subsets.get_subsets().iter().zip(&cov_types) {
            let cov_type = subset.get_covariance_matrix_type().unwrap();
            assert_eq!(cov_type, expected_type, "Covariance matrix type not set correctly");
        }

        // Test case 3: Error case when the length of cov_types does not match the number of subsets
        let wrong_cov_types = vec![CovMatrixType::Full; 4];  // Not enough covariance types
        let result = set_var_subsets.set_covariance_matrix_types(wrong_cov_types);
        assert!(result.is_err(), "Expected an error due to mismatch in cov_types length and subsets length");
    }
}

