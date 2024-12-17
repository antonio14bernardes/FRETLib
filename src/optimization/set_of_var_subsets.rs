use super::constraints::*;
use super::multivariate_gaussian::CovMatrixType;
use super::variable_subsets::*;
use rand::Rng;
use std::collections::HashSet;
use std::ops::{Add, Sub, Div, Mul};
use nalgebra::{DMatrix, DVector};
use std::fmt::Debug;



#[derive(Debug, Clone)]
pub struct SetVarSubsets<T>
where T: Clone,
{
    subsets: Vec<VariableSubset<T>>,
    indices:  Vec<Vec<usize>>,
}

impl<T> SetVarSubsets<T> 
where T: Copy + Clone + Default + Debug,
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
    
        // Use the scramble_population function to get the scrambled subsets
        let scrambled_population = scramble_population(&self.indices, &population);
    
        // Set the population for each subset using the scrambled population
        for (subset, scrambled_subset_population) in self.subsets.iter_mut().zip(scrambled_population) {
            subset.set_population(scrambled_subset_population);
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

    pub fn get_num_subsets(&self) -> usize {
        self.subsets.len()
    }

    pub fn get_subsets(&self) -> &Vec<VariableSubset<T>> {&self.subsets}

    pub fn get_subsets_mut(&mut self) -> &mut Vec<VariableSubset<T>> {&mut self.subsets}

    pub fn get_constraints(&self) -> Vec<Option<&OptimizationConstraint<T>>> {
        let mut vec_cosntraints = Vec::new();

        for subset in &self.subsets {
            vec_cosntraints.push(subset.get_constraint())
        }

        vec_cosntraints
    }

    pub fn check_ready(&self) -> bool {
        self.subsets.iter().all(|s| s.check_ready())
    }

}

impl<T> SetVarSubsets<T>
where
    T: Default + Copy + Clone + PartialOrd + 
    Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + 
    From<f64> + Into<f64> + Debug,
{
    pub fn new(indices: Vec<Vec<usize>>, population: Vec<Vec<T>>, constraints_option: Option<Vec<OptimizationConstraint<T>>>) -> Result<Self, VariableSubsetError> {
        
        // Get a set of subsets where each subset only has the indices stored
        let mut set_subsets = SetVarSubsets::<T>::new_empty(indices)?;

        // Set the population
        set_subsets.set_population(population)?;
        
        // Set the constraints
        if let Some(constraints) = constraints_option {
            set_subsets.set_constraints(constraints)?;
        }
      

        Ok(set_subsets)
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

    pub fn enforce_constraints(&mut self) -> Result<(), VariableSubsetError> {
        for subset in self.subsets.iter_mut() {
            subset.enforce_constraint()?;
        }

        Ok(())
    }

    pub fn enforce_constraint_external(&self, values: &Vec<Vec<T>>) -> Result<Vec<Vec<T>>, VariableSubsetError> {
        if values[0].len() != self.get_num_variables() {return Err(VariableSubsetError::IncompatiblePopulationShape)}
        let scrambled_values = scramble_population(&self.indices, values);
        
        let mut corrected_scrambled: Vec<Vec<Vec<T>>> = Vec::new();
        let subsets = &self.subsets;
        for (subset_values, subset) in scrambled_values.iter().zip(subsets) {
            if let Some(constraint) = subset.get_constraint() {
                let corrected_single = enforce_constraint_external(subset_values, constraint);
                corrected_scrambled.push(corrected_single);
            } else {
                corrected_scrambled.push(subset_values.clone());
            }
            
        }

        let unscrambled_values = unscramble_population(&self.indices, &corrected_scrambled);

        Ok(unscrambled_values)
    }

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

    // Set the distribution for the subsets manually
    pub fn set_distribution_manual(&mut self, means: &Vec<DVector<f64>>, covs: &Vec<DMatrix<f64>>) -> Result<(), VariableSubsetError> {
        // Check that the lengths of means and covariances match
        if means.len() != covs.len() {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
        
        // Verify that each mean vector's length matches the dimensions of its corresponding covariance matrix
        if means.iter().zip(covs).any(|(mean, cov)| mean.len() != cov.nrows() || cov.nrows() != cov.ncols()) {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
        
        // Check that the number of subsets matches the number of mean-covariance pairs
        if self.get_num_subsets() != means.len() {
            return Err(VariableSubsetError::IncompatibleInputSizes);
        }
    
        // If all checks pass, set the distributions for each subset
        for (i, subset) in self.subsets.iter_mut().enumerate() {
            subset.set_distribution_manual(means[i].clone(), covs[i].clone())?;
        }
    
        Ok(())
    }

    pub fn compute_distributions(&mut self) -> Result<(), VariableSubsetError> {
        self.subsets
        .iter_mut()
        .try_for_each(|subset| subset.compute_distribution())?;
         
        Ok(())
    }

    pub fn sample_individuals(&self, n: usize, rng: &mut impl Rng) -> Result<Vec<Vec<T>>, VariableSubsetError> {
        let scrambled_population: Result<Vec<Vec<Vec<T>>>, VariableSubsetError> = self
        .subsets
        .iter()
        .map(|subset| subset.sample_individuals(n, rng)) // Sample individuals
        .collect();

        let scrambled_population = scrambled_population?;

        // Unscramble the output to have it be in the original population shape
        let unscrambled_population = unscramble_population(&self.indices, &scrambled_population);

        Ok(unscrambled_population)
    }

    // Initialize population based on set distribution
    pub fn initialize_population(&mut self, pop_size: usize, rng: &mut impl Rng) -> Result<(), VariableSubsetError> {
        // Initialize all subsets' population
        for subset in &mut self.subsets {
            subset.initialize_population(pop_size, rng)?;
            
        }
        
        Ok(())
    }

    pub fn initialize_random_population(
        &mut self, 
        pop_size: usize, 
        rng: &mut impl Rng, 
        random_range: Option<(T, T)>, // Optional min/max range
    ) -> Result<(), VariableSubsetError> {
        
        // Initialize all subsets with random population
        for subset in &mut self.subsets {
            subset.initialize_random_population(pop_size, rng, random_range)?;
        }
        
        Ok(())
    }

}

pub fn unscramble_population<T: Clone + Copy + Default + std::fmt::Debug>(indices: &Vec<Vec<usize>>, population: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {

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

pub fn unscramble_means(indices: &Vec<Vec<usize>>, means: &Vec<DVector<f64>>) -> DVector<f64> {
    // Find the total number of variables from indices
    let total_vars = indices.iter().flatten().max().unwrap() + 1;

    // Create a new DVector with the total number of variables, initialized to zeros
    let mut original_mean = DVector::zeros(total_vars);

    // Iterate through each subset and place its values back in the original structure
    for (subset_idx, subset_indices) in indices.iter().enumerate() {
        let subset_mean = &means[subset_idx];

        // Iterate over values in the subset
        for (j, &value) in subset_mean.iter().enumerate() {
            let original_index = subset_indices[j];
            original_mean[original_index] = value;
        }
    }

    original_mean
}

pub fn scramble_population<T: Clone + Copy + Default + std::fmt::Debug>(
    indices: &Vec<Vec<usize>>,
    original_population: &Vec<Vec<T>>,
) -> Vec<Vec<Vec<T>>> {
    // Create a scrambled population structure based on the subset indices
    let mut scrambled_population: Vec<Vec<Vec<T>>> = indices
        .iter()
        .map(|subset_indices| vec![vec![T::default(); subset_indices.len()]; original_population.len()])
        .collect();

    // Iterate over each subset and extract values from the original structure
    for (subset_idx, subset_indices) in indices.iter().enumerate() {
        let subset_population = &mut scrambled_population[subset_idx];

        // Iterate over each individual in the original population
        for (individual_idx, original_individual) in original_population.iter().enumerate() {
            for (j, &original_index) in subset_indices.iter().enumerate() {
                subset_population[individual_idx][j] = original_individual[original_index];
            }
        }
    }

    scrambled_population
}

pub fn scramble_means(
    indices: &Vec<Vec<usize>>,
    original_mean: &DVector<f64>,
) -> Vec<DVector<f64>> {
    // Create scrambled means structure based on the subset indices
    let mut scrambled_means: Vec<DVector<f64>> = indices
        .iter()
        .map(|subset_indices| DVector::zeros(subset_indices.len()))
        .collect();

    // Iterate over each subset and extract values from the original mean
    for (subset_idx, subset_indices) in indices.iter().enumerate() {
        let subset_mean = &mut scrambled_means[subset_idx];

        // Iterate over each index in the subset and extract the corresponding value from the original mean
        for (j, &original_index) in subset_indices.iter().enumerate() {
            subset_mean[j] = original_mean[original_index];
        }
    }

    scrambled_means
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
            println!("Individual: {:?}", individual);
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

    #[test]
    fn test_enforce_constraints() {
        // Define indices for the subsets
        let indices = vec![
            vec![0, 1], // First subset contains variables 0 and 1
            vec![2, 3], // Second subset contains variables 2 and 3
        ];

        // Create an initial population
        let population = vec![
            vec![1.0, 2.0, 3.0, 4.0], // First individual
            vec![5.0, 6.0, 7.0, 8.0], // Second individual
        ];

        // Initialize SetVarSubsets with population and no constraints
        let mut set_var_subsets = SetVarSubsets::<f64>::new(indices, population, None).unwrap();

        // Define constraints for each subset
        let constraints = vec![
            OptimizationConstraint::MaxValue { max: vec![2.0, 2.0] },  // Constraint for the first subset
            OptimizationConstraint::MinValue { min: vec![4.0, 4.0] },  // Constraint for the second subset
        ];

        // Set the constraints
        set_var_subsets.set_constraints(constraints).unwrap();

        // Enforce the constraints
        let result = set_var_subsets.enforce_constraints();
        assert!(result.is_ok(), "Failed to enforce constraints");

        // Retrieve the modified population after enforcing constraints
        let enforced_population = set_var_subsets.get_population();

        // Check if constraints are enforced correctly for the first subset (MaxValue)
        for individual in &enforced_population {
            assert!(individual[0] <= 2.0, "Value for variable 0 exceeds maximum constraint");
            assert!(individual[1] <= 2.0, "Value for variable 1 exceeds maximum constraint");
        }

        // Check if constraints are enforced correctly for the second subset (MinValue)
        for individual in &enforced_population {
            assert!(individual[2] >= 4.0, "Value for variable 2 is below minimum constraint");
            assert!(individual[3] >= 4.0, "Value for variable 3 is below minimum constraint");
        }
    }

    #[test]
    fn test_unscramble_population() {
        let indices = vec![
            vec![0, 1],  // First subset corresponds to variables 0 and 1
            vec![2, 3],  // Second subset corresponds to variables 2 and 3
        ];

        // Example scrambled population
        let scrambled_population = vec![
            vec![ // First subset's population
                vec![1.0, 2.0], // Individual 1
                vec![5.0, 6.0], // Individual 2
            ],
            vec![ // Second subset's population
                vec![3.0, 4.0], // Individual 1
                vec![7.0, 8.0], // Individual 2
            ],
        ];

        // Expected unscrambled population
        let expected_population = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Individual 1
            vec![5.0, 6.0, 7.0, 8.0], // Individual 2
        ];

        // Perform unscrambling
        let result = unscramble_population(&indices, &scrambled_population);

        // Check if the result matches the expected population
        assert_eq!(result, expected_population, "Unscrambled population does not match the expected population");
    }

    #[test]
    fn test_unscramble_population_with_non_contiguous_indices() {
        let indices = vec![
            vec![0, 2],  // First subset corresponds to variables 0 and 2
            vec![1, 3],  // Second subset corresponds to variables 1 and 3
        ];

        // Example scrambled population
        let scrambled_population = vec![
            vec![
                vec![1.0, 3.0], // Individual 1
                vec![5.0, 7.0], // Individual 2
            ],
            vec![
                vec![2.0, 4.0], // Individual 1
                vec![6.0, 8.0], // Individual 2
            ],
        ];

        // Expected unscrambled population
        let expected_population = vec![
            vec![1.0, 2.0, 3.0, 4.0], // Individual 1
            vec![5.0, 6.0, 7.0, 8.0], // Individual 2
        ];

        // Perform unscrambling
        let result = unscramble_population(&indices, &scrambled_population);

        // Check if the result matches the expected population
        assert_eq!(result, expected_population, "Unscrambled population does not match the expected population");
    }

    #[test]
    fn test_unscramble_means() {
        let indices = vec![
            vec![0, 1],  // First subset corresponds to variables 0 and 1
            vec![2, 3],  // Second subset corresponds to variables 2 and 3
        ];

        // Example scrambled means
        let scrambled_means = vec![
            DVector::from_vec(vec![1.0, 2.0]), // First subset's mean
            DVector::from_vec(vec![3.0, 4.0]), // Second subset's mean
        ];

        // Expected unscrambled mean
        let expected_mean = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Perform unscrambling
        let result = unscramble_means(&indices, &scrambled_means);

        // Check if the result matches the expected mean
        assert_eq!(result, expected_mean, "Unscrambled mean does not match the expected mean");
    }

    #[test]
    fn test_unscramble_means_with_non_contiguous_indices() {
        let indices = vec![
            vec![0, 2],  // First subset corresponds to variables 0 and 2
            vec![1, 3],  // Second subset corresponds to variables 1 and 3
        ];

        // Example scrambled means
        let scrambled_means = vec![
            DVector::from_vec(vec![1.0, 3.0]), // First subset's mean
            DVector::from_vec(vec![2.0, 4.0]), // Second subset's mean
        ];

        // Expected unscrambled mean
        let expected_mean = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Perform unscrambling
        let result = unscramble_means(&indices, &scrambled_means);

        // Check if the result matches the expected mean
        assert_eq!(result, expected_mean, "Unscrambled mean does not match the expected mean");
    }

    #[test]
    fn test_scramble_population_non_contiguous() {
        // Non-contiguous subset indices
        let indices = vec![vec![0, 2], vec![1, 3]];

        // Original population with 4 variables
        let original_population = vec![
            vec![1.0, 2.0, 3.0, 4.0],  // First individual
            vec![5.0, 6.0, 7.0, 8.0],  // Second individual
        ];

        // Expected scrambled population
        let expected_scrambled_population = vec![
            vec![vec![1.0, 3.0], vec![5.0, 7.0]],  // Subset 0 (indices [0, 2])
            vec![vec![2.0, 4.0], vec![6.0, 8.0]],  // Subset 1 (indices [1, 3])
        ];

        // Perform scrambling
        let scrambled_population = scramble_population(&indices, &original_population);

        // Check if the scrambled population matches the expected result
        assert_eq!(
            scrambled_population, expected_scrambled_population,
            "Scrambled population does not match expected result"
        );
    }

    #[test]
    fn test_scramble_means_non_contiguous() {
        // Non-contiguous subset indices
        let indices = vec![vec![0, 2], vec![1, 3]];

        // Original mean vector with 4 variables
        let original_mean = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Expected scrambled means
        let expected_scrambled_means = vec![
            DVector::from_vec(vec![1.0, 3.0]),  // Subset 0 (indices [0, 2])
            DVector::from_vec(vec![2.0, 4.0]),  // Subset 1 (indices [1, 3])
        ];

        // Perform scrambling
        let scrambled_means = scramble_means(&indices, &original_mean);

        // Check if the scrambled means match the expected result
        assert_eq!(
            scrambled_means, expected_scrambled_means,
            "Scrambled means do not match expected result"
        );
    }

    #[test]
    fn test_enforce_constraint_external() {
        // Define indices for the subsets
        let indices = vec![
            vec![0, 1], // First subset for variables 0 and 1
            vec![2, 3], // Second subset for variables 2 and 3
        ];

        // Create SetVarSubsets with empty population and constraints for testing
        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();

        // Define constraints: Max for the first subset, Min for the second subset
        let constraints = vec![
            OptimizationConstraint::MaxValue { max: vec![3.0, 4.0] }, // Max values for subset 1
            OptimizationConstraint::MinValue { min: vec![5.0, 6.0] }, // Min values for subset 2
        ];

        // Set constraints to the SetVarSubsets instance
        set_var_subsets.set_constraints(constraints).unwrap();

        // External population to enforce constraints on
        let external_population = vec![
            vec![4.0, 5.0, 3.0, 2.0], // Individual 1
            vec![2.0, 6.0, 1.0, 7.0], // Individual 2
        ];

        // Enforce constraints externally
        let result = set_var_subsets.enforce_constraint_external(&external_population);
        assert!(result.is_ok(), "Failed to enforce constraints on external population");

        // Retrieve the corrected population
        let corrected_population = result.unwrap();

        // Validate corrected population against constraints
        for individual in &corrected_population {
            // Check for subset 1 max constraints
            assert!(individual[0] <= 3.0, "Value at index 0 exceeds max constraint of 3.0");
            assert!(individual[1] <= 4.0, "Value at index 1 exceeds max constraint of 4.0");

            // Check for subset 2 min constraints
            assert!(individual[2] >= 5.0, "Value at index 2 is below min constraint of 5.0");
            assert!(individual[3] >= 6.0, "Value at index 3 is below min constraint of 6.0");
        }

        println!("Corrected population after enforcing constraints: {:?}", corrected_population);
    }


    #[test]
    fn test_set_distribution_manual() {
        // Define subset indices
        let indices = vec![
            vec![0, 1], // Subset for variables 0 and 1
            vec![2, 3], // Subset for variables 2 and 3
        ];

        // Create a SetVarSubsets instance with empty constraints
        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices).unwrap();

        // Define means and covariance matrices for each subset
        let mean1 = DVector::from_vec(vec![1.5, 2.5]);
        let cov1 = DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);

        let mean2 = DVector::from_vec(vec![3.5, 4.5]);
        let cov2 = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);

        // Apply manual distribution settings
        let means = vec![mean1.clone(), mean2.clone()];
        let covs = vec![cov1.clone(), cov2.clone()];

        let result = set_var_subsets.set_distribution_manual(&means, &covs).unwrap();
        // assert!(result.is_ok(), "Failed to manually set distributions");

        // Verify that distributions were set correctly for each subset
        let subsets = set_var_subsets.get_subsets();
        for (i, subset) in subsets.iter().enumerate() {
            if let Some((stored_mean, stored_cov)) = subset.get_distribution() {
                assert_eq!(
                    stored_mean, &means[i],
                    "Mean for subset {} does not match the manually set mean",
                    i
                );
                assert_eq!(
                    stored_cov, &covs[i],
                    "Covariance matrix for subset {} does not match the manually set covariance",
                    i
                );
            } else {
                panic!("Distribution not set correctly for subset {}", i);
            }
        }

        println!("Manual distributions set successfully for all subsets.");
    }

    #[test]
    fn test_initialize_population_with_distributions() {
        // Define subset indices
        let indices = vec![
            vec![0, 1], // First subset for variables 0 and 1
            vec![2, 3], // Second subset for variables 2 and 3
        ];

        // Create an empty SetVarSubsets instance
        let mut set_var_subsets = SetVarSubsets::<f64>::new_empty(indices.clone()).unwrap();

        // Define mean vectors and covariance matrices for each subset
        let mean1 = DVector::from_vec(vec![1.0, 2.0]);
        let cov1 = DMatrix::from_vec(2, 2, vec![1.0, 0.3, 0.3, 1.5]);

        let mean2 = DVector::from_vec(vec![3.0, 4.0]);
        let cov2 = DMatrix::from_vec(2, 2, vec![2.0, 0.5, 0.5, 2.0]);

        // Manually set the distributions for each subset
        set_var_subsets.set_distribution_manual(&vec![mean1.clone(), mean2.clone()], &vec![cov1.clone(), cov2.clone()]).unwrap();

        // Define the population size
        let pop_size = 1000000;
        let mut rng = thread_rng();

        // Initialize population based on the set distributions
        let result = set_var_subsets.initialize_population(pop_size, &mut rng);
        assert!(result.is_ok(), "Failed to initialize population from distributions");

        // Retrieve the initialized population
        let population = set_var_subsets.get_population();

        // Check if the population has the correct number of individuals
        assert_eq!(population.len(), pop_size, "Population size does not match the expected size");

        // Check that each individual has the correct number of variables
        let num_vars = set_var_subsets.get_num_variables();
        for individual in &population {
            assert_eq!(individual.len(), num_vars, "Individual length does not match the number of variables");
        }

        // Optional: Check if the mean of the population is approximately close to the set means (within tolerance)
        // Only a rough check due to randomness, suitable for testing distributions approximately
        let mean_estimate: Vec<f64> = (0..num_vars)
            .map(|i| population.iter().map(|ind| ind[i]).sum::<f64>() / pop_size as f64)
            .collect();
        println!("Mean estimate: {:?}", mean_estimate);
        // Retrieve original full mean vector from scrambled subset means
        let expected_mean = unscramble_means(&indices, &vec![mean1.clone(), mean2.clone()]);
        for (i, mean_val) in expected_mean.iter().enumerate() {
            assert!(
                (mean_estimate[i] - mean_val).abs() < 0.01,
                "Estimated mean for variable {} differs significantly from the expected mean",
                i
            );
        }
    }
}

