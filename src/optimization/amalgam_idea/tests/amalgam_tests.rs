use nalgebra::DMatrix;
use nalgebra::DVector;

use super::super::super::variable_subsets::VariableSubsetError;
use super::super::super::amalgam_idea::*;
use super::super::super::optimizer::*;
use super::super::super::constraints::*;
use super::super::AmalgamIdea;
use super::super::amalgam_parameters::AmalgamIdeaParameters;

#[derive(Debug, Clone)]
struct ToyFitness {
    fitness: f64
}

impl OptimizationFitness for ToyFitness {
    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}

fn toy_fitness_function(values: &[f64]) -> ToyFitness {
    let mut sum = 0.0;
    for value in values {
        sum += value*value;
    }
    ToyFitness{fitness: sum}
}


#[test]
fn test_evaluate_with_valid_fitness_function() {
    // Set up an instance of AmalgamIdea with a simple fitness function
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Test a sample solution
    let solution = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected_fitness = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0;

    // Evaluate the solution using the AmalgamIdea instance
    let result = amalgam.evaluate(&solution).expect("Evaluation failed").fitness;

    // Check if the result matches the expected fitness value
    assert_eq!(result, expected_fitness, "Fitness evaluation did not match expected value");
}

#[test]
fn test_set_parameters_with_subsets_single_parameter() {
    // Case: Subsets are set, and a single parameter is provided
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Manually define subsets
    let subset_indices = vec![vec![0, 1], vec![2, 3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Create a single AmalgamIdeaParameter
    let param = AmalgamIdeaParameters::new(
        50, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );

    // Attempt to set parameters with a single item in the vector
    let result = amalgam.set_parameters(param);

    // Verify that the result is Ok and the parameters are replicated for all subsets
    assert!(result.is_ok(), "Expected successful setting of parameters");
    assert!(amalgam.get_parameters().is_some(), "Parameters should be set");
}

#[test]
fn test_set_initial_population_with_valid_data() {
    // Case: Valid initial population with correct problem size
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Create a valid initial population where each individual matches the problem size
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];

    // Attempt to set the initial population
    let result = amalgam.set_initial_population(initial_population);

    // Verify that the result is Ok and the initial population is set
    assert!(result.is_ok(), "Expected successful setting of initial population");
    assert!(amalgam.get_initial_population().is_some(), "Initial population should be set");
    assert_eq!(
        amalgam.get_initial_population().unwrap().len(),
        2,
        "Expected the initial population to have 2 individuals"
    );
}

#[test]
fn test_set_initial_population_with_incorrect_sizes() {
    // Case: Initial population has individuals with incorrect sizes
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Create an initial population where one individual does not match the problem size
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0], // Correct size
        vec![5.0, 4.0, 3.0],           // Incorrect size
    ];

    // Attempt to set the initial population
    let result = amalgam.set_initial_population(initial_population);

    // Verify that the result is an error of type PopulationIncompatibleWithProblemSize
    assert!(
        matches!(result, Err(AmalgamIdeaError::PopulationIncompatibleWithProblemSize)),
        "Expected PopulationIncompatibleWithProblemSize error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_initial_population_with_subsets_defined() {
    // Case: Subsets are defined and the initial population is valid
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define subsets manually
    let subset_indices = vec![vec![0, 1], vec![2, 3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Create a valid initial population matching the problem size
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];

    // Attempt to set the initial population
    let result = amalgam.set_initial_population(initial_population);

    // Verify that the result is Ok and the initial population is set
    assert!(result.is_ok(), "Expected successful setting of initial population");
    assert!(amalgam.get_initial_population().is_some(), "Initial population should be set");
    assert_eq!(
        amalgam.get_initial_population().unwrap().len(),
        2,
        "Expected the initial population to have 2 individuals"
    );

    // Verify that the subsets have their population set
    for subset in amalgam.get_subsets().unwrap().get_subsets() {
        assert!(subset.get_population().is_some(), "Subset population should be set")
    }
}

#[test]
fn test_set_initial_population_with_subsets_not_defined() {
    // Case: Subsets are not defined, but the initial population is valid
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Create a valid initial population where each individual matches the problem size
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];

    // Attempt to set the initial population
    let result = amalgam.set_initial_population(initial_population);

    // Verify that the result is Ok and the initial population is set
    assert!(result.is_ok(), "Expected successful setting of initial population");
    assert!(amalgam.get_initial_population().is_some(), "Initial population should be set");
    assert_eq!(
        amalgam.get_initial_population().unwrap().len(),
        2,
        "Expected the initial population to have 2 individuals"
    );

}

#[test]
fn test_set_dependency_subsets_already_defined() {
    // Case: Trying to set subsets when they are already defined
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define initial subsets
    let initial_subset_indices = vec![vec![0, 1], vec![2, 3, 4]];
    amalgam.set_dependency_subsets(initial_subset_indices).unwrap();

    // Attempt to set subsets again
    let new_subset_indices = vec![vec![0, 2], vec![1, 3, 4]];
    let result = amalgam.set_dependency_subsets(new_subset_indices);

    // Verify that the result is an error because subsets are already defined
    assert!(
        matches!(result, Err(AmalgamIdeaError::SubsetError { err: VariableSubsetError::SubsetsAlreadyDefined })),
        "Expected SubsetsAlreadyDefined error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_dependency_subsets_with_valid_data() {
    // Case: Setting valid subsets for the defined problem size
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define valid subsets
    let valid_subset_indices = vec![vec![0, 1], vec![2, 3, 4]];

    // Attempt to set the dependency subsets
    let result = amalgam.set_dependency_subsets(valid_subset_indices.clone());

    // Verify that the result is Ok and the subsets are set correctly
    assert!(result.is_ok(), "Expected successful setting of dependency subsets");
    assert!(amalgam.get_subsets().is_some(), "Subsets should be set");
    assert_eq!(
        *amalgam.get_subsets().as_ref().unwrap().get_subset_indices(),
        valid_subset_indices,
        "Expected the subsets to match the provided valid subset indices"
    );
}

#[test]
fn test_set_dependency_subsets_with_invalid_data() {
    // Case: Setting invalid subsets where the indices are out of range
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define invalid subsets where an index is out of range
    let invalid_subset_indices = vec![vec![0, 1, 4], vec![2, 3, 5]]; // Index 5 is out of range for problem size 5

    // Attempt to set the dependency subsets
    let result = amalgam.set_dependency_subsets(invalid_subset_indices);

    // Verify that the result is an error because of incompatible subset indices
    assert!(
        matches!(result, Err(AmalgamIdeaError::SubsetsIncompatibleWithProblemSize)),
        "Expected SubsetsIncompatibleWithProblemSize error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_dependency_subsets_with_initial_population() {
    // Case: Setting subsets when the initial population is present
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Create a valid initial population
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];
    amalgam.set_initial_population(initial_population).unwrap();

    // Define valid subsets
    let valid_subset_indices = vec![vec![0, 1], vec![2, 3, 4]];

    // Attempt to set the dependency subsets
    let result = amalgam.set_dependency_subsets(valid_subset_indices.clone());

    // Verify that the result is Ok and the subsets are set correctly
    assert!(result.is_ok(), "Expected successful setting of dependency subsets with initial population");
    assert!(amalgam.get_subsets().is_some(), "Subsets should be set");
    assert_eq!(
        *amalgam.get_subsets().as_ref().unwrap().get_subset_indices(),
        valid_subset_indices,
        "Expected the subsets to match the provided valid subset indices"
    );

    // Verify that the subsets have their population set
    for subset in amalgam.get_subsets().unwrap().get_subsets() {
        assert!(subset.get_population().is_some(), "Subset population should be set")
    }
}


#[test]
fn test_set_constraints_with_subsets_defined() {
    // Case: Subsets are already defined, and valid constraints are provided
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define initial subsets
    let subset_indices = vec![vec![0, 1], vec![2, 3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Define valid constraints
    let constraints = vec![
        OptimizationConstraint::MaxValue { max: vec![3.0] },
        OptimizationConstraint::MinValue { min: vec![1.0] },
    ];

    // Attempt to set constraints
    let result = amalgam.set_constraints(constraints.clone());

    // Verify that the result is Ok
    assert!(result.is_ok(), "Expected successful setting of constraints");

    // Check if constraints are set correctly
    // Check if constraints are set correctly
    let set_constraints = amalgam.get_subsets().unwrap().get_constraints();
    for constraint in set_constraints {
        assert!(constraint.is_some(), "Constraint should have been set");
    }
}

#[test]
fn test_set_constraints_without_subsets_single_constraint() {
    // Case: Subsets are not defined, but a single constraint is provided
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define a single constraint
    let constraints = vec![OptimizationConstraint::SumTo { sum: 10.0 }];

    // Attempt to set constraints
    let result = amalgam.set_constraints(constraints.clone());

    // Verify that the result is Ok
    assert!(result.is_ok(), "Expected successful setting of constraints with a single constraint");
    assert!(amalgam.get_subsets().is_some(), "Expected subsets to be created");

    // Check if constraints are set correctly
    let set_constraints = amalgam.get_subsets().unwrap().get_constraints();
    for constraint in set_constraints {
        assert!(constraint.is_some(), "Constraint should have been set");
    }

    // Verify that a single subset was created for all variables
    let subset_indices = amalgam.get_subsets().unwrap().get_subset_indices();
    assert_eq!(
        subset_indices.len(),
        1,
        "Expected exactly one subset to be created"
    );
    assert_eq!(
        subset_indices[0].len(),
        problem_size,
        "Expected the subset to cover all variables"
    );
}

#[test]
fn test_set_constraints_without_subsets_multiple_constraints() {
    // Case: Subsets are not defined, and multiple constraints are provided
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Define multiple constraints
    let constraints = vec![
        OptimizationConstraint::MaxValue { max: vec![3.0] },
        OptimizationConstraint::MinValue { min: vec![1.0] },
    ];

    // Attempt to set constraints
    let result = amalgam.set_constraints(constraints);

    // Verify that the result is an error due to subsets not being defined
    assert!(
        matches!(result, Err(AmalgamIdeaError::VariableSubsetsNotDefined)),
        "Expected VariableSubsetsNotDefined error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_population_with_incorrect_number_of_individuals_after_parameters_set() {
    // Case: Set parameters first, then set an initial population with a different size
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Set the parameters with a population_size of 3
    let param = AmalgamIdeaParameters::new(
        3, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );
    amalgam.set_parameters(param).unwrap();

    // Create a population with a different number of individuals (e.g., 2 instead of 3)
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];

    // Attempt to set the initial population
    let result = amalgam.set_initial_population(initial_population);

    // Verify that the result is an error indicating a mismatch in population size
    assert!(
        matches!(result, Err(AmalgamIdeaError::PopulationIncompatibleWithParameters)),
        "Expected PopulationIncompatibleWithParameters error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_parameters_with_incorrect_population_size_after_population_set() {
    // Case: Set the population first, then set parameters with a different population_size
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the fitness function in the optimizer
    amalgam.set_fitness_function(toy_fitness_function);

    // Set a population with 4 individuals
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6.0, 5.0, 4.0, 3.0, 2.0],
    ];
    amalgam.set_initial_population(initial_population).unwrap();

    // Set the parameters with a different population_size (e.g., 3 instead of 4)
    let param = AmalgamIdeaParameters::new(
        3, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );
    let result = amalgam.set_parameters(param);

    // Verify that the result is an error indicating a mismatch in population size
    assert!(
        matches!(result, Err(AmalgamIdeaError::PopulationIncompatibleWithParameters)),
        "Expected PopulationIncompatibleWithParameters error, but got: {:?}",
        result
    );
}

#[test]
fn test_set_initial_distribution() {
    // Problem size and subsets for a 4-variable problem
    let problem_size = 4;
    let indices = vec![
        vec![0, 1], // First subset for variables 0 and 1
        vec![2, 3], // Second subset for variables 2 and 3
    ];

    // Instantiate AmalgamIdea and set subsets
    let mut amalgam_idea = AmalgamIdea::new(problem_size, false);
    // Set the fitness function in the optimizer
    amalgam_idea.set_fitness_function(toy_fitness_function);

    amalgam_idea.set_dependency_subsets(indices).unwrap();

    // Define mean vectors and covariance matrices for each subset
    let mean1 = DVector::from_vec(vec![1.0, 2.0]);
    let cov1 = DMatrix::from_vec(2, 2, vec![1.0, 0.2, 0.2, 1.5]);

    let mean2 = DVector::from_vec(vec![3.0, 4.0]);
    let cov2 = DMatrix::from_vec(2, 2, vec![2.0, 0.3, 0.3, 2.0]);

    // Set initial distribution for each subset
    let result = amalgam_idea.set_initial_distribution(&vec![mean1.clone(), mean2.clone()], &vec![cov1.clone(), cov2.clone()]);
    assert!(result.is_ok(), "Failed to set initial distribution");

    // Retrieve and verify the distribution from each subset in SetVarSubsets
    if let Some(subsets) = amalgam_idea.get_subsets() {
        for (i, subset) in subsets.get_subsets().iter().enumerate() {
            if let Some((stored_mean, stored_cov)) = subset.get_distribution() {
                let expected_mean = if i == 0 { &mean1 } else { &mean2 };
                let expected_cov = if i == 0 { &cov1 } else { &cov2 };

                assert_eq!(stored_mean, expected_mean, "Mean vector for subset {} does not match", i);
                assert_eq!(stored_cov, expected_cov, "Covariance matrix for subset {} does not match", i);
            } else {
                panic!("Distribution not set correctly for subset {}", i);
            }
        }
    } else {
        panic!("Subsets not initialized correctly in AmalgamIdea");
    }

    println!("Initial distributions set successfully for all subsets.");
}

#[test]
fn test_set_pop_size_manual() {
    // Problem size and subsets for a 4-variable problem
    let problem_size = 4;

    let toy_fit = |ind: &[f64]| {
        let sum: f64 = ind.iter().sum();
        -sum
    };

    // Instantiate AmalgamIdea and set subsets
    let mut amalgam_idea = AmalgamIdea::new(problem_size, false);
    // Set the fitness function in the optimizer
    amalgam_idea.set_fitness_function(toy_fitness_function);
    
    let pop_size = 100;
    amalgam_idea.set_population_size(pop_size).unwrap();

    amalgam_idea.initialize().unwrap();

    let params = amalgam_idea.get_parameters().unwrap();

    assert_eq!(params.population_size, pop_size);

}