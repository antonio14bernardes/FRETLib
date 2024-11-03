use super::super::super::variable_subsets::VariableSubsetError;
use super::super::super::amalgam_idea::*;
use super::super::super::optimizer::*;
use super::super::super::constraints::*;
use super::super::amalgam_parameters::AmalgamIdeaParameters;

use nalgebra::{DMatrix, DVector};

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
        sum += value;
    }
    ToyFitness{fitness: sum}
}

#[test]
fn test_initialize_without_any_setup() {
    // Case: No subsets, parameters, or initial population are defined
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the toy fitness function
    amalgam.set_fitness_function(toy_fitness_function);

    // Attempt to initialize
    let result = amalgam.initialize();

    // Verify that the result is Ok
    assert!(result.is_ok(), "Expected successful initialization");

    // Check if subsets have been initialized
    assert!(amalgam.get_subsets().is_some(), "Expected subsets to be initialized");
    let subsets = amalgam.get_subsets().unwrap();

    assert_eq!(
        subsets.get_subset_indices().len(),
        1,
        "Expected a single subset covering all variables"
    );

    // Check that subsets have their distributions initialized
    for subset in subsets.get_subsets() {
        assert!(subset.get_distribution().is_some());
    }

    // Check if parameters have been set
    assert!(amalgam.get_parameters().is_some(), "Expected parameters to be set automatically");

    // Check if the initial population has been set
    assert!(amalgam.get_initial_population().is_some(), "Expected initial population to be generated");
    assert!(amalgam.get_current_population().len() > 0, "Expected current population to be initialized");

    // Very best solution is set
    assert!(amalgam.get_best_solution().is_some());
}

#[test]
fn test_initialize_with_predefined_subsets_and_population() {
    // Case: Subsets and initial population are defined, but parameters are not
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Define subsets manually
    let subset_indices = vec![vec![0, 1], vec![2, 3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Create a predefined initial population
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
    ];
    amalgam.set_initial_population(initial_population.clone()).unwrap();

    // Set the toy fitness function
    amalgam.set_fitness_function(toy_fitness_function);

    // Attempt to initialize
    let result = amalgam.initialize();
    println!("Error: {:?}", result);

    // Verify that the result is Ok
    assert!(result.is_ok(), "Expected successful initialization with predefined subsets and population");

    // Check if subsets have been kept as defined
    assert!(amalgam.get_subsets().is_some(), "Expected subsets to remain defined");
    let subsets = amalgam.get_subsets().unwrap();
    assert_eq!(
        subsets.get_subset_indices().len(),
        2,
        "Expected two predefined subsets to be present"
    );

    // Check that subsets have their distributions initialized
    for subset in subsets.get_subsets() {
        assert!(subset.get_distribution().is_some());
    }

    // Check if parameters have been set
    assert!(amalgam.get_parameters().is_some(), "Expected parameters to be set automatically");

    // Check if the initial population matches the predefined one
    assert_eq!(
        *amalgam.get_initial_population().as_ref().unwrap(),
        &initial_population,
        "Expected the initial population to match the predefined population"
    );

    // Verify that the current population is also correctly set
    assert_eq!(
        *amalgam.get_current_population(),
        initial_population,
        "Expected the current population to be the same as the initial population"
    );

    // Very best solution is set
    assert!(amalgam.get_best_solution().is_some());

    // Verify subsets 
}

#[test]
fn test_initialize_with_only_predefined_parameters() {
    // Case: Parameters are defined manually, but subsets and initial population are not
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Define custom parameters
    let custom_parameters = AmalgamIdeaParameters::new(
        50, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );
    amalgam.set_parameters(custom_parameters).unwrap();

    // Set the toy fitness function
    amalgam.set_fitness_function(toy_fitness_function);

    // Attempt to initialize
    let result = amalgam.initialize();
    println!("Error: {:?}", result);

    // Verify that the result is Ok
    assert!(result.is_ok(), "Expected successful initialization with predefined parameters");

    // Check if subsets have been initialized
    assert!(amalgam.get_subsets().is_some(), "Expected subsets to be initialized");
    let subsets = amalgam.get_subsets().unwrap();
    assert_eq!(
        subsets.get_subset_indices().len(),
        1,
        "Expected a single subset covering all variables"
    );

    // Check that subsets have their distributions initialized
    for subset in subsets.get_subsets() {
        assert!(subset.get_distribution().is_some());
    }

    // Check if the initial population has been set
    assert!(amalgam.get_initial_population().is_some(), "Expected initial population to be generated");
    assert!(amalgam.get_current_population().len() > 0, "Expected current population to be initialized");

    // Very best solution is set
    assert!(amalgam.get_best_solution().is_some());
}

#[test]
fn test_selection() {
    // Case: Selection based on current population and fitness values
    let problem_size = 5;
    let iter_memory = false;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set the toy fitness function
    amalgam.set_fitness_function(toy_fitness_function);

    // Define custom parameters with tau to select 50% of the population
    let custom_parameters = AmalgamIdeaParameters::new(
        6, // population size
        0.5, // tau
        1.1, 0.9, 0.1, 1e-4, 0.2, 0.3, 2.0, 25,
    );
    amalgam.set_parameters(custom_parameters).unwrap();

    // Create a predefined initial population
    let initial_population = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0], // Fitness = 15.0
        vec![5.0, 4.0, 3.0, 2.0, 1.0], // Fitness = 15.0
        vec![0.5, 1.5, 2.5, 3.5, 4.5], // Fitness = 12.5
        vec![2.0, 2.0, 2.0, 2.0, 2.0], // Fitness = 10.0
        vec![0.0, 0.0, 0.0, 0.0, 0.0], // Fitness = 0.0
        vec![3.0, 3.0, 3.0, 3.0, 3.0], // Fitness = 15.0
    ];
    amalgam.set_initial_population(initial_population.clone()).unwrap();

    // Initialize the algorithm to set up the population and fitnesses
    amalgam.initialize().unwrap();

    // Perform the selection
    let result = amalgam.selection();
    assert!(result.is_ok(), "Expected successful selection");

    // Check if the latest selection is correctly set
    let latest_selection = result.unwrap();

    // Calculate the expected number of selected individuals
    let expected_num_to_select = (0.5 * 6.0) as usize;
    assert_eq!(
        latest_selection.len(),
        expected_num_to_select,
        "Expected the number of selected individuals to match the calculated number"
    );

    // Verify that the selected individuals are the top ones based on fitness
    // The best fitness values are 15.0 (first, second, and sixth individuals)
    let expected_selection = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0], // Fitness = 15.0
        vec![5.0, 4.0, 3.0, 2.0, 1.0], // Fitness = 15.0
        vec![3.0, 3.0, 3.0, 3.0, 3.0], // Fitness = 15.0
    ];
    assert_eq!(
        latest_selection, expected_selection,
        "Expected the selected individuals to match the highest fitness values"
    );
}

#[test]
fn test_update_distribution() {
    // Setup the problem
    let problem_size = 5;
    let iter_memory = true;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Define a simple fitness function (sum of values)
    amalgam.set_fitness_function(toy_fitness_function);

    // Initialize subsets manually
    let subset_indices = vec![vec![0, 1, 2], vec![3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Define custom parameters with memory enabled
    let custom_parameters = AmalgamIdeaParameters::new(
        50, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );
    amalgam.set_parameters(custom_parameters).unwrap();

    // Initialize the optimizer
    amalgam.initialize().unwrap();

    // Store the previous distributions
    let prev_dists: Vec<(DVector<f64>, DMatrix<f64>)> = amalgam
        .get_subsets()
        .unwrap()
        .get_subsets()
        .iter()
        .map(|subset| {
            let (mean, cov) = subset.get_distribution().unwrap();
            (mean.clone(), cov.clone())
        })
        .collect();

    // Perform a selection
    let selection = amalgam.selection().unwrap();

    // Update the distribution
    amalgam.update_distribution(selection).unwrap();

    // Check if the new distributions are different from the previous distributions
    let new_dists: Vec<(DVector<f64>, DMatrix<f64>)> = amalgam
            .get_subsets()
            .unwrap()
            .get_subsets()
            .iter()
            .map(|subset| {
                let (mean, cov) = subset.get_distribution().unwrap();
                (mean.clone(), cov.clone())
            })
            .collect();

    // Verify that at least one subset's mean or covariance has changed
    let mut distribution_changed = false;
    for (prev, new) in prev_dists.iter().zip(new_dists.iter()) {
        if prev.0 != new.0 || prev.1 != new.1 {
            distribution_changed = true;
            break;
        }
    }

    assert!(
        distribution_changed,
        "Expected the distributions to change after updating, but they did not"
    );
}

#[test]
fn test_update_population() {
    // Setup the problem
    let problem_size = 5;
    let iter_memory = true;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Define a simple fitness function (sum of values)
    amalgam.set_fitness_function(toy_fitness_function);

    // Initialize subsets manually
    let subset_indices = vec![vec![0, 1, 2], vec![3, 4]];
    amalgam.set_dependency_subsets(subset_indices).unwrap();

    // Define custom parameters with a population size of 50
    let custom_parameters = AmalgamIdeaParameters::new(
        50, 0.5, 1.1, 0.9, 1e-4, 0.1, 0.2, 0.3, 2.0, 25,
    );
    amalgam.set_parameters(custom_parameters).unwrap();

    // Initialize the optimizer
    amalgam.initialize().unwrap();

    // Store the previous population and fitnesses
    let prev_population = amalgam.get_current_population().clone();
    let prev_fitnesses = amalgam.get_current_fitness_values().clone();

    // Perform population update
    amalgam.update_population().unwrap();

    // Check if the population size is as expected
    let population_size = amalgam.get_parameters().as_ref().unwrap().population_size;
    assert_eq!(
        amalgam.get_current_population().len(),
        population_size,
        "Expected population size of {}, but got {}",
        population_size,
        amalgam.get_current_population().len()
    );

    // Check if there has been any change in the population
    assert_ne!(
        *amalgam.get_current_population(), prev_population,
        "The current population should have changed after the update"
    );

    // Check if the fitness vector length is as expected
    assert_eq!(
        amalgam.get_current_fitnesses().len(),
        population_size,
        "Expected fitness vector length of {}, but got {}",
        population_size,
        amalgam.get_current_fitnesses().len()
    );

    // Check if there has been any change in the fitnesses
    assert_ne!(
        *amalgam.get_current_fitness_values(), prev_fitnesses,
        "The fitness values should have changed after the population update"
    );
}


