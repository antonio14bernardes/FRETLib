use core::num;
use std::cmp::min;
use std::collections::HashSet;

use fret_lib::optimization::amalgam_idea::{self, AmalgamIdea};
use fret_lib::optimization::constraints::OptimizationConstraint;
use fret_lib::optimization::optimizer::Optimizer;
use fret_lib::signal_analysis::hmm::hmm_instance::{HMMInstance, HMMInstanceError};
use fret_lib::signal_analysis::hmm::optimization_tracker::{StateCollapseHandle, TerminationCriterium};
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::{baum_welch, HMM};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use nalgebra::{DMatrix, DVector};
use rand::{seq, thread_rng, Rng};
use plotters::prelude::*;
use fret_lib::signal_analysis::hmm::occams_razor::*;
use fret_lib::signal_analysis::hmm::kmeans::*;
use fret_lib::signal_analysis::hmm::amalgam_tools::*;


fn main() {
    // Define real system to get simulated time sequence
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 22.0, 3.0).unwrap();

    let real_states = vec![real_state1, real_state2, real_state3];
    let num_states = real_states.len();

    let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
    let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

    let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.8, 0.1, 0.1],
        vec![0.1, 0.7, 0.2],
        vec![0.2, 0.05, 0.75],
    ];
    let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

    let (_sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);

    // Analyze received sequence for preprocessing and set constraints
    let max_value = sequence_values.iter().cloned().fold(f64::NAN, f64::max);
    let min_value = sequence_values.iter().cloned().fold(f64::NAN, f64::min);
    let range = max_value - min_value;
    let max_noise = range / (num_states as f64 * 2.0);
    let min_noise = max_noise / 1e2;

    println!("Max: {}, Min: {}", max_value, min_value);

    // Get optimizer object
    let problem_size = num_states * 3 + num_states * num_states; // Each state has value and noise as separate variables
    let iter_memory = true;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    // Set fitness function
    let fitness_closure = |individual: &[f64]| fitness_fn_direct(individual, &sequence_values);
    amalgam.set_fitness_function(fitness_closure);

    // Set population size
    amalgam.set_population_size(100).unwrap();

    // Set max iterations
    amalgam.set_max_iterations(1000);

    // Setup dependencies in variables: Make each element its own subset
    let subsets: Vec<Vec<usize>> = vec![
        vec![0], vec![1], vec![2], // Value subsets
        vec![3], vec![4], vec![5], // Noise subsets
        vec![6, 7, 8], // Start matrix subset
        vec![9, 10, 11], // First row Transition matrix
        vec![12, 13, 14], // Second row Transition matrix
        vec![15, 16, 17], // Third row Transition matrix
    ];
    amalgam.set_dependency_subsets(subsets).unwrap();

    // Define constraints for each variable independently
    let constraints = vec![
        OptimizationConstraint::MaxMinValue { max: vec![max_value], min: vec![min_value] },
        OptimizationConstraint::MaxMinValue { max: vec![max_value], min: vec![min_value] },
        OptimizationConstraint::MaxMinValue { max: vec![max_value], min: vec![min_value] },
        OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] },
        OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] },
        OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] },
        OptimizationConstraint::PositiveSumTo { sum: 1.0 },
        OptimizationConstraint::PositiveSumTo { sum: 1.0 },
        OptimizationConstraint::PositiveSumTo { sum: 1.0 },
        OptimizationConstraint::PositiveSumTo { sum: 1.0 },
    ];
    amalgam.set_constraints(constraints).unwrap();





    // Obtain preprocessing clusters, their means, and std deviations
    let num_clusters = num_states;
    let (cluster_means, cluster_assignments) = k_means_1D(&sequence_values, num_clusters, 100, 1e-3);
    println!("Cluster means: {:?}", cluster_means);

    let mut cluster_stds = vec![0.0; num_clusters];
    let mut count_per_cluster = vec![0; num_states];
    for assignment in cluster_assignments.iter() {
        count_per_cluster[*assignment] += 1;
    }
    for (value, &assignment) in sequence_values.iter().zip(&cluster_assignments) {
        cluster_stds[assignment] += (value - cluster_means[assignment]).powi(2);
    }
    for i in 0..cluster_stds.len() {
        cluster_stds[i] = (cluster_stds[i] / count_per_cluster[i] as f64).sqrt();
    }
    println!("Cluster STDs: {:?}", cluster_stds);

    let mean_of_cluster_stds = cluster_stds.iter().sum::<f64>() / cluster_stds.len() as f64;
    let std_of_cluster_stds = cluster_stds.iter().map(|&std| (std - mean_of_cluster_stds).powi(2)).sum::<f64>().sqrt() / num_clusters as f64;



    // Define initial values and std distributions for the states as independent variables
    let initial_value_means = cluster_means.clone();
    let initial_value_stds = cluster_stds.clone();
    let initial_noise_means = vec![(max_noise + min_noise) / 2.0; num_states];
    let initial_noise_stds = vec![std_of_cluster_stds; num_states];
  

    // Define vectors of DVector and DMatrix for initial values and noise for the states
    let initial_values_means_vec: Vec<DVector<f64>> = initial_value_means
    .iter()
    .map(|&mean| DVector::from_vec(vec![mean]))
    .collect();

    let initial_values_stds_vec: Vec<DMatrix<f64>> = initial_value_stds
    .iter()
    .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
    .collect();

    let initial_noise_means_vec: Vec<DVector<f64>> = initial_noise_means
    .iter()
    .map(|&mean| DVector::from_vec(vec![mean]))
    .collect();

    let initial_noise_stds_vec: Vec<DMatrix<f64>> = initial_noise_stds
    .iter()
    .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
    .collect();

    // Define initial probabilities for start matrix and transition matrix
    let initial_prob_mean: f64 = 1.0 / 3.0;
    let initial_prob_std: f64 = 0.25;

    // Define the DVector and DMatrix of each probability row

    let initial_start_probs_means_vec: Vec<DVector<f64>> = vec![DVector::from_vec(vec![initial_prob_mean; num_states])];

    let initial_start_probs_cov_vec: Vec<DMatrix<f64>> = vec![DMatrix::from_diagonal(&DVector::from_vec(vec![initial_prob_std.powi(2); num_states]))];

    let initial_transition_probs_means_vec: Vec<DVector<f64>> = (0..num_states)
        .map(|_| DVector::from_vec(vec![initial_prob_mean; num_states]))
        .collect();

    let initial_transition_probs_cov_vec: Vec<DMatrix<f64>> = (0..num_states)
        .map(|_| DMatrix::from_diagonal(&DVector::from_element(num_states, initial_prob_std.powi(2))))
        .collect();

    // Combine all the means into a single Vec<DVector<f64>>
    let initial_means: Vec<DVector<f64>> = [
        initial_values_means_vec,
        initial_noise_means_vec,
        initial_start_probs_means_vec,
        initial_transition_probs_means_vec,
    ]
    .concat();

    // Combine all the covariances into a single Vec<DMatrix<f64>>
    let initial_covs: Vec<DMatrix<f64>> = [
        initial_values_stds_vec,
        initial_noise_stds_vec,
        initial_start_probs_cov_vec,
        initial_transition_probs_cov_vec,
    ]
    .concat();
    
    // Set the initial distribution for amalgam
    amalgam.set_initial_distribution(&initial_means, &initial_covs).unwrap();

    // Run optimization
    amalgam.run().unwrap();

    let (best_solution, best_fitness) = amalgam.get_best_solution().unwrap();
    println!("Best Solution: {:?}, Best Fitness: {:?}", best_solution, best_fitness);
}

