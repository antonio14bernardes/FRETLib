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
use rand_distr::num_traits::Pow;


/* Individual is [means[..], std[..]] */

fn set_initial_states_with_jitter(
    baum: &mut BaumWelch,
    states: &Vec<State>,
    rng: &mut impl Rng,
    max_attempts: usize,
) -> Result<(), BaumWelchError> {
    let mut jittered_states: Vec<State> = states.clone();
    for attempt in 0..max_attempts {
        // Attempt to set initial states
        match baum.set_initial_states(jittered_states.clone()) {
            Ok(()) => return Ok(()), // Success, return early
            Err(BaumWelchError::InvalidInitialStateSet{error: HMMInstanceError::DuplicateStateValues}) => {
                // Apply jitter only in case of DuplicateStateValues error
                jittered_states= states
                    .iter()
                    .map(|state| {
                        let jitter_amount = state.get_value() / 100.0; // Jitter is 1% of the state's value
                        let new_value = state.get_value() + jitter_amount * (rng.gen::<f64>() - 0.5);
                        State::new(state.id, new_value, state.get_noise_std()).unwrap()
                    })
                    .collect();

            }
            Err(e) => return Err(e), // For any other error, return immediately
        }
    }
    Err(BaumWelchError::InvalidInitialStateSet{error: HMMInstanceError::DuplicateStateValues})
}

fn fit_fn(individual: &[f64], sequence_values: &[f64]) -> f64 {

    // println!("New individual: {:?}", individual);
    // Retrieve the states that the optimization algo wants to evaluate

    let num_states = individual.len() / 2;
    let state_means = individual[..num_states].to_vec();
    let state_stds = individual[num_states..].to_vec();

    let trial_states: Vec<State> = state_means.iter()
    .zip(&state_stds)
    .enumerate()
    .map(|(i, (mean, std))| State::new(i, *mean, *std).unwrap())
    .collect();

    // Setup the initial start and transition matrices for this setup
    let initial_start_matrix = StartMatrix::new_balanced(trial_states.len());
    let initial_transition_matrix = TransitionMatrix::new_balanced(trial_states.len());


    // Setup the Baum-Welch algorithm to obtain the transition matrices 
    let mut baum = BaumWelch::new(num_states as u16);

    // baum.set_initial_states(trial_states).unwrap();
    let mut rng = thread_rng();
    set_initial_states_with_jitter(&mut baum, &trial_states, &mut rng, 10).unwrap();
    baum.set_initial_start_matrix(initial_start_matrix).unwrap();
    baum.set_initial_transition_matrix(initial_transition_matrix).unwrap();
    baum.set_state_collapse_handle(StateCollapseHandle::Abort);

    let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 1000 };

    let output_res = baum.run_optimization(&sequence_values, termination_criterium);

    let fitness: f64;

    if let Ok(_) = output_res {
        fitness = baum.take_log_likelihood().unwrap();
    } else {
        fitness = f64::NEG_INFINITY;
    }

    fitness
    
}

fn ez_fit(individual: &[f64]) -> f64 {
    let squared: Vec<f64> = individual.iter().map(|v| v*v).collect();
    -squared.iter().sum::<f64>()
}

fn main() {
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 30.0, 3.0).unwrap();

    let real_states = [real_state1, real_state2, real_state3].to_vec();

    let num_states = real_states.len();

    let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
    let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
    let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.8, 0.1, 0.1],
        vec![0.1, 0.7, 0.2],
        vec![0.2, 0.05, 0.75],
    ];
    let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

    let (_sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 500);

    let max_value = sequence_values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_value = sequence_values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max_value - min_value;
    let max_noise = range / (num_states as f64 * 2.0);
    let min_noise = max_noise / 1e2;
    println!("Max: {}, Min: {}", &max_value, &min_value);

    let problem_size = num_states * 2;
    let iter_memory = true;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    let subsets: Vec<Vec<usize>>= vec![vec![0,3], vec![1,4], vec![2,5]];
    amalgam.set_dependency_subsets(subsets).unwrap();

    let constraints = OptimizationConstraint::MaxMinValue { max: vec![max_value,max_noise], min: vec![min_value, min_noise] };
    let constraints = vec![constraints; num_states];
    amalgam.set_constraints(constraints).unwrap();


    let fitness_closure = |individual: &[f64]| fit_fn(individual, &sequence_values);

    amalgam.set_fitness_function(fitness_closure);

    amalgam.run(10000).unwrap();

    let (best_solution, best_fitness) = amalgam.get_best_solution().unwrap();
    

}

fn main_setup1() {
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 30.0, 3.0).unwrap();

    let real_states = [real_state1, real_state2, real_state3].to_vec();

    let num_states = real_states.len();

    let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
    let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
    let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.8, 0.1, 0.1],
        vec![0.1, 0.7, 0.2],
        vec![0.2, 0.05, 0.75],
    ];
    let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

    let (_sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 500);

    let max_value = sequence_values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_value = sequence_values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let range = max_value - min_value;
    let max_noise = range / (num_states as f64);
    let min_noise = max_noise / 1e2;
    println!("Max: {}, Min: {}", &max_value, &min_value);

    let problem_size = num_states * 2;
    let iter_memory = true;
    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);

    let subsets: Vec<Vec<usize>>= vec![vec![0,1,2], vec![3,4,5]];
    amalgam.set_dependency_subsets(subsets).unwrap();

    let means_constraints = OptimizationConstraint::MaxMinValue { max: vec![max_value], min: vec![min_value] };
    let stds_constraints = OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] };
    let constraints = vec![means_constraints, stds_constraints];
    amalgam.set_constraints(constraints).unwrap();


    let fitness_closure = |individual: &[f64]| fit_fn(individual, &sequence_values);

    amalgam.set_fitness_function(fitness_closure);

    amalgam.run(10000).unwrap();

    let (best_solution, best_fitness) = amalgam.get_best_solution().unwrap();
    

}

fn main_old() {

    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 30.0, 3.0).unwrap();

    let real_states = [real_state1, real_state2, real_state3].to_vec();

    let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
    let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
    let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.8, 0.1, 0.1],
        vec![0.1, 0.7, 0.2],
        vec![0.2, 0.05, 0.75],
    ];
    let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

    let (sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 500);

    let max_value = sequence_values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_value = sequence_values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let noise_std = (max_value - min_value) / 20.0;

    let fake_states = HMMInstance::generate_sparse_states(real_states.len() as u16, max_value, min_value, noise_std).unwrap();
    

    let fake_start_matrix = StartMatrix::new_balanced(real_states.len());
    let fake_transition_matrix = TransitionMatrix::new_balanced(real_states.len());

    let mut baum = BaumWelch::new(3);

    baum.set_initial_states(fake_states).unwrap();
    baum.set_initial_start_matrix(fake_start_matrix).unwrap();
    baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();
    baum.set_state_collapse_handle(StateCollapseHandle::RestartRandom { allowed_trials: 10 });

    let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 1000 };

    let output_res = baum.run_optimization(&sequence_values, termination_criterium);

    if output_res.is_err() {
        println!("Optimization failed: {:?}", output_res);
        return
    }


    let opt_states = baum.take_states().unwrap();
    let opt_start_matrix = baum.take_start_matrix().unwrap();
    let opt_transition_matrix = baum.take_transition_matrix().unwrap();
    let log_likelihood = baum.take_log_likelihood().unwrap();

    println!("New states: {:?}", &opt_states);
    println!("New start matrix: {:?}", &opt_start_matrix);
    println!("New transition matrix: {:?}", &opt_transition_matrix);
    println!("Final observations prob: {}", &log_likelihood);

}

// fn main() {
//         let real_state1 = State::new(0, 10.0, 1.0).unwrap();
//         let real_state2 = State::new(1, 20.0, 2.0).unwrap();
//         let real_state3 = State::new(2, 30.0, 3.0).unwrap();

//         let real_states = [real_state1, real_state2, real_state3].to_vec();

//         let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
//         let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
//         let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
//             vec![0.8, 0.1, 0.1],
//             vec![0.1, 0.7, 0.2],
//             vec![0.2, 0.05, 0.75],
//         ];
//         let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

//         let (sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 900);


//         /****** Create slightly off states and matrices ******/


//         let fake_state1 = State::new(0, 14.8, 1.2).unwrap();
//         let fake_state2 = State::new(1, 15.5, 1.2).unwrap();
//         let fake_state3 = State::new(2, 25.5, 2.7).unwrap();

//         let fake_states = [fake_state1, fake_state2, fake_state3].to_vec();

//         let fake_start_matrix_raw: Vec<f64> = vec![0.4, 0.3, 0.3];
//         let fake_start_matrix = StartMatrix::new(fake_start_matrix_raw);

        
//         let fake_transition_matrix_raw: Vec<Vec<f64>> = vec![
//             vec![0.3, 0.3, 0.4],
//             vec![0.4, 0.5, 0.1],
//             vec![0.6, 0.1, 0.3],
//         ];
//         let fake_transition_matrix = TransitionMatrix::new(fake_transition_matrix_raw);

//         let mut baum = BaumWelch::new(3);

//         baum.set_initial_states(fake_states).unwrap();
//         baum.set_initial_start_matrix(fake_start_matrix).unwrap();
//         baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();

//         let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 1000 };

//         let output_res = baum.run_optimization(&sequence_values, termination_criterium);

//         if output_res.is_err() {
//             println!("Optimization failed: {:?}", output_res);
//             return
//         }



//         // For plotting
//         let opt_states = baum.take_states().unwrap();
//         let opt_start_matrix = baum.take_start_matrix().unwrap();
//         let opt_transition_matrix = baum.take_transition_matrix().unwrap();
//         let obs_prob = baum.take_observations_prob().unwrap();

//         println!("New states: {:?}", &opt_states);
//         println!("New start matrix: {:?}", &opt_start_matrix);
//         println!("New transition matrix: {:?}", &opt_transition_matrix);
//         println!("Final observations prob: {}", &obs_prob);

//         let mut viterbi = Viterbi::new(&opt_states, &opt_start_matrix, &opt_transition_matrix);
//         viterbi.run(&sequence_values, true);

//         let predictions = viterbi.get_prediction().unwrap();

//         let sum: u16 = predictions.iter().zip(sequence_ids).map(|(a,b)| if *a == b {1_u16} else {0_u16}).sum();
//         let accuracy = (sum as f64) / (predictions.len() as f64);

//         println!("Prediction accuracy {}", accuracy);

//         let pred_ideal_sequence: Vec<f64> = predictions.iter().map(|id| opt_states[*id].get_value()).collect();

//         plot_sequences(&sequence_values, &pred_ideal_sequence).expect("Error while plotting sequences");

// }

    

