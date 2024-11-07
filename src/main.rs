use core::{f64, num};
use std::cmp::min;
use std::collections::HashSet;

use fret_lib::optimization::amalgam_idea::{self, AmalgamIdea};
use fret_lib::optimization::constraints::OptimizationConstraint;
use fret_lib::optimization::optimizer::{FitnessFunction, Optimizer};
use fret_lib::signal_analysis::hmm::hmm_instance::{HMMInstance, HMMInstanceError};
use fret_lib::signal_analysis::hmm::learning::hmm_learner::HMMLearner;
use fret_lib::signal_analysis::hmm::learning::learner_trait::{HMMLearnerTrait, LearnerSpecificInitialValues, LearnerSpecificSetup, LearnerType};
use fret_lib::signal_analysis::hmm::number_states_finder::hmm_num_states_finder::{HMMNumStatesFinder, NumStatesFindStrat};
use fret_lib::signal_analysis::hmm::optimization_tracker::{self, StateCollapseHandle, TerminationCriterium};
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::{baum_welch};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use nalgebra::{DMatrix, DVector};
use rand::{seq, thread_rng, Rng};
use plotters::prelude::*;
use fret_lib::signal_analysis::hmm::initialization::kmeans::*;
use fret_lib::signal_analysis::hmm::initialization::eval_clusters::*;
use fret_lib::signal_analysis::hmm::initialization::hmm_initializer::*;
use fret_lib::signal_analysis::hmm::amalgam_integration::{amalgam_fitness_functions::*, amalgam_modes::*};



// fn trial_fn_baum(sequence: &[f64], num_states: usize) -> (f64, usize, usize) {
//     // Define number of parameters
//     let num_parameters = num_states * 3 + num_states * num_states;

//     // Define num_samples
//     let num_samples = sequence.len();

//     // Define states
//     let state_value_init = StateValueInitMethod::KMeansClustering { max_iters: 100, tolerance: 1e-5, num_tries: 10, eval_method: ClusterEvaluationMethod::Silhouette };
//     let state_values_res = state_value_init.get_state_values(num_states, sequence);
//     if state_values_res.is_err() {return (f64::NEG_INFINITY, num_parameters, num_samples) }
//     let state_values = state_values_res.unwrap();
//     println!("State_values: {:?}", state_values);
    
//     let state_noise_init = StateNoiseInitMethod::Sparse { mult: 0.5 };
//     let state_noises_res = state_noise_init.get_state_noises(num_states, sequence);
//     if state_noises_res.is_err() {return (f64::NEG_INFINITY, num_parameters, num_samples) }
//     let state_noises = state_noises_res.unwrap();
    
//     let trial_states = HMMInstance::generate_state_set(&state_values, &state_noises).unwrap();

//     // Setup the initial start and transition matrices for this setup
//     let initial_start_matrix = StartMatrix::new_balanced(trial_states.len());
//     let initial_transition_matrix = TransitionMatrix::new_balanced(trial_states.len());


//     // Setup the Baum-Welch algorithm to obtain the transition matrices 
//     let mut baum = BaumWelch::new(num_states as u16);

//     // baum.set_initial_states(trial_states).unwrap();
//     let mut rng = thread_rng();
//     set_initial_states_with_jitter_baum(&mut baum, &trial_states, &mut rng, 10).unwrap();
//     baum.set_initial_start_matrix(initial_start_matrix).unwrap();
//     baum.set_initial_transition_matrix(initial_transition_matrix).unwrap();
//     baum.set_state_collapse_handle(StateCollapseHandle::Abort);

//     let termination_criterium = TerminationCriterium::PlateauConvergence {epsilon: 1e-4, plateau_len: 20, max_iterations: Some(500)};

//     let output_res = baum.run_optimization(&sequence, termination_criterium);

//     let fitness: f64;

//     if let Ok(_) = output_res {
//         fitness = baum.take_log_likelihood().unwrap();
//     } else {
//         println!("Result: {:?}",output_res);
//         fitness = f64::NEG_INFINITY;
//     }

//     (fitness, num_parameters, num_samples)

// }




struct FitnessStruct {
    sequence_values: Vec<f64>,
    num_states: usize,
}
impl FitnessStruct {
    pub fn new(num_states: usize, sequence_values: &[f64]) -> Self {
        Self{num_states, sequence_values: sequence_values.to_vec()}
    }
}
impl FitnessFunction<f64, AmalgamHMMFitness> for FitnessStruct{
    fn evaluate(&self,individual: &[f64]) -> AmalgamHMMFitness {
        fitness_fn_direct(individual, self.num_states, &self.sequence_values)
    }
}
// let fitness_closure = |individual: &[f64]| fitness_fn_direct(individual, &sequence_values);


fn main_wait() {
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

    let (_sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);
    
    
    
    // Setup amalgam
    let iter_memory = true;
    let dependence_type = AmalgamDependencies::AllIndependent;
    let fitness_type = AmalgamFitness::Direct;
    let mut amalgam = get_amalgam_object(iter_memory, num_states, &sequence_values, &dependence_type, &fitness_type).unwrap();
    amalgam.set_max_iterations(1000);
    amalgam.run().unwrap();
}


// fn main_num_states() {
//     // Define real system to get simulated time sequence
//     let real_state1 = State::new(0, 10.0, 1.0).unwrap();
//     let real_state2 = State::new(1, 20.0, 2.0).unwrap();
//     let real_state3 = State::new(2, 24.0, 3.0).unwrap();

//     let real_states = vec![real_state1, real_state2, real_state3];
//     let num_states = real_states.len();

//     let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
//     let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

//     let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
//         vec![0.8, 0.1, 0.1],
//         vec![0.1, 0.7, 0.2],
//         vec![0.2, 0.05, 0.75],
//     ];
//     let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

//     let (_sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);

//     // Starting the BIC with BAUM
//     println!("STARTING BIC WITH BAUM");

//     let bic_fn = |num_states: usize| {
//         let sequence = sequence_values.clone();

//         trial_fn_baum(&sequence, num_states)
//     };

//     let min_n = 1;
//     let max_n = 10;
//     let output_bic = bayes_information_criterion_binary_search(bic_fn, min_n, max_n);
//     println!("Output BIC: {:?}", output_bic);

//     println!("STARTING SILHOUETTE ANALYSIS");
//     let min_k = 1;
//     let max_k = 10;
//     let simplified = false;
//     let output_silhouette = silhouette_analysis(&sequence_values, min_k, max_k, simplified);

//     println!("Output silhouette: {:?}", output_bic);

    
// }

fn main() {
    // Define real system to get simulated time sequence
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 23.0, 3.0).unwrap();

    let real_states = vec![real_state1, real_state2, real_state3];
    let _num_states = real_states.len();

    let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
    let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

    let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.3, 0.45, 0.25],
        vec![0.3, 0.2, 0.5],
        vec![0.4, 0.45, 0.15],
    ];
    let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

    let (_sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);

    

    // Setup the learner
    let mut learner = HMMLearner::new(LearnerType::AmalgamIdea);

    let learner_setup = 
    LearnerSpecificSetup::AmalgamIdea { 
        iter_memory: Some(true),
        dependence_type: Some(AmalgamDependencies::AllIndependent),
        fitness_type: Some(AmalgamFitness::Direct), max_iterations: Some(1000) };

    // learner.setup_learner(learner_setup).unwrap();

    // Setup the initializer
    let initializer = HMMInitializer::new(learner.get_setup_data().unwrap());

    // Setup the number of states finder
    let mut num_states_finder = HMMNumStatesFinder::new();
    // num_states_finder.set_strategy(NumStatesFindStrat::KMeansClustering { num_tries: Some(10), max_iters: None, tolerance: None, method: None }).unwrap();
    num_states_finder.set_strategy(NumStatesFindStrat::CurrentSetup { initializer: initializer.clone(), learner: learner.clone() }).unwrap();

    // Get number of states
    let num_states = num_states_finder.get_number_of_states(&sequence_values).unwrap();

    // Get initialization for learner
    let learner_init = initializer.get_intial_values(&sequence_values, num_states).unwrap();

    // Initialize learner
    learner.initialize_learner(num_states, &sequence_values, learner_init).unwrap();

    // Learn this shi
    let (opt_states, opt_start_matrix, opt_transition_matrix) = learner.learn().unwrap();

    println!("States: {:?}", opt_states);
    println!("Start Matrix: {:?}", opt_start_matrix);
    println!("Transition Matrix: {:?}", opt_transition_matrix);
}

