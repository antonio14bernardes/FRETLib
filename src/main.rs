use core::{f64, num};
use std::cmp::min;
use std::collections::HashSet;

use fret_lib::optimization::amalgam_idea::{self, AmalgamIdea};
use fret_lib::optimization::constraints::OptimizationConstraint;
use fret_lib::optimization::optimizer::{FitnessFunction, Optimizer};
use fret_lib::signal_analysis::hmm::hmm_instance::{HMMInstance, HMMInstanceError};
use fret_lib::signal_analysis::hmm::hmm_struct::{HMMInput, NumStatesFindStratWrapper, HMM};
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

    let mut sequence_set = Vec::new();

    for _ in 0..3 {
        let (_sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);
        sequence_set.push(sequence_values);
    }

    

    // Setup HMM

    let mut hmm = HMM::new();

    hmm.add_learner();
    hmm.set_learner_type(LearnerType::AmalgamIdea).unwrap();
    hmm.setup_learner(
        LearnerSpecificSetup::AmalgamIdea { 
            iter_memory: None, dependence_type: None, fitness_type: None, max_iterations: None
        }).unwrap();
    hmm.add_initializer().unwrap();
    hmm.add_number_of_states_finder().unwrap();
    hmm.set_state_number_finder_strategy(NumStatesFindStratWrapper::CurrentSetup).unwrap();

    let input = HMMInput::NumStatesFinder { sequence_set };

    hmm.run(input).unwrap();

    let analyzer = hmm.get_analyzer();
    let opt_states = analyzer.get_states().unwrap();
    let opt_start_matrix = analyzer.get_start_matrix().unwrap();
    let opt_transition_matrix = analyzer.get_transition_matrix().unwrap();
    let state_occupancy = analyzer.get_state_occupancy().unwrap();

    
    println!("States: {:?}", opt_states);
    println!("Start Matrix: {:?}", opt_start_matrix);
    println!("Transition Matrix: {:?}", opt_transition_matrix);
    println!("State Occupancy: {:?}", state_occupancy);
}

