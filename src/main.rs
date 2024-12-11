use core::{f64, num};
use std::cmp::min;
use std::collections::HashSet;
use std::iter::Filter;
use std::thread;

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
use fret_lib::signal_analysis::hmm::{baum_welch, StateValueInitMethod};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use fret_lib::trace_selection::filter::FilterSetup;
use fret_lib::trace_selection::set_of_points::SetOfPoints;
use nalgebra::{DMatrix, DVector};
use rand::{seq, thread_rng, Rng};
use plotters::prelude::*;
use rfd::FileDialog;



use fret_lib::signal_analysis::hmm::initialization::hmm_initializer::*;



use fret_lib::interface::app::*;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "FRET Analysis GUI",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )  
}

// Example of not using the GUI
fn _main_hmm() {
    // Define real system to get simulated time sequence
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 20.0, 2.0).unwrap();
    let real_state3 = State::new(2, 30.0, 3.0).unwrap();

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


    let mut sequence_set: Vec<Vec<f64>> = Vec::new();
    let mut sequence_ids_set: Vec<Vec<usize>> = Vec::new();
    let num_sequences = 10;
    for _ in 0..num_sequences {
        let (sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 100);
        sequence_set.push(sequence_values);
        sequence_ids_set.push(sequence_ids);
    }

    // Setup HMM

    let mut hmm = HMM::new();

    hmm.add_learner();
    // hmm.set_learner_type(LearnerType::AmalgamIdea).unwrap();
    // hmm.setup_learner(
    //     LearnerSpecificSetup::AmalgamIdea { 
    //         iter_memory: None, dependence_type: None, fitness_type: None, max_iterations: None
    //     }).unwrap();

    hmm.set_learner_type(LearnerType::BaumWelch).unwrap();
    hmm.setup_learner(LearnerSpecificSetup::BaumWelch { termination_criterion: None }).unwrap();

    hmm.add_initializer().unwrap();
    let mut init_method = InitializationMethods::new();
    init_method.state_values_method = Some(StateValueInitMethod::StateHints { state_values: vec![10.0, 20.0, 30.0] });
    hmm.set_initialization_method(init_method).unwrap();

    let input = HMMInput::Initializer { num_states: real_states.len(), sequence_set: sequence_set };






    // hmm.run(input).unwrap();


    // Launch a thread to run the HMM 
    let handle = thread::spawn(move || {
        match hmm.run(input) {
            Ok(()) => Ok(hmm),
            Err(e) => Err(e),
        }
        
    });
    let updated_hmm_result = handle.join().expect("Thread panicked");
    let updated_hmm = updated_hmm_result.unwrap();

    let analyzer = updated_hmm.get_analyzer();
    let opt_states = analyzer.get_states().unwrap();
    let opt_start_matrix = analyzer.get_start_matrix().unwrap();
    let opt_transition_matrix = analyzer.get_transition_matrix().unwrap();
    let state_occupancy = analyzer.get_state_occupancy().unwrap();


    
    println!("States: {:?}", opt_states);
    println!("Start Matrix: {:?}", opt_start_matrix);
    println!("Transition Matrix: {:?}", opt_transition_matrix);
    println!("State Occupancy: {:?}", state_occupancy);
}