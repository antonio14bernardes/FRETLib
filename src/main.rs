use core::{f64, num};
use std::cmp::min;
use std::collections::HashSet;
use std::iter::Filter;

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
use fret_lib::trace_selection::filter::FilterSetup;
use fret_lib::trace_selection::set_of_points::SetOfPoints;
use nalgebra::{DMatrix, DVector};
use rand::{seq, thread_rng, Rng};
use plotters::prelude::*;
use fret_lib::signal_analysis::hmm::initialization::kmeans::*;
use fret_lib::signal_analysis::hmm::initialization::eval_clusters::*;
use fret_lib::signal_analysis::hmm::initialization::hmm_initializer::*;
use fret_lib::signal_analysis::hmm::amalgam_integration::{amalgam_fitness_functions::*, amalgam_modes::*};


use fret_lib::trace_selection::individual_trace::*;
use fret_lib::trace_selection::point_traces::{self, PointTraces};
use fret_lib::trace_selection::trace_loader::*;
use fret_lib::trace_selection::individual_trace::*;


fn main_hmm() {
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


    let mut sequence_set: Vec<Vec<f64>> = Vec::new();
    let mut sequence_ids_set: Vec<Vec<usize>> = Vec::new();
    let num_sequences = 10;
    for _ in 0..num_sequences {
        let (sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);
        sequence_set.push(sequence_values);
        sequence_ids_set.push(sequence_ids);
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

    let input = HMMInput::Initializer { num_states: real_states.len(), sequence_set: sequence_set };

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


fn main_single_file() {
    let normal_path: &str = "/Users/antonio14bernardes/Documents/Internship/traces_data/Traces_T23_01_sif_pair5.txt";
    let d_bleach_path = "/Users/antonio14bernardes/Documents/Internship/traces_data/D_bleach_T23_01_sif_pair5.txt";
    let a_bleach_path = "/Users/antonio14bernardes/Documents/Internship/traces_data/a_bleach_T23_01_sif_pair7.txt";

    let file_path = d_bleach_path;

    let mut point_traces = parse_file(file_path).unwrap();

    // let trace = point_traces.take_trace(&TraceType::AemDexc).unwrap();
    println!("Trace types: {:?}", point_traces.get_types());

    let pb_filter = PhotobleachingFilterValues::default();
    let fl_filter = FretLifetimesFilterValues::default();

    let values_to_filter = point_traces.prepare_filter_values(&pb_filter, &fl_filter);

    println!("Output: {:?}", values_to_filter);


    let filter = FilterSetup::default();
    let output = filter.check_valid(&values_to_filter.unwrap());


    println!("Filter ouptut: {:?}", output);    
}

fn main() {
    let traces_dir = "/Users/antonio14bernardes/Documents/Internship/traces_data";

    let mut set_of_points = SetOfPoints::new();

    set_of_points.add_points_from_dir(traces_dir).unwrap();

    set_of_points.clear_filter();

    set_of_points.filter();

    // println!("Output: {:?}", set_of_points);


    let vecs: Vec<&PointTraces> = set_of_points.get_points().values().collect();
    let base_file_path = "fret_plot".to_string();
    let mut values_set = Vec::new();

    for (i, vec) in vecs.iter().enumerate() {
        let values = vec.get_valid_fret().unwrap();
        values_set.push(values.to_vec());
    
        // Create a unique file path by appending the counter value
        let file_path = format!("{}_{}.png", base_file_path, i);
    
        plot_vec(values.to_vec(), &file_path).unwrap();
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

    let input = HMMInput::NumStatesFinder { sequence_set: values_set };

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

use plotters::prelude::*;
use std::error::Error;

fn plot_vec(data: Vec<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
    // Define the size of the output image in pixels
    let root_area = BitMapBackend::new(file_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Set up the chart with a title, labels, and margin settings
    let mut chart = ChartBuilder::on(&root_area)
        .caption("Vector Plot", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len(), *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()..*data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())?;

    chart.configure_mesh().draw()?;

    // Plot the data as a line
    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?;

    Ok(())
}