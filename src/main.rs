use std::cmp::min;
use std::collections::HashSet;

use fret_lib::optimization::amalgam_idea::AmalgamIdea;
use fret_lib::optimization::optimizer::Optimizer;
use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::optimization_tracker::{StateCollapseHandle, TerminationCriterium};
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::{baum_welch, HMM};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use nalgebra::{DMatrix, DVector};
use rand::seq;
use plotters::prelude::*;
use fret_lib::signal_analysis::hmm::occams_razor::*;
use fret_lib::signal_analysis::hmm::kmeans::*;



fn main() {
        let real_state1 = State::new(0, 10.0, 1.0).unwrap();
        let real_state2 = State::new(1, 20.0, 2.0).unwrap();
        let real_state3 = State::new(2, 23.0, 3.0).unwrap();

        let real_states = [real_state1, real_state2, real_state3].to_vec();

        let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
        let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
        let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.2, 0.05, 0.75],
        ];
        let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

        let (sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 900);

        

        /****** Create slightly off states and matrices ******/
        let max_value = sequence_values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_value = sequence_values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let noise_std = (max_value - min_value) / 20.0;

        let n: usize = 3;

        let (cluster_means, _) = k_means_1D(&sequence_values, n, 1000, 1e-3);
        println!("Cluster means: {:?}", &cluster_means);
        let fake_states = HMMInstance::generate_state_set(&cluster_means, &vec![noise_std; n]).unwrap();
        // let fake_states = HMMInstance::generate_sparse_states(n as u16, max_value, min_value, noise_std).unwrap();

        let fake_start_matrix = StartMatrix::new_balanced(n);
        let fake_transition_matrix = TransitionMatrix::new_balanced(n);

        let mut baum = BaumWelch::new(n as u16);

        baum.set_initial_states(fake_states).unwrap();
        baum.set_initial_start_matrix(fake_start_matrix).unwrap();
        baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();
        baum.set_state_collapse_handle(StateCollapseHandle::RemoveCollapsedState);


        let termination_criterium = TerminationCriterium::PlateauConvergence {epsilon: 1e-4, plateau_len: 20, max_iterations: Some(500)};

        let output_res = baum.run_optimization(&sequence_values, termination_criterium);

        if output_res.is_err() {
            println!("Optimization failed: {:?}", output_res);
            return
        }



        // For plotting
        let opt_states = baum.take_states().unwrap();
        let opt_start_matrix = baum.take_start_matrix().unwrap();
        let opt_transition_matrix = baum.take_transition_matrix().unwrap();
        let obs_prob = baum.take_log_likelihood().unwrap();

        println!("New states: {:?}", &opt_states);
        println!("New start matrix: {:?}", &opt_start_matrix);
        println!("New transition matrix: {:?}", &opt_transition_matrix);
        println!("Final observations prob: {}", &obs_prob);



        for state in opt_states {
            println!("{}", state.get_value());
        }

        

}



fn main_BIC() {
    let out = bayes_information_criterion_binary_search(trial_function, 1, 5);

    println!("Result: {:?}", out);
}


pub fn trial_function(n: usize) -> (f64, usize, usize) {


    // Compute num of parameters
    let state_params = 2 * n;
    let start_mat_params = n;
    let trans_mat_params = n * n;
    let num_params = state_params + start_mat_params + trans_mat_params;





    // Keep going
    let real_state1 = State::new(0, 10.0, 1.0).unwrap();
    let real_state2 = State::new(1, 27.0, 2.0).unwrap();
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
    let noise_std = (max_value - min_value) / 30.0;

    let fake_states = HMMInstance::generate_sparse_states(n as u16, max_value, min_value, noise_std).unwrap();

    let fake_start_matrix = StartMatrix::new_balanced(n);
    let fake_transition_matrix = TransitionMatrix::new_balanced(n);

    let mut baum = BaumWelch::new(n as u16);

    baum.set_initial_states(fake_states).unwrap();
    baum.set_initial_start_matrix(fake_start_matrix).unwrap();
    baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();
    baum.set_state_collapse_handle(StateCollapseHandle::RestartRandom { allowed_trials: 10 });

    let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 1000 };

    let output_res = baum.run_optimization(&sequence_values, termination_criterium);

    if output_res.is_err() {
        return (f64::MAX, num_params, sequence_values.len())
    }


    let opt_states = baum.take_states().unwrap();
    let opt_start_matrix = baum.take_start_matrix().unwrap();
    let opt_transition_matrix = baum.take_transition_matrix().unwrap();
    let log_likelihood = baum.take_log_likelihood().unwrap();

    println!("New states: {:?}", &opt_states);
    println!("New start matrix: {:?}", &opt_start_matrix);
    println!("New transition matrix: {:?}", &opt_transition_matrix);
    println!("Final observations prob: {}", &log_likelihood);

    (log_likelihood, num_params, sequence_values.len())

}

    

}