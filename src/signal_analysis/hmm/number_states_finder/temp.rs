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
