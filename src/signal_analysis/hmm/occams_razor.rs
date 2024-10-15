use super::{baum_welch::BaumWelch, hmm_instance::HMMInstance, optimization_tracker::{StateCollapseHandle, TerminationCriterium}, StartMatrix, State, TransitionMatrix, HMM};

pub fn bayes_information_criterion_binary_search<F>
(test_function: F,  min_n: usize, max_n: usize) -> (usize, f64)
where
    F: Fn(usize) -> (f64, usize, usize), // log-likelihood, total num parameters, num samples
{
    let mut left = min_n;
    let mut right = max_n;
    let mut best_n = 1;
    let mut best_bic = f64::MAX;

    while left <= right {
        let mid = (left + right) / 2;

        // Evaluate BIC at mid and mid+1
        let (ll_mid, num_parameters_mid, samples_mid) = test_function(mid);
        let (ll_next, num_parameters_next, samples_next) = test_function(mid + 1);
        let bic_mid = compute_bic(ll_mid, num_parameters_mid, samples_mid);
        let bic_next = compute_bic(ll_next, num_parameters_next, samples_next);

        if bic_mid < best_bic {
            best_bic = bic_mid;
            best_n = mid;
        }

        // Find direction in which bic is improving
        if bic_mid > bic_next {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    (best_n, best_bic)
}

fn compute_bic(log_likelihood: f64, k: usize, n_samples: usize) -> f64 {
    -2.0 * log_likelihood + (k as f64) * (n_samples as f64).ln()
}



pub fn trial_function(n: usize) -> (f64, usize, usize) {


    // Compute num of parameters
    let state_params = 2 * n;
    let start_mat_params = n;
    let trans_mat_params = n * n;
    let num_params = state_params + start_mat_params + trans_mat_params;





    // Keep going
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