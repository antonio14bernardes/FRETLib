use crate::optimization::optimizer::{FitnessFunction, OptimizationFitness};
use crate::signal_analysis::hmm::baum_welch::add_jitter_to_state_values_vec;

use super::super::optimization_tracker::{TerminationCriterium, StateCollapseHandle};
use super::super::state::*;
use super::super::hmm_instance::*;
use super::super::hmm_matrices::*;
use super::super::baum_welch::*;
use super::amalgam_modes::AmalgamFitness;

use rand::thread_rng;

#[derive(Clone, Debug)]
pub struct AmalgamHMMFitness {
    fitness: f64,
    states: Option<Vec<State>>,
    start_matrix: Option<StartMatrix>,
    transition_matrix: Option<TransitionMatrix>,
}

impl OptimizationFitness for AmalgamHMMFitness {
    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}

impl AmalgamHMMFitness {
    pub fn get_states(&self) -> Option<&Vec<State>> {
        self.states.as_ref()
    }
    pub fn get_start_matrix(&self) -> Option<&StartMatrix> {
        self.start_matrix.as_ref()
    }
    pub fn get_transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix.as_ref()
    }
}

#[derive(Clone)]
pub struct AmalgamHMMFitnessFunction {
    num_states: usize,
    sequence_values: Vec<Vec<f64>>,
    fitness_type: AmalgamFitness,
}

impl AmalgamHMMFitnessFunction {
    pub fn new(num_states: usize, sequence_values: &Vec<Vec<f64>>, fitness_type: &AmalgamFitness) -> Self {
        Self{
            num_states,
            sequence_values: sequence_values.clone(),
            fitness_type: fitness_type.clone()}
    }

    pub fn get_fitness_type(&self) -> &AmalgamFitness {
        &self.fitness_type
    }
}

impl FitnessFunction<f64, AmalgamHMMFitness> for AmalgamHMMFitnessFunction {
    fn evaluate(&self,individual: &[f64]) -> AmalgamHMMFitness {
        match self.fitness_type {
            AmalgamFitness::Direct => fitness_fn_direct(individual, self.num_states, &self.sequence_values),
            AmalgamFitness::BaumWelch => fitness_fn_baum_welch(individual, self.num_states, &self.sequence_values),
        }
        
    }
}




pub fn fitness_fn_baum_welch(individual: &[f64], num_states: usize, sequence_set: &Vec<Vec<f64>>) -> AmalgamHMMFitness {

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

    // Run for each sequence in the set

    
    let total_sequence_len: usize = sequence_set.iter().map(|sequence| sequence.len()).sum();
    let num_states = trial_states.len();
    let mut avg_log_likelihood: f64 = 0.0;
    let mut avg_state_values: Vec<f64> = vec![0.0; num_states];
    let mut avg_state_noise: Vec<f64> = vec![0.0; num_states];
    let mut avg_start_matrix: Vec<f64> = vec![0.0; num_states];
    let mut avg_transition_matrix: Vec<Vec<f64>> = vec![vec![0.0; num_states]; num_states];

    for sequence_values in sequence_set {
        // Setup the Baum-Welch algorithm
        let mut baum = BaumWelch::new(num_states as u16);

        let mut rng = thread_rng();
        set_initial_states_with_jitter_baum(&mut baum, &trial_states, &mut rng, 1000).unwrap();
        baum.set_initial_start_matrix(initial_start_matrix.clone()).unwrap();
        baum.set_initial_transition_matrix(initial_transition_matrix.clone()).unwrap();
        baum.set_state_collapse_handle(StateCollapseHandle::Abort);

        let termination_criterium = TerminationCriterium::PlateauConvergence {epsilon: 1e-4, plateau_len: 20, max_iterations: Some(500)};

        let output_res = baum.run_optimization(&sequence_values, termination_criterium);

        let log_likelihood: f64;
        let states: Option<Vec<State>>;
        let start_matrix: Option<StartMatrix>;
        let transition_matrix: Option<TransitionMatrix>;


        if let Ok(_) = output_res {
            log_likelihood = baum.take_log_likelihood().unwrap();
            states = Some(baum.take_states().unwrap());
            start_matrix = Some(baum.take_start_matrix().unwrap());
            transition_matrix = Some(baum.take_transition_matrix().unwrap());
        } else {
            return AmalgamHMMFitness {
                fitness: f64::NEG_INFINITY,
                states: None,
                start_matrix: None,
                transition_matrix: None,
            }
        }

        // Update weighted averages based on the sequence length
        let current_sequence_len = sequence_values.len();
        let coefficient = current_sequence_len as f64 / total_sequence_len as f64;
        avg_log_likelihood += coefficient * log_likelihood;

        for state in states.unwrap().iter() {
            let id = state.get_id();
            avg_state_values[id] += coefficient * state.get_value();
            avg_state_noise[id] += coefficient * state.get_noise_std();
        }

        for (idx, value) in start_matrix.unwrap().matrix.iter().enumerate() {
            avg_start_matrix[idx] += coefficient * value;
        }

        for (idx_row, row) in transition_matrix.unwrap().matrix.raw_matrix.iter().enumerate() {
            for (idx_column, value) in row.iter().enumerate() {
                avg_transition_matrix[idx_row][idx_column] += coefficient * value;
            }
        }

    }

    // Normalize start matrix
    let sum: f64 = avg_start_matrix.iter().sum();
    avg_start_matrix.iter_mut().for_each(|value| *value = *value / sum);

    // Normalize transition matrix
    for row in avg_transition_matrix.iter_mut() {
        let sum: f64 = row.iter().sum();
        row.iter_mut().for_each(|value| *value = *value / sum);
    }


    // Pack stuff
    let mut new_state_res = HMMInstance::generate_state_set(&avg_state_values, &avg_state_noise);
    let new_states: Vec<State>;
    let mut rng = thread_rng();
    loop {
        if let Ok(new_state_value) = new_state_res.clone() {
            new_states = new_state_value;
            break;
        } else {
            let new_state_values = add_jitter_to_state_values_vec(&avg_state_values, &mut rng);
            new_state_res = HMMInstance::generate_state_set(&new_state_values, &avg_state_noise);
        }
    }
    let new_start_matrix = StartMatrix::new(avg_start_matrix);
    let new_transition_matrix = TransitionMatrix::new(avg_transition_matrix);


    return AmalgamHMMFitness {
        fitness: avg_log_likelihood,
        states: Some(new_states),
        start_matrix: Some(new_start_matrix),
        transition_matrix: Some(new_transition_matrix),
    }

    // AmalgamHMMFitness{fitness, states, start_matrix, transition_matrix} this will be updated
    
}



pub fn fitness_fn_direct(individual: &[f64], num_states: usize, sequence_set: &Vec<Vec<f64>>) -> AmalgamHMMFitness {
    // Split the individual into the respective parts
    let (means, rest) = individual.split_at(num_states);
    let (stds, rest) = rest.split_at(num_states);
    let (start_probs, flattened_transition) = rest.split_at(num_states);

    // Convert means and stds into Vec<State>
    let mut states: Vec<State> = means.iter()
        .zip(stds.iter())
        .enumerate()
        .map(|(i, (&mean, &std))| State::new(i, mean, std).unwrap())
        .collect();

    // Create the StartMatrix using the start_probs extracted
    let start_matrix = StartMatrix::new(start_probs.to_vec());

    // Convert flattened_transition into a TransitionMatrix
    let transition_matrix_vec: Vec<Vec<f64>> = flattened_transition
        .chunks(num_states)
        .map(|chunk| chunk.to_vec())
        .collect();
    let transition_matrix = TransitionMatrix::new(transition_matrix_vec);

    let mut rng = thread_rng();

    // For each sequence in the population of sequences
    let mut average_log_likelihood: f64 = 0.0;
    let total_sequence_len: usize = sequence_set.iter().map(|sequence| sequence.len()).sum();

    for sequence_values in sequence_set {
        // Start the hmm instance and try to run viterbi
        let max_attempts = 100_usize;
        let mut hmm_instance = HMMInstance::new(&states, &start_matrix, &transition_matrix);
        for _ in 0..max_attempts {
            
            match hmm_instance.run_viterbi(&sequence_values) {
                Ok(()) => break, // Success, return early
                Err(HMMInstanceError::DuplicateStateValues) => {
                    // Apply jitter only in case of DuplicateStateValues error
                    let jittered_states = add_jitter_to_states(&states, &mut rng);
                    states = jittered_states;
                    hmm_instance = HMMInstance::new(&states, &start_matrix, &transition_matrix);

                }
                Err(_) => {panic!()}, // For any other error, return immediately
            }
        }

        // Run forward backward algorithm
        hmm_instance.run_alphas_scaled(sequence_values).unwrap();

        let log_likelihood_option = hmm_instance.take_log_likelihood();    
        let log_likelihood: f64;

        if let Some(out) = log_likelihood_option {
            log_likelihood = out;
        } else {
            log_likelihood = f64::NEG_INFINITY;
        }

        let current_sequence_len = sequence_values.len();

        average_log_likelihood += (current_sequence_len as f64 / total_sequence_len as f64) * log_likelihood;
        
    }

    AmalgamHMMFitness {
        fitness: average_log_likelihood,
        states: Some(states),
        start_matrix: Some(start_matrix),
        transition_matrix: Some(transition_matrix),
    }

}