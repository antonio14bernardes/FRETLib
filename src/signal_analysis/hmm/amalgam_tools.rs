use crate::optimization::amalgam_idea;
use crate::optimization::optimizer::OptimizationFitness;

use super::super::super::optimization::amalgam_idea::*;
use super::hmm_tools::{StateMatrix1D, StateMatrix2D, StateMatrix3D};
use super::optimization_tracker::{TerminationCriterium, StateCollapseHandle};
use super::{state::*, viterbi};
use super::hmm_instance::*;
use super::hmm_matrices::*;
use super::baum_welch::*;

use rand::{seq, thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct AmalgamHMMFitness {
    pub fitness: f64,
    pub states: Option<Vec<State>>,
    pub start_matrix: Option<StartMatrix>,
    pub transition_matrix: Option<TransitionMatrix>, 
}

impl OptimizationFitness for AmalgamHMMFitness {
    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}

impl AmalgamHMMFitness {
    pub fn new(fitness: f64) -> Self {
        Self {
            fitness,
            states: None,
            start_matrix: None,
            transition_matrix: None,
        }
    }

    pub fn new_full
    (
        fitness: f64,
        states: Option<Vec<State>>,
        start_matrix: Option<StartMatrix>,
        transition_matrix: Option<TransitionMatrix>
    ) -> Self {
        Self {fitness, states, start_matrix, transition_matrix}
    } 
}

fn add_jitter_to_states(states: &Vec<State>, rng: &mut impl Rng) -> Vec<State> {
    let jittered_states= states
    .iter()
    .map(|state| {
        let jitter_amount = state.get_value() / 100.0; // Jitter is 1% of the state's value
        let new_value = state.get_value() + jitter_amount * (rng.gen::<f64>() - 0.5);
        State::new(state.id, new_value, state.get_noise_std()).unwrap()
    })
    .collect();

    jittered_states
}

fn set_initial_states_with_jitter_baum(
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
                jittered_states = add_jitter_to_states(states, rng);

            }
            Err(e) => return Err(e), // For any other error, return immediately
        }
    }
    Err(BaumWelchError::InvalidInitialStateSet{error: HMMInstanceError::DuplicateStateValues})
}


pub fn fitness_fn_baum_welch(individual: &[f64], num_states: usize, sequence_values: &[f64]) -> AmalgamHMMFitness {

    // println!("New individual: {:?}", individual);
    // Retrieve the states that the optimization algo wants to evaluate

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
    set_initial_states_with_jitter_baum(&mut baum, &trial_states, &mut rng, 10).unwrap();
    baum.set_initial_start_matrix(initial_start_matrix).unwrap();
    baum.set_initial_transition_matrix(initial_transition_matrix).unwrap();
    baum.set_state_collapse_handle(StateCollapseHandle::Abort);

    let termination_criterium = TerminationCriterium::PlateauConvergence {epsilon: 1e-4, plateau_len: 20, max_iterations: Some(500)};

    let output_res = baum.run_optimization(&sequence_values, termination_criterium);

    let fitness: f64;

    if let Ok(_) = output_res {
        fitness = baum.take_log_likelihood().unwrap();
    } else {
        println!("Result: {:?}",output_res);
        fitness = f64::NEG_INFINITY;
    }

    let fitness_struct = AmalgamHMMFitness::new(fitness);
    fitness_struct
    
}



pub fn fitness_fn_direct(individual: &[f64], sequence_values: &[f64]) -> AmalgamHMMFitness {
    let num_states = 3; // Adjust this to reflect your actual number of states

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

    let fitness: f64;

    if let Some(out) = log_likelihood_option {
        fitness = out;
    } else {
        fitness = f64::NEG_INFINITY;
    }

    AmalgamHMMFitness::new(fitness)

}
