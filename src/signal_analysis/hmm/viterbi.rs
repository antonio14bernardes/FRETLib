use std::backtrace;

use super::state::*;
use super::probability_matrices::{TransitionMatrix, StartMatrix, ProbabilityMatrix, MatrixValidationError};
use super::hmm_tools::{StateMatrix1D, StateMatrix2D};

pub struct Viterbi<'a> {
    states: Option<&'a [State]>,
    start_matrix:Option<&'a StartMatrix>,
    transition_matrix: Option<&'a TransitionMatrix>,

    viterbi_prob: Option<Vec<Vec<f64>>>,
    bactrace: Option<Vec<Vec<f64>>>,
    ml_path: Option<Vec<usize>>
}

impl<'a> Viterbi<'a> {
    pub fn new(states: &'a [State], 
        start_matrix: &'a StartMatrix, 
        transition_matrix: &'a TransitionMatrix) -> Self  {

        
        Self{
            states: Some(states),
            start_matrix: Some(start_matrix), 
            transition_matrix: Some(transition_matrix),

            viterbi_prob: None,
            bactrace: None,
            ml_path: None,
        }
    }

    pub fn new_empty() -> Self {
        Self{
            states: None,
            start_matrix: None, 
            transition_matrix: None,

            viterbi_prob: None,
            bactrace: None,
            ml_path: None,
        }
    }

    pub fn set_states(&mut self, states: &'a [State]) {
        self.states = Some(states);
    }

    pub fn set_start_matrix(&mut self, start_matrix: &'a StartMatrix) {
        self.start_matrix = Some(start_matrix);
    }

    pub fn set_transition_matrix(&mut self, transition_matrix: &'a TransitionMatrix) {
        self.transition_matrix = Some(transition_matrix);
    }

    pub fn reset(&mut self) {
        self.states = None;
        self.start_matrix = None; 
        self.transition_matrix = None;

        self.viterbi_prob = None;
        self.bactrace = None;
        self.ml_path = None;
    }

    // Check input validity
    pub fn pre_run_validity(&self) -> Result<(), ViterbiError> {

        // Check if any of the input matrices are None
        if self.states.is_none() { return Err(ViterbiError::UndefinedStates) }
        if self.start_matrix.is_none() { return Err(ViterbiError::UndefinedStartMatrix) }
        if self.transition_matrix.is_none() { return Err(ViterbiError::UndefinedTransitionMatrix) }

        // Unwrap the matrices
        let states = self.states.unwrap();
        let start_matrix = self.start_matrix.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();

        // Get dims
        let states_dim = states.len();
        let start_matrix_dim = start_matrix.matrix.len();
        let transition_matrix_dim_0 = transition_matrix.matrix.len();
        let transition_matrix_dim_1 = transition_matrix.matrix[0].len();

        // Check compatibility of dims
        if !(states_dim == start_matrix_dim && states_dim == transition_matrix_dim_0) {
            return Err(ViterbiError::IncopatibleDimensions { 
                dim_states: states_dim, 
                dim_start_matrix: start_matrix_dim,
                dim_transition_matrix: [transition_matrix_dim_0, transition_matrix_dim_1] })
        }

        // Check intrinsic validity of probability matrices
        start_matrix.validate().map_err(|error| ViterbiError::InvalidMatrix { error })?;
        transition_matrix.validate().map_err(|error| ViterbiError::InvalidMatrix { error })?;

        Ok(())
    }

    pub fn setup_algo(&self, time_trace: &[f64]) -> 
    Result<(StateMatrix2D<f64>, StateMatrix2D<usize>, StateMatrix1D<usize>), ViterbiError> {
        // Check validity of inputs
        self.pre_run_validity()?;

        // Get dimensions
        let num_states = self.states.unwrap().len();
        let num_observations = time_trace.len();

        // Create the viterbi algo related matrices from StateMatrix objects
        let mut viterbi_probs = StateMatrix2D::<f64>::empty((num_states, num_observations));
        let mut backtrace_mat = StateMatrix2D::<usize>::empty((num_states, num_observations - 1));

        let mut ml_path = StateMatrix1D::<usize>::empty(num_observations);

        Ok((viterbi_probs, backtrace_mat, ml_path))
    }

    pub fn run(&mut self, time_trace: &[f64]) -> Result<(),ViterbiError> {
        
        // Collect the allocated memory
        let (mut viterbi_probs, mut bactrace_mat, mut ml_path) = self.setup_algo(time_trace)?;

        // Unwrap states and matrices
        let states = self.states.unwrap();
        let start_matrix = self.start_matrix.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
    
        /*********** Viterbi's algorithn ***********/

        // Run first step of the Viterbi algo using the start probabilities
        for state in states {
            // Calculate the initial Viterbi probability for each state
            let start_prob = start_matrix[state];  // Start probability for the state
            let emission_prob = state.standard_gaussian_emission_probability(time_trace[0]);  // Emission probability for the first observation
        
            // Store the initial Viterbi probability in the viterbi_probs
            viterbi_probs[state][0] = start_prob * emission_prob;
        
            // No need to change backtrace matrix since it is created for num_time_steps - 1 
        }

        // Run remaining steps of Viterbi using the transition matrices
        for i in 1..time_trace.len() {
            for next_state in states {
                let mut max_prob: f64 = 0.;
                let mut best_prev_state: usize = 0;

                let emission_prob = next_state.standard_gaussian_emission_probability(time_trace[i]); // Emission probability for the ith observation

                for previous_state in states {
                    let transition_prob = transition_matrix[(previous_state, next_state)]; // Transition probability for the transition prev_state -> next_state

                    let total_prob = transition_prob * emission_prob * viterbi_probs[previous_state][i-1]; // Total probability until now

                    // If better result, update trackers
                    if total_prob > max_prob {
                        max_prob = total_prob;
                        best_prev_state = previous_state.id;
                    }   
                }

                // Update the viterbi and backtrece matrices
                viterbi_probs[next_state][i] = max_prob;
                bactrace_mat[next_state][i-1] = best_prev_state;
                
            }
        }
    

        /*********** Backtrace ***********/

        // Find best last state
        let mut last_state = 0;
        let mut max_final_prob = 0.0;

        for state in states {
            let final_prob = viterbi_probs[state][time_trace.len() - 1];
            if final_prob > max_final_prob {
                max_final_prob = final_prob;
                last_state = state.id;
            }
        }

        ml_path[time_trace.len() - 1] = last_state;


        // Follow the backtrace matrix backwards to get the most likely states
        let mut next_state = last_state;
        for t in (0..time_trace.len() - 1).rev() {
            // Get the previous state from the backtrace matrix
            let prev_state = bactrace_mat[next_state][t] as usize;
            ml_path[t] = prev_state;

            // Update the last_state to continue the backtrace
            next_state = prev_state;
        }


        Ok(())
    }
}

#[derive(Debug)]
pub enum ViterbiError {
    UndefinedStates,                                // States field is None
    UndefinedStartMatrix,                           // StartMatrix is None
    UndefinedTransitionMatrix,                      // TransitionMatrix is None
    IncopatibleDimensions {                         // Dimentions of the inputs do not match
        dim_states: usize,
        dim_start_matrix: usize, 
        dim_transition_matrix: [usize;2]
    },
    InvalidMatrix {error: MatrixValidationError}    // If there is an error from the probability matrix validation
}