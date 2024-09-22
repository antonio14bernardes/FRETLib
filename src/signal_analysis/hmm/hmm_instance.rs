use std::collections::HashSet;

use super::hmm_tools::StateMatrix2D;
use super::state::*;
use super::probability_matrices::*;
use super::viterbi::*;

pub struct HMMInstance<'a> {
    states: Option<&'a [State]>,
    start_matrix:Option<&'a StartMatrix>,
    transition_matrix: Option<&'a TransitionMatrix>,

    viterbi: Viterbi<'a>,

    forward_mat: Option<StateMatrix2D<f64>>,  // Matrix with probabilities of ending up in state i on timestep j
    final_prob: Option<f64>, // P(Observations | States, Start Matrix, Transition Matrix)
    backward_mat: Option<StateMatrix2D<f64>>, // Matrix with probabilities of ending up in state i given the following observations and states
}

impl<'a> HMMInstance<'a> {
    pub fn new(states: &'a [State], 
        start_matrix: &'a StartMatrix, 
        transition_matrix: &'a TransitionMatrix) -> Self  {

        let viterbi = Viterbi::new(states, start_matrix, transition_matrix);

        
        Self{
            states: Some(states),
            start_matrix: Some(start_matrix), 
            transition_matrix: Some(transition_matrix),

            viterbi,

            forward_mat: None,
            final_prob: None,
            backward_mat: None
        }
    }
    
    pub fn new_empty() -> Self {
        let viterbi = Viterbi::new_empty();

        Self { states: None, start_matrix: None, transition_matrix: None, viterbi,
        forward_mat: None, final_prob: None, backward_mat: None }
    }

    pub fn reset(&mut self) {
        self.states = None;
        self.start_matrix = None;
        self.transition_matrix = None;

        self.viterbi.reset();
    }

    pub fn set_states(&mut self, states: &'a [State]) {
        self.states = Some(states);
        self.viterbi.set_states(states);
    }

    pub fn set_start_matrix(&mut self, start_matrix: &'a StartMatrix) {
        self.start_matrix = Some(start_matrix);
        self.viterbi.set_start_matrix(start_matrix);
    }

    pub fn set_transition_matrix(&mut self, transition_matrix: &'a TransitionMatrix) {
        self.transition_matrix = Some(transition_matrix);
        self.viterbi.set_transition_matrix(transition_matrix);
    }

    pub fn check_validity(
        states: Option<&'a [State]>,
        start_matrix: Option<&'a StartMatrix>,
        transition_matrix: Option<&'a TransitionMatrix>
    ) -> Result<(), HMMInstanceError> {
        // Check if any of the input matrices are None
        if states.is_none() { return Err(HMMInstanceError::UndefinedStates) }
        if start_matrix.is_none() { return Err(HMMInstanceError::UndefinedStartMatrix) }
        if transition_matrix.is_none() { return Err(HMMInstanceError::UndefinedTransitionMatrix) }

        // Unwrap the matrices
        let states = states.unwrap();
        let start_matrix = start_matrix.unwrap();
        let transition_matrix = transition_matrix.unwrap();

        // Get dims
        let states_dim = states.len();
        let start_matrix_dim = start_matrix.matrix.len();
        let transition_matrix_dim_0 = transition_matrix.matrix.len();
        let transition_matrix_dim_1 = transition_matrix.matrix[0].len();

        // Check compatibility of dims
        if !(states_dim == start_matrix_dim && states_dim == transition_matrix_dim_0) {
            return Err(HMMInstanceError::IncopatibleDimensions { 
                dim_states: states_dim, 
                dim_start_matrix: start_matrix_dim,
                dim_transition_matrix: [transition_matrix_dim_0, transition_matrix_dim_1] })
        }

        // Check intrinsic validity of probability matrices
        start_matrix.validate().map_err(|error| HMMInstanceError::InvalidMatrix { error })?;
        transition_matrix.validate().map_err(|error| HMMInstanceError::InvalidMatrix { error })?;

        // Check if all state IDs are unique and in sequence from 0 to N-1
        let mut state_ids: HashSet<usize> = HashSet::new();
        let mut expected_ids: Vec<usize> = (0..states_dim).collect(); // Expected sequence: [0, 1, 2, ..., N-1]

        for state in states {
            let state_id = state.get_id();
            if !state_ids.insert(state_id) {
                return Err(HMMInstanceError::DuplicateStateId { id: state_id });
            }
        }

        // Check if the state IDs are exactly the expected sequence [0, 1, ..., N-1]
        let mut actual_ids: Vec<usize> = state_ids.into_iter().collect();
        actual_ids.sort_unstable(); // Sort to compare with expected

        if actual_ids != expected_ids {
            return Err(HMMInstanceError::InvalidStateIdSequence {
                expected: expected_ids,
                found: actual_ids,
            });
        }

    


        Ok(())
    }

    pub fn run_viterbi(&mut self, observations: &[f64]) -> Result<(),HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        self.viterbi.run(observations, false)
    }

    pub fn run_forward(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let start_matrix = self.start_matrix.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
        
        // Create the alphas matrix
        let alphas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        self.forward_mat = Some(alphas);
        

        compute_alphas(states, observations, 
            start_matrix, transition_matrix,
            self.forward_mat.as_mut().unwrap());
        
        let mut final_prob: f64 = 0.0;
        for state in self.states.unwrap() {
            final_prob += self.forward_mat.as_ref().unwrap()[state][observations.len()-1]
        }
        self.final_prob = Some(final_prob);

        Ok(())
    }

    pub fn run_backward(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
        
        // Create the alphas matrix
        let betas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        self.backward_mat = Some(betas);
        

        compute_betas(states, observations, transition_matrix, self.backward_mat.as_mut().unwrap());
        
        Ok(())
    }

    pub fn get_viterbi_prediction(&self) -> &Option<Vec<usize>> {
        &self.viterbi.get_prediction()
    }

    pub fn get_forward_probabilities_matrix(&self) -> Option<&StateMatrix2D<f64>> {
        if let Some(matrix) = self.forward_mat.as_ref() {
            return Some(matrix)
        } else { return None }
    }

    pub fn get_backward_probabilities_matrix(&self) -> Option<&StateMatrix2D<f64>> {
        if let Some(matrix) = self.backward_mat.as_ref() {
            return Some(matrix)
        } else { return None }
    }





}


#[derive(Debug)]
pub enum HMMInstanceError {
    UndefinedStates,                                // States field is None
    UndefinedStartMatrix,                           // StartMatrix is None
    UndefinedTransitionMatrix,                      // TransitionMatrix is None
    IncopatibleDimensions {                         // Dimentions of the inputs do not match
        dim_states: usize,
        dim_start_matrix: usize, 
        dim_transition_matrix: [usize;2]
    },
    InvalidMatrix {error: MatrixValidationError},   // If there is an error from the probability matrix validation
    DuplicateStateId { id: usize },                 // ID is repeated between states
    InvalidStateIdSequence {                        // State IDs are not in sequence starting from 0
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}


fn compute_alpha_i_t(
    current_state: &State, time_step: usize, observation_value: &f64,
    start_matrix: &StartMatrix, transition_matrix: &TransitionMatrix, 
    alphas_matrix: &mut StateMatrix2D<f64>) {

    if time_step == 0 {
        // Base case: alpha at time step 0 is just the start probability * emission probability
        alphas_matrix[current_state.get_id()][time_step] = start_matrix[current_state];
    } else {
        let mut new_value: f64 = 0.;

        // Sum over all previous states, following the transition probabilities
        for previous_state_id in 0..start_matrix.matrix.len() {
            let previous_alpha = alphas_matrix[previous_state_id][time_step - 1];

            let transition_prob = transition_matrix[(previous_state_id, current_state.id)];

            new_value += previous_alpha * transition_prob;
        }

        new_value *= current_state.standard_gaussian_emission_probability(*observation_value);

        alphas_matrix[current_state.get_id()][time_step] = new_value;

    }

    
}

fn compute_alphas(
    states: &[State], observations: &[f64],
    start_matrix: &StartMatrix, transition_matrix: &TransitionMatrix,
    alphas_matrix: &mut StateMatrix2D<f64>) {

    
    for (time_step, observation_value) in observations.iter().enumerate(){
        for state in states {
            compute_alpha_i_t(
                state, time_step, observation_value, 
                start_matrix, transition_matrix, alphas_matrix
            );
        }
    }

}

fn compute_beta_i_t(current_state: &State, time_step: usize, observations: &[f64],
    states: &[State], transition_matrix: &TransitionMatrix, betas_matrix: &mut StateMatrix2D<f64>) {

    if time_step == observations.len() - 1 { // time_step_from_end = 0 for the last time step
        betas_matrix[current_state][time_step] = 1.;
    } else {
        let mut sum:f64 = 0.0;
        for next_state in states {
            let transition_prob = transition_matrix[(current_state, next_state)];
            let emission_prob = next_state.standard_gaussian_emission_probability(observations[time_step + 1]);
            let prev_beta = betas_matrix[next_state][time_step + 1];
            sum +=  transition_prob * emission_prob * prev_beta;
        }

        betas_matrix[current_state][time_step] = sum;
    }


}

fn compute_betas(
    states: &[State], observations: &[f64],
    transition_matrix: &TransitionMatrix, betas_matrix: &mut StateMatrix2D<f64>) {
    
    for time_step in (0..observations.len()).rev() {
        for state in states {
            compute_beta_i_t(state, time_step, observations, states, transition_matrix, betas_matrix);
        }
    }
}