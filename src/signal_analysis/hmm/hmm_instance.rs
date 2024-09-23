use std::collections::HashSet;

use super::hmm_tools::{StateMatrix2D, StateMatrix3D};
use super::state::*;
use super::probability_matrices::*;
use super::viterbi::*;

pub struct HMMInstance<'a> {
    states: Option<&'a [State]>,
    start_matrix:Option<&'a StartMatrix>,
    transition_matrix: Option<&'a TransitionMatrix>,

    viterbi: Viterbi<'a>,

    alphas: Option<StateMatrix2D<f64>>,  // Matrix with probabilities of ending up in state i on timestep t
    betas: Option<StateMatrix2D<f64>>, // Matrix with probabilities of ending up in state i given the following observations and states
    gammas: Option<StateMatrix2D<f64>>, // Probability of seeing state i on timestep t given both the previous and following observations and states
    xis: Option<StateMatrix3D<f64>>, // Probability of seeing transition i -> j on timestep t given everything we know 
    observations_prob: Option<f64>, // P(Observations | States, Start Matrix, Transition Matrix)
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

            alphas: None,
            betas: None,
            gammas: None,
            xis: None,
            observations_prob: None,
        }
    }
    
    pub fn new_empty() -> Self {
        let viterbi = Viterbi::new_empty();

        Self { states: None, start_matrix: None, transition_matrix: None, viterbi,
        alphas: None, betas: None, gammas: None, xis: None, observations_prob: None }
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

    // Checks validity of the start matrix
    pub fn check_start_matrix_validity(
        start_matrix: &StartMatrix,
        num_states: usize,
    ) -> Result<(), HMMInstanceError> {
        // Validate dimensions of the start matrix
        if start_matrix.matrix.len() != num_states {
            return Err(HMMInstanceError::IncopatibleDimensions {
                dim_states: Some(num_states),
                dim_start_matrix: Some(start_matrix.matrix.len()), 
                dim_transition_matrix: None,
            });
        }

        // Validate the start matrix itself
        start_matrix
            .validate()
            .map_err(|error| HMMInstanceError::InvalidMatrix { error })?;

        Ok(())
    }

    // Checks validity of the transition matrix
    pub fn check_transition_matrix_validity(
        transition_matrix: &TransitionMatrix,
        num_states: usize,
    ) -> Result<(), HMMInstanceError> {
        // Validate the dimensions of the transition matrix
        let transition_matrix_dim_0 = transition_matrix.matrix.len();
        let transition_matrix_dim_1 = transition_matrix.matrix[0].len();

        if transition_matrix_dim_0 != num_states || transition_matrix_dim_1 != num_states {
            return Err(HMMInstanceError::IncopatibleDimensions {
                dim_states: Some(num_states),
                dim_start_matrix: None, 
                dim_transition_matrix: Some([transition_matrix_dim_0, transition_matrix_dim_1]),
            });
        }

        // Validate the transition matrix itself
        transition_matrix
            .validate()
            .map_err(|error| HMMInstanceError::InvalidMatrix { error })?;

        Ok(())
    }

    // Checks validity of the states
    pub fn check_states_validity(states: &[State]) -> Result<(), HMMInstanceError> {
        let mut state_ids: HashSet<usize> = HashSet::new();
        let mut state_values: Vec<f64> = Vec::new(); // Store state values here to manually check for duplicates
        let mut expected_ids: Vec<usize> = (0..states.len()).collect(); // Expected sequence: [0, 1, 2, ..., N-1]
    
        let epsilon = 1e-8; // Tolerance for comparing floating-point numbers
    
        for state in states {
            let state_id = state.get_id();
            let state_value = state.get_value(); // Assuming State has a method to get the value
    
            // Check for duplicate state IDs
            if !state_ids.insert(state_id) {
                return Err(HMMInstanceError::DuplicateStateId { id: state_id });
            }
    
            // Check for duplicate state values 
            if state_values.iter().any(|&v| (v - state_value).abs() < epsilon) {
                return Err(HMMInstanceError::DuplicateStateValues);
            }
    
            state_values.push(state_value);
        }
    
        // Check if the state IDs are exactly the expected sequence [0, 1, ..., N-1]
        let mut actual_ids: Vec<usize> = state_ids.into_iter().collect();
        actual_ids.sort_unstable();
    
        if actual_ids != expected_ids {
            return Err(HMMInstanceError::InvalidStateIdSequence {
                expected: expected_ids,
                found: actual_ids,
            });
        }
    
        Ok(())
    }

    pub fn check_validity(
        states: Option<&[State]>,
        start_matrix: Option<&StartMatrix>,
        transition_matrix: Option<&TransitionMatrix>,
    ) -> Result<(), HMMInstanceError> {
        // Unwrap the Option values or return the corresponding error
        let states = states.ok_or(HMMInstanceError::UndefinedStates)?;
        let start_matrix = start_matrix.ok_or(HMMInstanceError::UndefinedStartMatrix)?;
        let transition_matrix = transition_matrix.ok_or(HMMInstanceError::UndefinedTransitionMatrix)?;

        // Call the sub-functions to validate the components
        Self::check_start_matrix_validity(start_matrix, states.len())?;
        Self::check_transition_matrix_validity(transition_matrix, states.len())?;
        Self::check_states_validity(states)?;

        Ok(())
    }
    
    
    pub fn run_viterbi(&mut self, observations: &[f64]) -> Result<(),HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        self.viterbi.run(observations, false)
    }

    pub fn run_alphas(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let start_matrix = self.start_matrix.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
        
        // Create the alphas matrix
        let alphas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        self.alphas = Some(alphas);
        

        compute_alphas(states, observations, 
            start_matrix, transition_matrix,
            self.alphas.as_mut().unwrap());
        
        let mut observations_prob: f64 = 0.0;
        for state in self.states.unwrap() {
            observations_prob += self.alphas.as_ref().unwrap()[state][observations.len()-1]
        }
        self.observations_prob = Some(observations_prob);

        Ok(())
    }

    pub fn run_betas(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
        
        // Create the betas matrix
        let betas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        self.betas = Some(betas);
        

        compute_betas(states, observations, transition_matrix, self.betas.as_mut().unwrap());
        
        Ok(())
    }

    pub fn run_gammas(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError>{
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        if self.alphas.is_none() || self.betas.is_none() { return Err(HMMInstanceError::AlphasORBetasNotYetDefined)}


        // Extract states and matrices
        let states = self.states.unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        let betas = self.betas.as_ref().unwrap();

        // Create the gammas matrix
        let gammas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        self.gammas = Some(gammas);

        compute_gammas(states, observations, alphas, betas, self.gammas.as_mut().unwrap());


        Ok(())
    }

    pub fn run_xis(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError>{
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        if self.alphas.is_none() || self.betas.is_none() { return Err(HMMInstanceError::AlphasORBetasNotYetDefined)}


        // Extract states and matrices
        let states: &[State] = self.states.unwrap();
        let transition_matrix: &TransitionMatrix = self.transition_matrix.unwrap();
        let alphas: &StateMatrix2D<f64> = self.alphas.as_ref().unwrap();
        let betas: &StateMatrix2D<f64> = self.betas.as_ref().unwrap();

        // Create the gammas matrix
        let xis: StateMatrix3D<f64> = StateMatrix3D::<f64>::empty((states.len(), states.len(), observations.len()-1));
        self.xis = Some(xis);

        compute_xis(states, observations, transition_matrix, alphas, betas, self.xis.as_mut().unwrap());


        Ok(())
    }

    pub fn run_all_probability_matrices(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        self.run_alphas(observations)?;
        self.run_betas(observations)?;
        self.run_gammas(observations)?;
        self.run_xis(observations)
    }

    pub fn get_viterbi_prediction(&self) -> &Option<Vec<usize>> {
        &self.viterbi.get_prediction()
    }

    pub fn get_alphas(&self) -> Option<&StateMatrix2D<f64>> {
        if let Some(matrix) = self.alphas.as_ref() {
            return Some(matrix)
        } else { return None }
    }

    pub fn get_betas(&self) -> Option<&StateMatrix2D<f64>> {
        if let Some(matrix) = self.betas.as_ref() {
            return Some(matrix)
        } else { return None }
    }

    pub fn get_gammas(&self) -> Option<&StateMatrix2D<f64>> {
        if let Some(matrix) = self.gammas.as_ref() {
            return Some(matrix)
        } else { return None }
    }

    pub fn get_xis(&self) -> Option<&StateMatrix3D<f64>> {
        if let Some(matrix) = self.xis.as_ref() {
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
        dim_states: Option<usize>,
        dim_start_matrix: Option<usize>, 
        dim_transition_matrix: Option<[usize;2]>,
    },
    InvalidMatrix {error: MatrixValidationError},   // If there is an error from the probability matrix validation
    DuplicateStateId { id: usize },                 // ID is repeated between states
    InvalidStateIdSequence {                        // State IDs are not in sequence starting from 0
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    DuplicateStateValues,                             // State values should be unique
    AlphasORBetasNotYetDefined,                     // Occurs when trying to compute gammas and xis without first computing alphas and betas
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


fn compute_gammas(
    states: &[State], observations: &[f64],
    alphas: &StateMatrix2D<f64>, betas: &StateMatrix2D<f64>, gammas: &mut StateMatrix2D<f64>
) {
    for i in 0..observations.len() {

        let mut sum: f64 = 0.0;
        for state in states {
            let unnormalized_gamma = alphas[state][i] * betas[state][i];
            sum += unnormalized_gamma;
            gammas[state][i] = unnormalized_gamma;
        }

        for state in states {
            gammas[state][i] /= sum;
        }
    }
}


fn compute_xis(states: &[State], observations: &[f64], transition_matrix: &TransitionMatrix,
    alphas: &StateMatrix2D<f64>, betas: &StateMatrix2D<f64>, xis: &mut StateMatrix3D<f64>
) {
    for i in 0..observations.len()-1 {
        let mut sum: f64 = 0.;
        for state_from in states {
            for state_to in states {
                let unnormalized_xi = alphas[state_from][i]
                            * transition_matrix[(state_from, state_to)]
                            * state_to.standard_gaussian_emission_probability(observations[i+1])
                            * betas[state_to][i+1];
                
                sum += unnormalized_xi;

                xis[(state_from, state_to, i)] = unnormalized_xi;
            }
        }

        for state_from in states {
            for state_to in states {
                xis[(state_from, state_to, i)] /= sum;
            }
        }        
    }
}