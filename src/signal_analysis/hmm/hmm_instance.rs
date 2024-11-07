use std::collections::HashSet;

use rand_distr::{Normal, Distribution};
use rand::Rng;

use super::hmm_tools::{StateMatrix2D, StateMatrix3D};
use super::state::*;
use super::hmm_matrices::*;
use super::viterbi::*;
use super::probability_matrices::*;

pub const VALUE_SEQUENCE_THRESHOLD: f64 = 1e-7;

pub struct HMMInstance<'a> {
    states: Option<&'a [State]>,
    start_matrix:Option<&'a StartMatrix>,
    transition_matrix: Option<&'a TransitionMatrix>,

    viterbi: Viterbi<'a>,

    alphas: Option<StateMatrix2D<f64>>,  // Matrix with probabilities P(O_1, O_2, O_3, ..., O_t, state_t = Si | trans_mat, start_mat, state_set)
    alphas_scaled: Option<StateMatrix2D<f64>>,
    betas: Option<StateMatrix2D<f64>>, // Matrix with probabilities P(O_t+1, O_t+2, O_t+3, ..., O_T | state_t = Si, trans_mat, start_mat, state_set)
    betas_scaled: Option<StateMatrix2D<f64>>,
    scaling_factors: Option<Vec<f64>>,
    gammas: Option<StateMatrix2D<f64>>, // Probability of seeing state i on timestep t given both the previous and following observations and states
    xis: Option<StateMatrix3D<f64>>, // Probability of seeing transition i -> j on timestep t given everything we know 
    log_likelihood: Option<f64>, // P(Observations | States, Start Matrix, Transition Matrix)
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
            alphas_scaled: None,
            betas: None,
            betas_scaled: None,
            scaling_factors: None,
            gammas: None,
            xis: None,

            
        
            log_likelihood: None,
        }
    }
    
    pub fn new_empty() -> Self {
        let viterbi = Viterbi::new_empty();

        Self { states: None, start_matrix: None, transition_matrix: None, viterbi,
        alphas: None, alphas_scaled: None, betas: None, betas_scaled: None, scaling_factors: None,
        gammas: None, xis: None, log_likelihood: None }
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

    pub fn get_states(&self) -> Option<&[State]> {
        self.states
    }
    
    pub fn get_start_matrix(&self) -> Option<&StartMatrix> {
        self.start_matrix
    }
    
    pub fn get_transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix
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
        let expected_ids: Vec<usize> = (0..states.len()).collect(); // Expected sequence: [0, 1, 2, ..., N-1]
    
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
        // println!("start matrix was fine");
        Self::check_transition_matrix_validity(transition_matrix, states.len())?;
        // println!("transition matrix was fine");
        Self::check_states_validity(states)?;
        // println!("states was fine");

        Ok(())
    }

    pub fn check_sequence_validity(observations: &[f64], tolerance: f64) -> Result<(), HMMInstanceError> {
        if observations.len() == 0 {return Err(HMMInstanceError::EmptyValueSequence)}

        let min_value= observations.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_value= observations.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        let range = max_value - min_value;
        if range <= tolerance {return Err(HMMInstanceError::ConstantValueSequence)}

        if observations.iter().any(|value| value.is_nan()) {return Err(HMMInstanceError::ValueSequenceContainsNANs)}

        Ok(())
    }
    
    pub fn run_viterbi(&mut self, observations: &[f64]) -> Result<(),HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

        self.viterbi.run(observations, false)
    }

    pub fn run_alphas(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

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
        self.log_likelihood = Some(observations_prob.ln());

        Ok(())
    }

    pub fn run_betas(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

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
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;
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
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;
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
        self.run_xis(observations)?;

        let mut sum_vec: Vec<f64> = Vec::new();
        for t in 0..observations.len() - 1 {
            let mut sum = 0.0;
            for state_from in self.states.unwrap() {
                for state_to in self.states.unwrap() {
                    sum += &self.xis.as_ref().unwrap()[(state_from, state_to, t)];
                }
            }
            sum_vec.push(sum);
        }

        Ok(())

    }

    pub fn run_alphas_scaled(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {

        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let start_matrix = self.start_matrix.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();
        
        // Create the alphas matrix
        let mut alphas_scaled = StateMatrix2D::<f64>::empty((states.len(), observations.len()));        

        let scaling_factors = compute_scaled_alphas
        (states, observations, start_matrix, transition_matrix, &mut alphas_scaled);

        self.alphas_scaled = Some(alphas_scaled);

        self.scaling_factors = Some(scaling_factors);
        
        // Compute the log-likelihood by summing the logs of the scaling factors
        let log_likelihood: f64 = self
        .scaling_factors
        .as_ref()
        .unwrap()
        .iter()
        .map(|factor| factor.ln()) // Take the log of each scaling factor
        .sum(); // Sum the logs of the scaling factors

        self.log_likelihood = Some(log_likelihood);

        Ok(())
    }

    pub fn run_betas_scaled(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

        // Extract states and matrices
        let states = self.states.unwrap();
        let transition_matrix = self.transition_matrix.unwrap();

        // Extract scaling factors
        let scaling_factors = self.scaling_factors.as_ref().ok_or(HMMInstanceError::ScalingFactorsNotYetDefined)?;
        
        // Create the betas matrix
        let mut betas_scaled = StateMatrix2D::<f64>::empty((states.len(), observations.len()));
        
        compute_scaled_betas(states, observations, transition_matrix, &mut betas_scaled, scaling_factors);
        
        self.betas_scaled = Some(betas_scaled);
        
        Ok(())
    }

    pub fn run_gammas_with_scaled(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError>{
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;
        if self.alphas_scaled.is_none() || self.betas_scaled.is_none() { return Err(HMMInstanceError::AlphasORBetasNotYetDefined)}

        // Extract states and matrices
        let states = self.states.unwrap();
        let alphas_scaled: &StateMatrix2D<f64> = self.alphas_scaled.as_ref().unwrap();
        let betas_scaled = self.betas_scaled.as_ref().unwrap();

        // Create the gammas matrix
        let mut gammas = StateMatrix2D::<f64>::empty((states.len(), observations.len()));

        compute_gammas_with_scaled(states, observations, alphas_scaled, betas_scaled, &mut gammas);

        self.gammas = Some(gammas);

        Ok(())
    }

    pub fn run_xis_with_scaled(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError>{
        Self::check_validity(self.states, self.start_matrix, self.transition_matrix)?;
        Self::check_sequence_validity(observations, VALUE_SEQUENCE_THRESHOLD)?;

        if self.alphas_scaled.is_none() || self.betas_scaled.is_none() { return Err(HMMInstanceError::AlphasORBetasNotYetDefined)}
        if self.scaling_factors.is_none() {return Err(HMMInstanceError::ScalingFactorsNotYetDefined)}

        // Extract states and matrices
        let states: &[State] = self.states.unwrap();
        let transition_matrix: &TransitionMatrix = self.transition_matrix.unwrap();
        let alphas_scaled: &StateMatrix2D<f64> = self.alphas_scaled.as_ref().unwrap();
        let betas_scaled: &StateMatrix2D<f64> = self.betas_scaled.as_ref().unwrap();
        let scaling_factors = self.scaling_factors.as_ref().unwrap();

        // Create the gammas matrix
        let mut xis: StateMatrix3D<f64> = StateMatrix3D::<f64>::empty((states.len(), states.len(), observations.len()-1));

        compute_xis_with_scaled(states, observations, transition_matrix, alphas_scaled, betas_scaled, &mut xis, scaling_factors);

        self.xis = Some(xis);

        

        Ok(())
    }

    pub fn run_all_probability_matrices_with_scaled(&mut self, observations: &[f64]) -> Result<(), HMMInstanceError> {
        self.run_alphas_scaled(observations)?;
        self.run_betas_scaled(observations)?;
        self.run_gammas_with_scaled(observations)?;
        self.run_xis_with_scaled(observations)?;


        Ok(())

    }

    pub fn get_viterbi_prediction(&self) -> Option<&Vec<usize>> {
        self.viterbi.get_prediction()
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

    pub fn get_log_likelihood(&self) -> Option<&f64> {
        self.log_likelihood.as_ref()
    }

    pub fn take_viterbi_prediction(&mut self) -> Option<Vec<usize>> {
        self.viterbi.take_prediction()
    }

    pub fn take_alphas(&mut self) -> Option<StateMatrix2D<f64>> {
        self.alphas.take()
    }
    
    pub fn take_betas(&mut self) -> Option<StateMatrix2D<f64>> {
        self.betas.take()
    }
    
    pub fn take_gammas(&mut self) -> Option<StateMatrix2D<f64>> {
        self.gammas.take()
    }
    
    pub fn take_xis(&mut self) -> Option<StateMatrix3D<f64>> {
        self.xis.take()
    }

    pub fn take_log_likelihood(&mut self) -> Option<f64> {
        self.log_likelihood.take()
    }

    pub fn generate_random_states(
        num_states: u16,
        max_value: Option<f64>,
        min_value: Option<f64>,
        max_noise: Option<f64>,
        min_noise: Option<f64>,
    ) -> Result<Vec<State>, StateError> {
        loop {
            let mut states = Vec::with_capacity(num_states as usize);
    
            // Generate the states
            for id in 0..num_states {
                let state = State::new_random(
                    id as usize,
                    max_value,  // max value for the state
                    min_value,  // min value for the state
                    max_noise,  // max noise
                    min_noise,  // min noise
                )?;
                states.push(state);
            }
    
            // Validate the generated states, retry if invalid
            match HMMInstance::check_states_validity(&states) {
                Ok(_) => return Ok(states),  // Return if valid
                Err(_) => continue,          // Retry if validation fails
            }
        }
    }

    // The max and min values aren't those of the states but rather those of a possible observation.
    // This function will divide the value range into num_states intervals and set each state with the mean of each partition
    pub fn generate_sparse_states(
        num_states: u16,
        max_value: f64,
        min_value: f64,
        noise_std: f64,
    ) -> Result<Vec<State>, HMMInstanceError> {
        
        let value_range = max_value - min_value;
        if value_range <= 0.0 { return Err( HMMInstanceError::InvalidMaxMinValues { max: max_value, min: min_value })}

    
        let interval = value_range / num_states as f64;
        let mut states = Vec::with_capacity(num_states as usize);
    
        // Assign the mean value of each interval to the state
        for id in 0..num_states {
            let state_value = min_value + (id as f64 + 0.5) * interval;
            let state = State::new(id as usize, state_value, noise_std)
            .map_err(|error| HMMInstanceError::StateError { error })?;
            states.push(state);
        }
    
        // Validate generated states
        HMMInstance::check_states_validity(&states)?;
    
        Ok(states)
    }

    pub fn generate_state_set(
        values: &[f64],
        noise_std: &[f64],
    ) -> Result<Vec<State>, HMMInstanceError> {
        // Check if the dimensions of values and noise_std match
        if values.len() != noise_std.len() {
            return Err(HMMInstanceError::InvalidStateParametersLen);
        }
    
        let num_states = values.len();
        let mut states = Vec::with_capacity(num_states);
    
        // Generate states based on provided values and noise_std
        for (id, (&value, &std)) in values.iter().zip(noise_std.iter()).enumerate() {
            let state = State::new(id, value, std)
                .map_err(|error| HMMInstanceError::StateError { error })?;
            states.push(state);
        }
    
        // Validate generated states
        HMMInstance::check_states_validity(&states)?;
    
        Ok(states)
    }

    pub fn gen_sequence(
        states: &[State], 
        start_matrix: &StartMatrix, 
        transition_matrix: &TransitionMatrix, 
        time_steps: usize
    ) -> (Vec<usize>, Vec<f64>) {
        let mut rng = rand::thread_rng();
        let mut sequence = Vec::with_capacity(time_steps);
        let mut values = Vec::with_capacity(time_steps);
    
        // Start by choosing the initial state based on the start matrix
        let mut cumulative_prob = 0.0;
        let random_value = rng.gen_range(0.0..1.0);
        let mut current_state = 0;
    
        for (state, &start_prob) in start_matrix.matrix.iter().enumerate() {
            cumulative_prob += start_prob;
            if random_value < cumulative_prob {
                current_state = state;
                break;
            }
        }
        sequence.push(current_state);
    
        // Get theb objects for the Normal dist of each state
        let normal_distributions: Vec<Normal<f64>> = states
            .iter()
            .map(|state| Normal::new(state.value, state.noise_std).unwrap())
            .collect();
    
        // Sample the value for the initial state
        let initial_value = normal_distributions[current_state].sample(&mut rng);
        values.push(initial_value);
    
        // Generate the remaining states based on the transition matrix
        for _ in 1..time_steps {
            cumulative_prob = 0.0;
            let random_value = rng.gen_range(0.0..1.0);
            let transition_probs = &transition_matrix.matrix[current_state];
    
            for (next_state, &trans_prob) in transition_probs.iter().enumerate() {
                cumulative_prob += trans_prob;
                if random_value < cumulative_prob {
                    current_state = next_state;
                    break;
                }
            }
    
            sequence.push(current_state);
    
            // Sample from this state's dist to get the current value
            let state_value = normal_distributions[current_state].sample(&mut rng);
            values.push(state_value);
        }
    
        (sequence, values)
    }



    







}


#[derive(Debug, Clone)]
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
    DuplicateStateValues,                           // State values should be unique
    AlphasORBetasNotYetDefined,                     // Occurs when trying to compute gammas and xis without first computing alphas and betas
    ScalingFactorsNotYetDefined,

    InvalidMaxMinValues {max: f64, min: f64},       // Max must be > than min
    StateError {error: StateError},                 // Wrapper for a state error

    InvalidStateParametersLen,
    EmptyValueSequence,
    InvalidValueSequence,
    ConstantValueSequence,
    ValueSequenceContainsNANs,
}
