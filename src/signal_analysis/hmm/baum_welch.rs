use super::hmm_tools::{StateMatrix2D, StateMatrix3D};
use super::optimization_tracker::{TerminationCriterium, StateCollapseHandle};
use super::state::*;
use super::hmm_matrices::*;
use super::hmm_instance::*;
use super::optimization_tracker::*;

use std::collections::HashSet;

pub const BAUM_WELCH_MAX_ITERS_DEFAULT: usize = 500;
pub const BAUM_WELCH_TERMINATION_DEFAULT: TerminationCriterium = TerminationCriterium::PlateauConvergence {epsilon: 1e-4, plateau_len: 20, max_iterations: Some(BAUM_WELCH_MAX_ITERS_DEFAULT as u32)};

#[derive(Debug, Clone)]
pub struct BaumWelch {
    num_states: u16,

    initial_states: Option<Vec<State>>,
    initial_start_matrix: Option<StartMatrix>,
    initial_transition_matrix: Option<TransitionMatrix>,

    final_states: Option<Vec<State>>,
    final_start_matrix: Option<StartMatrix>,
    final_transition_matrix: Option<TransitionMatrix>,

    final_log_likelihood: Option<f64>,

    state_collapse_handle: StateCollapseHandle,
}

impl BaumWelch {
    pub fn new(num_states: u16) -> Self {

        Self {
            num_states,

            initial_states: None,
            initial_start_matrix: None,
            initial_transition_matrix: None,

            final_states: None,
            final_start_matrix: None,
            final_transition_matrix: None,

            final_log_likelihood: None,

            state_collapse_handle: StateCollapseHandle::Abort, // Default to aborting
        }
        
    }

    pub fn reset(&mut self) {
        self.initial_states = None;
        self.initial_start_matrix = None;
        self.initial_transition_matrix = None;

        self.final_states = None;
        self.final_start_matrix = None;
        self.final_transition_matrix = None;

        self.final_log_likelihood = None;

        // Assuming you want to reset `state_collapse_handle` to a default value
        self.state_collapse_handle = StateCollapseHandle::Abort;
    }

    pub fn set_initial_states(&mut self, states: Vec<State>) -> Result<(), BaumWelchError>{

        // Check if the number of states is correct according to the initial argument to the constructor
        let expected_num = self.num_states;
        let given_num = states.len();
        if self.num_states != states.len() as u16 {
            return Err(BaumWelchError::IncorrectNumberOfInitialStates{expected: expected_num, given: given_num as u16});
        }

        // Check if the states are valid by calling
        HMMInstance::check_states_validity(&states)
            .map_err(|error| BaumWelchError::InvalidInitialStateSet { error })?;

        self.initial_states = Some(states); // Set the initial states if everything is valid

        Ok(())
    }

    pub fn set_initial_start_matrix(&mut self, start_matrix: StartMatrix) -> Result<(), BaumWelchError> {
        
        // Check if the states are valid 
        HMMInstance::check_start_matrix_validity(&start_matrix, self.num_states as usize)
            .map_err(|error| BaumWelchError::InvalidInitialStartMatrix { error })?;

        self.initial_start_matrix = Some(start_matrix);

        Ok(())
    }

    pub fn set_initial_transition_matrix(&mut self, transition_matrix: TransitionMatrix) -> Result<(), BaumWelchError> {
        
        // Check if the states are valid
        HMMInstance::check_transition_matrix_validity(&transition_matrix, self.num_states as usize)
            .map_err(|error| BaumWelchError::InvalidInitialTransitionMatrix { error })?;

        self.initial_transition_matrix = Some(transition_matrix);

        Ok(())
    }

    pub fn set_state_collapse_handle(&mut self, state_collapse_handle: StateCollapseHandle) {

        self.state_collapse_handle = state_collapse_handle;
    }

    pub fn update_start_matrix(states: &[State], gammas: &StateMatrix2D<f64>, start_matrix: &mut StartMatrix) {
        for state in states {
            start_matrix[state] = gammas[state][0];
        }
    }

    // If a state has collapsed, the transition matrix values for the remaining states will still be computed
    pub fn update_transition_matrix(states: &[State],
        xis: &StateMatrix3D<f64>,
        gammas: &StateMatrix2D<f64>, 
        transition_matrix: &mut TransitionMatrix
    ) -> Result<(), BaumWelchError> {

        // Set aux variables to check for collapsed states
        let mut state_collapse: bool = false;
        let mut collapsed_states: Vec<usize> = Vec::new();


        for state_from in states {

            // Compute sum for gamma[state_from][t=0 .. t = gammas.len() - 1]
            let normalization_factor: f64 = gammas[state_from][0..gammas.shape().1-1].iter().sum();
            
            
            let epsilon = 1e-12;

            if normalization_factor < epsilon {
                state_collapse = true;
                collapsed_states.push(state_from.get_id());
            } else {
                for state_to in states {
                    // Sum over all time steps (still unnormalized), and assign to running transition matrix
                    transition_matrix[(state_from, state_to)] = xis[(state_from, state_to)].iter().sum();

                    

                    // Normalize
                    transition_matrix[(state_from, state_to)] /= normalization_factor;
                }
            }
        }

        if state_collapse {
            return Err(BaumWelchError::CollapsedStates { states: collapsed_states });
        }


        Ok(())

    }


    // If states have collapsed, the remaining ones still are updated
    pub fn update_states(
        observations: &[f64],
        viterbi_pred: &[usize],
        states: &mut [State]
    ) -> Result<(), BaumWelchError> {

        // Collect each occurrence of each state into its corresponding vector
        let mut per_state_collection: StateMatrix2D<f64> = StateMatrix2D::empty((states.len(), 0));

        for (observation, assigned_state) in observations.iter().zip(viterbi_pred.iter()) {
            per_state_collection[*assigned_state].push(*observation);
        }

        // Set aux variables to check for collapsed states
        let mut state_collapse: bool = false;
        let mut collapsed_states: Vec<usize> = Vec::new();

        for state in states {
            if per_state_collection[state.id].len() <= 1 {
                state_collapse = true;
                collapsed_states.push(state.get_id());
            } else {
                
                let state_observations = &per_state_collection[state.get_id()];

                // Compute mean of the observations for the state
                let mean: f64 = state_observations.iter().sum::<f64>() / state_observations.len() as f64;

                // Compute standard deviation of the observations
                let variance: f64 = state_observations.iter()
                    .map(|obs| (obs - mean).powi(2))
                    .sum::<f64>() / state_observations.len() as f64;
                let std_dev = variance.sqrt();

                // Update state.value and state.noise_std
                state.value = mean;
                state.noise_std = std_dev;
            }
        }

        if state_collapse {
            return Err(BaumWelchError::CollapsedStates { states: collapsed_states });
        }

        Ok(())
    }

    pub fn run_step(
        states: &mut [State],
        start_matrix: &mut StartMatrix,
        transition_matrix: &mut TransitionMatrix,
        observations: &[f64]
    ) -> Result<f64, BaumWelchError> {
        
        let mut hmm_instance = HMMInstance::new(states, start_matrix, transition_matrix);
        
        // println!("Before first");
        hmm_instance.run_viterbi(observations)
        .map_err(|error| BaumWelchError::HMMInstanceError { error })?;
        // println!("First is fine");

        // The below code was used for computing non scaled prob. matrices:
        
        // hmm_instance.run_all_probability_matrices(observations)
        // .map_err(|error| BaumWelchError::HMMInstanceError { error })?;
    
        hmm_instance.run_all_probability_matrices_with_scaled(observations)
        .map_err(|error| BaumWelchError::HMMInstanceError { error })?;

        // Take ownership of the relevant date from hmm_instance
        let viterbi_pred_option = hmm_instance.take_viterbi_prediction();
        let gammas_option = hmm_instance.take_gammas();
        let xis_option = hmm_instance.take_xis();
        let log_likelihood_option = hmm_instance.take_log_likelihood();

        // Extract stuff from the options
        let viterbi_pred = viterbi_pred_option.ok_or(BaumWelchError::ViterbiPredictionNotFound)?;
        let gammas = gammas_option.ok_or(BaumWelchError::HMMGammasNotFound)?;
        let xis = xis_option.ok_or(BaumWelchError::HMMXisNotFound)?;
        let log_likelihood = log_likelihood_option.ok_or(BaumWelchError::HMMObservationProbNotFound)?;

        Self::update_start_matrix(states, &gammas, start_matrix);
        let update_tr_mat_err = Self::update_transition_matrix(states, &xis, &gammas, transition_matrix);
        let update_states_err = Self::update_states(observations, &viterbi_pred, states);

        let mut collapsed_states: HashSet<usize> = HashSet::new();

        if let Err(BaumWelchError::CollapsedStates { states }) = update_tr_mat_err {
            collapsed_states.extend(states);
        }

        if let Err(BaumWelchError::CollapsedStates { states }) = update_states_err {
            collapsed_states.extend(states);
        }

        if collapsed_states.len() != 0 {return Err(BaumWelchError::CollapsedStates { states: collapsed_states.into_iter().collect() })}

        Ok(log_likelihood)
    }


    pub fn run_optimization(&mut self, observations: &[f64], termination_criterium: TerminationCriterium) -> Result<(&[State], &StartMatrix, &TransitionMatrix), BaumWelchError> {

        let mut tracker = OptimizationTracker::new(termination_criterium);

        if let StateCollapseHandle::RestartRandom { allowed_trials} = self.state_collapse_handle {
            tracker.set_max_resets(allowed_trials);
        }

        let (mut running_states, mut running_start_matrix, mut running_transition_matrix) =
        setup_baum_welch(&self, observations, InitMode::GivenInits);
        
        loop {
            
            let new_output = Self::run_step(
                &mut running_states,
                &mut running_start_matrix,
                &mut running_transition_matrix,
                observations);

            match new_output {
                Ok(new_eval) => {
                    self.final_log_likelihood = Some(new_eval);
                    if tracker.step(new_eval) {break}
                },

                Err(BaumWelchError::CollapsedStates { states }) => {

                    match self.state_collapse_handle {
                        StateCollapseHandle::Abort => {
                            return Err(BaumWelchError::CollapsedStates { states: states })
                        },

                        StateCollapseHandle::RestartRandom{..} => {
                            tracker.reset();

                            setup_with_random(
                                &self, observations,
                                &mut running_states,
                                &mut running_start_matrix,
                                &mut running_transition_matrix
                            );

                        },

                        StateCollapseHandle::RemoveCollapsedState => {
                            self.num_states -= states.len() as u16;
                            running_states = remove_from_state_vec(&running_states, &states);
                            running_start_matrix.remove_states(&states);
                            running_transition_matrix.remove_states(&states);
                        },
                    }

                },
                Err(err) => {return Err(err)}
            } 
            
        }

        self.final_states = Some(running_states);
        self.final_start_matrix = Some(running_start_matrix);
        self.final_transition_matrix = Some(running_transition_matrix);


        Ok((self.final_states.as_ref().unwrap(), self.final_start_matrix.as_ref().unwrap(), self.final_transition_matrix.as_ref().unwrap()))
    }

    pub fn get_states(&self) -> Option<&Vec<State>> {
        self.final_states.as_ref()
    }

    pub fn get_start_matrix(&self) -> Option<&StartMatrix> {
        self.final_start_matrix.as_ref()
    }

    pub fn get_transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.final_transition_matrix.as_ref()
    }

    pub fn get_log_likelihood(&self) -> Option<f64> {
        self.final_log_likelihood.clone()
    }
    
    pub fn take_states(&mut self) -> Option<Vec<State>> {
        self.final_states.take()
    }

    pub fn take_start_matrix(&mut self) -> Option<StartMatrix> {
        self.final_start_matrix.take()
    }

    pub fn take_transition_matrix(&mut self) -> Option<TransitionMatrix> {
        self.final_transition_matrix.take()
    }

    pub fn take_log_likelihood(&mut self) -> Option<f64> {
        self.final_log_likelihood.take()
    }
}


fn setup_baum_welch(baum_welch: &BaumWelch, observations: &[f64], init_mode: InitMode) 
    -> (Vec<State>, StartMatrix, TransitionMatrix) {

        let mut running_states: Vec<State> = Vec::new();
        let mut running_start_matrix: StartMatrix = StartMatrix::empty(baum_welch.num_states as usize);
        let mut running_transition_matrix: TransitionMatrix = TransitionMatrix::empty(baum_welch.num_states as usize);

        match init_mode {
            InitMode::GivenInits => {
                setup_with_given_inits(
                    baum_welch, observations, &mut running_states,
                    &mut running_start_matrix, &mut running_transition_matrix);
            }

            InitMode::Random => {
                setup_with_random(
                    baum_welch, observations, &mut running_states,
                    &mut running_start_matrix, &mut running_transition_matrix);
            }

            InitMode::Sparse => {
                setup_with_sparse(
                    baum_welch, observations, &mut running_states,
                    &mut running_start_matrix, &mut running_transition_matrix);
            }
        }


        (running_states, running_start_matrix, running_transition_matrix)
}

fn setup_with_given_inits(
    baum_welch: &BaumWelch,
    observations: &[f64],
    states: &mut Vec<State>,
    start_matrix: &mut StartMatrix,
    transition_matrix: &mut TransitionMatrix) {
      
    let observations_max = observations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let observations_min = observations.iter().cloned().fold(f64::INFINITY, f64::min);

    if let Some(init) = baum_welch.initial_states.as_ref() {
        *states = init.to_vec();
    } else {
        let std = (observations_max - observations_max) / baum_welch.num_states as f64;
        *states = HMMInstance::generate_sparse_states(baum_welch.num_states, observations_max, observations_min, std).unwrap();
    }

    if let Some(init) = baum_welch.initial_start_matrix.as_ref() {
        *start_matrix = init.clone();
    } else {
        *start_matrix = StartMatrix::new_random(baum_welch.num_states as usize);
    }

    if let Some(init) = baum_welch.initial_transition_matrix.as_ref() {
        *transition_matrix = init.clone();
    } else {
        *transition_matrix = TransitionMatrix::new_random(baum_welch.num_states as usize);
    }

}

fn setup_with_random(
    baum_welch: &BaumWelch,
    observations: &[f64],
    states: &mut Vec<State>,
    start_matrix: &mut StartMatrix,
    transition_matrix: &mut TransitionMatrix){
    
    let observations_max = observations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let observations_min = observations.iter().cloned().fold(f64::INFINITY, f64::min);

    *states = HMMInstance::generate_random_states(baum_welch.num_states, Some(observations_max), Some(observations_min), None, None).unwrap();
     
    *start_matrix = StartMatrix::new_random(baum_welch.num_states as usize);

    *transition_matrix = TransitionMatrix::new_random(baum_welch.num_states as usize);

}

fn setup_with_sparse(
    baum_welch: &BaumWelch,
    observations: &[f64],
    states: &mut Vec<State>,
    start_matrix: &mut StartMatrix,
    transition_matrix: &mut TransitionMatrix){
    
    let observations_max = observations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let observations_min = observations.iter().cloned().fold(f64::INFINITY, f64::min);

    let std = (observations_max - observations_max) / baum_welch.num_states as f64;

    *states = HMMInstance::generate_sparse_states(baum_welch.num_states, observations_max, observations_min, std).unwrap();
     
    *start_matrix = StartMatrix::new_random(baum_welch.num_states as usize);
    
    *transition_matrix = TransitionMatrix::new_random(baum_welch.num_states as usize);
}


#[derive(Debug, Clone)]
pub enum BaumWelchError {
    // Initialization errors
    IncorrectNumberOfInitialStates { expected: u16, given: u16},
    InvalidInitialStateSet {error: HMMInstanceError},
    InvalidInitialStartMatrix {error: HMMInstanceError},
    InvalidInitialTransitionMatrix {error: HMMInstanceError},
    HMMInstanceError { error: HMMInstanceError},

    // HMMInstance running errors
    ViterbiPredictionNotFound,
    HMMGammasNotFound,
    HMMXisNotFound,
    HMMObservationProbNotFound,

    // State collapse
    CollapsedStates {states: Vec<usize>},
}

pub enum InitMode {
    GivenInits,
    Random,
    Sparse,
}