use super::hmm_tools::{StateMatrix2D, StateMatrix3D};
use super::state::*;
use super::probability_matrices::*;
use super::hmm_instance::*;

pub struct BaumWelch {
    num_states: usize,

    initial_states: Option<Vec<State>>,
    initial_start_matrix: Option<StartMatrix>,
    initial_transition_matrix: Option<TransitionMatrix>,

    final_states: Option<Vec<State>>,
    final_start_matrix: Option<StartMatrix>,
    final_transition_matrix: Option<TransitionMatrix>,
}


impl BaumWelch {
    pub fn new(num_states: usize) -> Self {

        Self {
            num_states,

            initial_states: None,
            initial_start_matrix: None,
            initial_transition_matrix: None,

            final_states: None,
            final_start_matrix: None,
            final_transition_matrix: None,
        }
        
    }

    pub fn set_initial_states(&mut self, states: Vec<State>) -> Result<(), BaumWelchError>{

        // Check if the number of states is correct according to the initial argument to the constructor
        let expected_num = self.num_states;
        let given_num = states.len();
        if self.num_states != states.len() {
            return Err(BaumWelchError::IncorrectNumberOfInitialStates{expected: expected_num, given: given_num});
        }

        // Check if the states are valid by calling
        HMMInstance::check_states_validity(&states)
            .map_err(|error| BaumWelchError::InvalidInitialStateSet { error })?;

        self.initial_states = Some(states); // Set the initial states if everything is valid

        Ok(())
    }

    pub fn set_initial_start_matrix(&mut self, start_matrix: StartMatrix) -> Result<(), BaumWelchError> {
        
        // Check if the states are valid 
        HMMInstance::check_start_matrix_validity(&start_matrix, self.num_states)
            .map_err(|error| BaumWelchError::InvalidInitialStartMatrix { error })?;

        self.initial_start_matrix = Some(start_matrix);

        Ok(())
    }

    pub fn set_initial_transition_matrix(&mut self, transition_matrix: TransitionMatrix) -> Result<(), BaumWelchError> {
        
        // Check if the states are valid
        HMMInstance::check_transition_matrix_validity(&transition_matrix, self.num_states)
            .map_err(|error| BaumWelchError::InvalidInitialTransitionMatrix { error })?;

        self.initial_transition_matrix = Some(transition_matrix);

        Ok(())
    }

    pub fn update_start_matrix(states: &[State], gammas: StateMatrix2D<f64>, start_matrix: &mut StartMatrix) {
        for state in states {
            start_matrix[state] = gammas[state][0];
        }
    }

    pub fn update_transition_matrix(states: &[State],
        xis: StateMatrix3D<f64>,
        gammas: StateMatrix2D<f64>, 
        transition_matrix: &mut TransitionMatrix
    ) {

       
    }

    pub fn update_states() {

       
    }

    pub fn run_step(
        states: &[State],
        start_matrix: &StartMatrix,
        transition_matrix: &TransitionMatrix
    ) -> Result<(), BaumWelchError> {





        Ok(())
    }
}


pub enum BaumWelchError {
    IncorrectNumberOfInitialStates { expected: usize, given: usize},
    InvalidInitialStateSet {error: HMMInstanceError},
    InvalidInitialStartMatrix {error: HMMInstanceError},
    InvalidInitialTransitionMatrix {error: HMMInstanceError},
    HMMInstanceError { error: HMMInstanceError},
    HMMAlphasNotFound,
    HMMBetasNotFound,
    HMMGammasNotFound,
    HMMXisNotFound,
}