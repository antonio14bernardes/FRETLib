use super::hmm_tools::{StateMatrix1D, StateMatrix2D, StateMatrix3D};
use super::optimization_tracker::{self, TerminationCriterium};
use super::{state::*, viterbi};
use super::hmm_matrices::*;
use super::hmm_instance::*;
use super::optimization_tracker::*;

#[derive(Debug)]
pub struct BaumWelch {
    num_states: u16,

    initial_states: Option<Vec<State>>,
    initial_start_matrix: Option<StartMatrix>,
    initial_transition_matrix: Option<TransitionMatrix>,

    final_states: Option<Vec<State>>,
    final_start_matrix: Option<StartMatrix>,
    final_transition_matrix: Option<TransitionMatrix>,
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
        }
        
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
            if per_state_collection[state.id].len() == 0 {
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
        
        hmm_instance.run_viterbi(observations)
        .map_err(|error| BaumWelchError::HMMInstanceError { error })?;

        // hmm_instance.run_all_probability_matrices(observations)
        // .map_err(|error| BaumWelchError::HMMInstanceError { error })?;
    
        hmm_instance.run_all_probability_matrices_with_scaled(observations)
        .map_err(|error| BaumWelchError::HMMInstanceError { error })?;

        // Take ownership of the relevant date from hmm_instance
        let viterbi_pred_option = hmm_instance.take_viterbi_prediction();
        let gammas_option = hmm_instance.take_gammas();
        let xis_option = hmm_instance.take_xis();
        let observations_prob_option = hmm_instance.take_observations_prob();



        // Debug stuff
        // let alphas = hmm_instance.take_alphas().unwrap();
        // let betas = hmm_instance.take_betas().unwrap();

        


        // Kill hmm_instance
        drop(hmm_instance);

        // Extract stuff from the options
        if viterbi_pred_option.is_none() {return Err(BaumWelchError::ViterbiPredictionNotFound)}
        if gammas_option.is_none() {return Err(BaumWelchError::HMMGammasNotFound)}
        if xis_option.is_none() {return Err(BaumWelchError::HMMXisNotFound)}
        if observations_prob_option.is_none() {return Err(BaumWelchError::HMMObservationProbNotFound)}

        let viterbi_pred = viterbi_pred_option.unwrap();
        let gammas = gammas_option.unwrap();
        let xis = xis_option.unwrap();
        let observations_prob = observations_prob_option.unwrap();

        // println!("New alphas: {:?}\n", alphas);
        // println!("New betas: {:?}\n", betas);
        // println!("New gammas: {:?}\n", gammas);
        // println!("New xis: {:?}\n", xis);
        // println!("\n\n\n\n");

        Self::update_start_matrix(states, &gammas, start_matrix);
        Self::update_transition_matrix(states, &xis, &gammas, transition_matrix)?;
        Self::update_states(observations, &viterbi_pred, states);


        Ok(observations_prob)
    }

    pub fn run_optimization(&mut self, observations: &[f64], termination_criterium: TerminationCriterium) -> Result<(&[State], &StartMatrix, &TransitionMatrix), BaumWelchError> {

        let mut tracker = OptimizationTracker::new(termination_criterium);

        let mut running_states: Vec<State>;
        let mut running_start_matrix: StartMatrix;
        let mut running_transition_matrix: TransitionMatrix;

        let observations_max = observations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let observations_min = observations.iter().cloned().fold(f64::INFINITY, f64::min);



        if let Some(init) = self.initial_states.as_ref() {
            running_states = init.to_vec();
        } else {
            running_states = HMMInstance::generate_random_states(self.num_states, Some(observations_max), Some(observations_min), None, None).unwrap();
        }

        if let Some(init) = self.initial_start_matrix.as_ref() {
            running_start_matrix = init.clone();
        } else {
            running_start_matrix = StartMatrix::new_random(self.num_states as usize);
        }

        if let Some(init) = self.initial_transition_matrix.as_ref() {
            running_transition_matrix = init.clone();
        } else {
            running_transition_matrix = TransitionMatrix::new_random(self.num_states as usize);
        }

        loop {

            let new_eval = Self::run_step(&mut running_states, &mut running_start_matrix, &mut running_transition_matrix, observations)?;


            if tracker.step(new_eval) {break}
        }

        self.final_states = Some(running_states);
        self.final_start_matrix = Some(running_start_matrix);
        self.final_transition_matrix = Some(running_transition_matrix);


        Ok((self.final_states.as_ref().unwrap(), self.final_start_matrix.as_ref().unwrap(), self.final_transition_matrix.as_ref().unwrap()))
    }
}

#[derive(Debug)]
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