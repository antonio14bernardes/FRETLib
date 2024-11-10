use core::f64;

use crate::signal_analysis::hmm::{hmm_instance::{HMMInstance, HMMInstanceError, VALUE_SEQUENCE_THRESHOLD}, StartMatrix, State, TransitionMatrix};

#[derive(Debug)]
pub struct HMMAnalyzer {
    sequence_set: Option<Vec<Vec<f64>>>,

    states: Option<Vec<State>>,
    start_matrix: Option<StartMatrix>,
    transition_matrix: Option<TransitionMatrix>,

    sequence_states: Option<Vec<Vec<usize>>>,
    state_occupancy: Option<Vec<f64>>,
}

impl HMMAnalyzer {
    pub fn new() -> Self {
        Self {
            sequence_set: None,
            
            states: None,
            start_matrix: None,
            transition_matrix: None,

            sequence_states: None,
            state_occupancy: None,
        }
    }

    pub fn set_sequence_set(&mut self, sequence_set: &Vec<Vec<f64>>) -> Result<(), HMMAnalyzerError> {
        for sequence_values in sequence_set {
            HMMInstance::check_sequence_validity(sequence_values, VALUE_SEQUENCE_THRESHOLD)
            .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;
        }
        
        self.sequence_set = Some(sequence_set.clone());

        Ok(())
    }

    pub fn set_states(&mut self, states: Vec<State>) -> Result<(), HMMAnalyzerError> {
        HMMInstance::check_states_validity(&states)
        .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;

        self.states = Some(states);

        Ok(())
    }

    pub fn set_start_matrix(&mut self, start_matrix: StartMatrix) -> Result<(), HMMAnalyzerError> {
        if self.states.is_none() {return Err(HMMAnalyzerError::StatesNotDefined)}

        let num_states = self.states.as_ref().unwrap().len();

        HMMInstance::check_start_matrix_validity(&start_matrix, num_states)
        .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;

        self.start_matrix = Some(start_matrix);

        Ok(())
    }

    pub fn set_transition_matrix(&mut self, transition_matrix: TransitionMatrix) -> Result<(), HMMAnalyzerError> {
        if self.states.is_none() {return Err(HMMAnalyzerError::StatesNotDefined)}

        let num_states = self.states.as_ref().unwrap().len();

        HMMInstance::check_transition_matrix_validity(&transition_matrix, num_states)
        .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;

        self.transition_matrix = Some(transition_matrix);

        Ok(())
    }

    pub fn setup(
        &mut self,
        sequence_set: &Vec<Vec<f64>>,
        states: Vec<State>,
        start_matrix: StartMatrix,
        transition_matrix: TransitionMatrix,
    )  -> Result<(), HMMAnalyzerError> 
    {
        self.set_sequence_set(sequence_set)?;
        self.set_states(states)?;
        self.set_start_matrix(start_matrix)?;
        self.set_transition_matrix(transition_matrix)?;

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), HMMAnalyzerError> {
        // Check if everything is set
        if self.sequence_set.is_none(){return Err(HMMAnalyzerError::SequenceValuesNotDefined)}
        if self.states.is_none() {return Err(HMMAnalyzerError::StatesNotDefined)}
        if self.start_matrix.is_none() {return Err(HMMAnalyzerError::StartMatrixNotDefined)}
        if self.transition_matrix.is_none() {return Err(HMMAnalyzerError::TransitionMatrixNotDefined)}

        // if everything is set, get the references to the hmm parameters
        let sequence_set_ref = self.sequence_set.as_ref().unwrap();
        let states_ref = self.states.as_ref().unwrap();
        let start_matrix_ref = self.start_matrix.as_ref().unwrap();
        let transition_matrix_ref = self.transition_matrix.as_ref().unwrap();

        // Get viterbi predicted state sequence
        let viterbi_predictions = 
        Self::compute_viterbi_predictions(sequence_set_ref, states_ref, start_matrix_ref, transition_matrix_ref)?;
        
        
        self.sequence_states = Some(viterbi_predictions.clone());

        // Get state occupancy
        let state_occupancy = Self::compute_state_occupancy(&viterbi_predictions);
        self.state_occupancy = Some(state_occupancy);

        Ok(())
    }

    pub fn compute_viterbi_predictions
    (
        sequence_set: &Vec<Vec<f64>>,
        states: &[State],
        start_matrix: &StartMatrix,
        transition_matrix: &TransitionMatrix
    ) -> Result<Vec<Vec<usize>>, HMMAnalyzerError> {
        // Get the hmm_instance
        let mut hmm_instance = HMMInstance::new(states, start_matrix, transition_matrix);

        // Get container for predictions
        let mut predictions: Vec<Vec<usize>> = Vec::new();

        // Run the viterbi analysis
        for sequence_values in sequence_set {
            let _ = hmm_instance.run_viterbi(sequence_values)
            .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;
            let pred = hmm_instance.take_viterbi_prediction().ok_or(HMMAnalyzerError::CouldNotGetViterbiPred)?;

            predictions.push(pred);
        }
        

        Ok(predictions)
    }

    pub fn compute_state_occupancy(state_sequences: &Vec<Vec<usize>>) -> Vec<f64>{
        // Return empty vector if state sequence is empty
        if state_sequences.is_empty() {return Vec::new()}
        if state_sequences.iter().all(|sequence| sequence.is_empty()) { return Vec::new(); }

        let num_states = 
        state_sequences.iter().map(|sequence| sequence.iter().max().copied().unwrap()).max().unwrap() + 1; // Will never be none bc sequence cant be empty here
        
        let total_sequence_len: usize = state_sequences.iter().map(|sequence| sequence.len()).sum();   
        let mut state_occupancy = vec![0.0; num_states];
        
        for sequence in state_sequences {
            let current_sequence_len = sequence.len();
            let coefficient = current_sequence_len as f64 * total_sequence_len as f64;
            let mut state_dwell_count = vec![0_usize; num_states];
            sequence.iter().for_each(|state_id| state_dwell_count[*state_id] += 1);

            state_occupancy.iter_mut().zip(&state_dwell_count)
            .for_each(|(occ, count)| *occ = *occ + coefficient * *count as f64);
        }

        state_occupancy
    }

    pub fn get_states(&self) -> Option<&Vec<State>> {
        self.states.as_ref()
    }

    pub fn get_start_matrix(&self) -> Option<&StartMatrix> {
        self.start_matrix.as_ref()
    }

    pub fn get_transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix.as_ref()
    }

    pub fn get_sequences(&self) -> Option<&Vec<Vec<f64>>> {
        self.sequence_set.as_ref()
    }

    pub fn get_state_sequences(&self) -> Option<&Vec<Vec<usize>>> {
        self.sequence_states.as_ref()
    }

    pub fn get_state_occupancy(&self) -> Option<&Vec<f64>> {
        self.state_occupancy.as_ref()
    }
}

#[derive(Debug)]
pub enum HMMAnalyzerError {
    HMMInstanceError{err: HMMInstanceError},

    StatesNotDefined,
    StartMatrixNotDefined,
    TransitionMatrixNotDefined,
    SequenceValuesNotDefined,

    CouldNotGetViterbiPred,
}