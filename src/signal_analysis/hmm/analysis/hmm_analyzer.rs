use crate::signal_analysis::hmm::{hmm_instance::{HMMInstance, HMMInstanceError, VALUE_SEQUENCE_THRESHOLD}, StartMatrix, State, TransitionMatrix};

#[derive(Debug)]
pub struct HMMAnalyzer {
    sequence_values: Option<Vec<f64>>,
    sequence_states: Option<Vec<usize>>,
    states: Option<Vec<State>>,
    start_matrix: Option<StartMatrix>,
    transition_matrix: Option<TransitionMatrix>,
}

impl HMMAnalyzer {
    pub fn new() -> Self {
        Self {
            sequence_values: None,
            sequence_states: None,
            states: None,
            start_matrix: None,
            transition_matrix: None,
        }
    }

    pub fn set_sequence_values(&mut self, sequence_values: &[f64]) -> Result<(), HMMAnalyzerError> {
        HMMInstance::check_sequence_validity(sequence_values, VALUE_SEQUENCE_THRESHOLD)
        .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;
        self.sequence_values = Some(sequence_values.to_vec());

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
        sequence_values: &[f64],
        states: Vec<State>,
        start_matrix: StartMatrix,
        transition_matrix: TransitionMatrix,
    )  -> Result<(), HMMAnalyzerError> 
    {
        self.set_sequence_values(sequence_values)?;
        self.set_states(states)?;
        self.set_start_matrix(start_matrix)?;
        self.set_transition_matrix(transition_matrix)?;

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), HMMAnalyzerError> {
        // Check if everything is set
        if self.sequence_values.is_none(){return Err(HMMAnalyzerError::SequenceValuesNotDefined)}
        if self.states.is_none() {return Err(HMMAnalyzerError::StatesNotDefined)}
        if self.start_matrix.is_none() {return Err(HMMAnalyzerError::StartMatrixNotDefined)}
        if self.transition_matrix.is_none() {return Err(HMMAnalyzerError::TransitionMatrixNotDefined)}

        // if everything is set, get the references to the hmm parameters
        let states_ref = self.states.as_ref().unwrap();
        let start_matrix_ref = self.start_matrix.as_ref().unwrap();
        let transition_matrix_ref = self.transition_matrix.as_ref().unwrap();

        // Get the hmm_instance
        let mut hmm_instance = HMMInstance::new(states_ref, start_matrix_ref, transition_matrix_ref);

        // Run the viterbi analysis
        let sequence_values = self.sequence_values.as_ref().unwrap();
        let _ = hmm_instance.run_viterbi(sequence_values)
        .map_err(|err| HMMAnalyzerError::HMMInstanceError { err })?;
        self.sequence_states = hmm_instance.take_viterbi_prediction();

        Ok(())
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

    pub fn get_value_sequence(&self) -> Option<&Vec<f64>> {
        self.sequence_values.as_ref()
    }

    pub fn get_states_sequence(&self) -> Option<&Vec<usize>> {
        self.sequence_states.as_ref()
    }
}

#[derive(Debug)]
pub enum HMMAnalyzerError {
    HMMInstanceError{err: HMMInstanceError},

    StatesNotDefined,
    StartMatrixNotDefined,
    TransitionMatrixNotDefined,
    SequenceValuesNotDefined,
}