use core::f64;

use crate::signal_analysis::hmm::{state, StateError};

use super::super::super::hmm::{State, StartMatrix, TransitionMatrix, hmm_instance::*};

use super::{eval_clusters::*, kmeans::k_means_1D};

pub struct HMMInitializer {
    state_values_method: StateValueInitMethod,
    state_noises_method: StateNoiseInitMethod,

    start_matrix_method: StartMatrixInitMethod,
    transition_matrix_method: TransitionMatrixInitMethod,
}

impl HMMInitializer {
    pub fn new() -> Self {
        HMMInitializer{
            state_values_method: StateValueInitMethod::KMeansClustering { max_iters: 100, tolerance: 1e-5, num_tries: 5, eval_method: ClusterEvaluationMethod::SimplifiedSilhouette },
            state_noises_method: StateNoiseInitMethod::Sparse{mult: 0.5},
            start_matrix_method: StartMatrixInitMethod::Balanced,
            transition_matrix_method: TransitionMatrixInitMethod::Balanced,
        }
    }

    pub fn set_state_values_method(&mut self, method: StateValueInitMethod) {
        self.state_values_method = method;
    }

    pub fn set_state_noises_method(&mut self, method: StateNoiseInitMethod) {
        self.state_noises_method = method;
    }

    pub fn set_start_matrix_method(&mut self, method: StartMatrixInitMethod) {
        self.start_matrix_method = method;
    }

    pub fn set_transition_matrix_method(&mut self, method:TransitionMatrixInitMethod) {
        self.transition_matrix_method = method;
    }

    pub fn get_initial_state_values(&self, num_states: usize, sequence_values: &[f64]) -> Result<Vec<f64>, HMMInitializerError> {
        let state_values = self.state_values_method.get_state_values(num_states, sequence_values)?;
        Ok(state_values)
    }

    pub fn get_initial_state_noises(&self, num_states: usize, sequence_values: &[f64]) -> Result<Vec<f64>, HMMInitializerError> {
        let state_noises = self.state_noises_method.get_state_noises(num_states, sequence_values)?;
        Ok(state_noises)
    }

    pub fn get_initial_start_matrix(&self, num_states: usize, sequence_values: &[f64]) -> Result<StartMatrix, HMMInitializerError> {
        let start_matrix = self.start_matrix_method.get_start_matrix(num_states)?;
        Ok(start_matrix)
    }

    pub fn get_initial_transition_matrix(&self, num_states: usize, sequence_values: &[f64]) -> Result<TransitionMatrix, HMMInitializerError> {
        let transition_matrix = self.transition_matrix_method.get_transition_matrix(num_states)?;
        Ok(transition_matrix)
    }
}

fn check_validity(num_states: usize, sequence_values:&[f64]) -> Result<(), HMMInitializerError> {
    if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}

    HMMInstance::check_sequence_validity(sequence_values, VALUE_SEQUENCE_THRESHOLD)
    .map_err(|err| HMMInitializerError::HMMInstanceError { err })?;

    Ok(())
}
pub enum StateValueInitMethod {
    KMeansClustering{max_iters: usize, tolerance: f64, num_tries: usize, eval_method: ClusterEvaluationMethod},
    Random,
    Sparse,
    // Add others when they are implemented
}

impl StateValueInitMethod {
    pub fn get_state_values(&self, num_states: usize, sequence_values: &[f64]) -> Result<Vec<f64>, HMMInitializerError>{
        check_validity(num_states, sequence_values)?;

        let min_value= sequence_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_value= sequence_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();        


        match self {
            StateValueInitMethod::KMeansClustering { max_iters, tolerance, num_tries, eval_method } => {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_state_values = vec![0.0; num_states];
                

                for _ in 0..*num_tries {
                    let (state_values, assignments) = k_means_1D(sequence_values, num_states, *max_iters, *tolerance);
                    let score: f64;

                    match eval_method {
                        ClusterEvaluationMethod::Silhouette => {
                            score = silhouette_score_1D(&state_values, &assignments, sequence_values);
                        }

                        ClusterEvaluationMethod::SimplifiedSilhouette => {
                            score = simplified_silhouette_score_1D(&state_values, &assignments, sequence_values);
                        }
                    }
                    // println!("Before. Num states: {}. Score: {}", num_states, score);
                    if score > best_score {
                        println!("Was here");
                        best_score = score;
                        best_state_values = state_values;
                    }

                }
                
                if best_score == f64::NEG_INFINITY {
                    return Err(HMMInitializerError::EmptyInitializationCluster)
                }
                return Ok(best_state_values);
            }

            StateValueInitMethod::Sparse => {
                // Get full states but we only care about values here
                let states = HMMInstance::generate_sparse_states(num_states as u16, *max_value, *min_value, 1.0)
                .map_err(|err| HMMInitializerError::HMMInstanceError { err })?;

                let state_values: Vec<f64> = states.iter().map(|state| state.get_value()).collect();

                return Ok(state_values);
            }

            StateValueInitMethod::Random => {
                // Get full states but we only care about values here
                let states = HMMInstance::generate_random_states(num_states as u16, Some(*max_value), Some(*min_value), None, None)
                .map_err(|err| HMMInitializerError::StateError { err })?;

                let state_values: Vec<f64> = states.iter().map(|state| state.get_value()).collect();

                return Ok(state_values);
            }
        }

        Ok(vec![0.0])
    }
}

pub enum StateNoiseInitMethod {
    Sparse{mult: f64},
}

impl StateNoiseInitMethod {
    pub fn get_state_noises(&self, num_states: usize, sequence_values: &[f64]) -> Result<Vec<f64>, HMMInitializerError> {
        check_validity(num_states, sequence_values)?;

        let min_value= sequence_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_value= sequence_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        match self {
            StateNoiseInitMethod::Sparse { mult } => {
                let range = max_value - min_value;
                let noise = range / (num_states as f64) * mult; 

                return Ok(vec![noise; num_states]);
            }
        }
    }
}

pub enum StartMatrixInitMethod {
    Balanced,
    Random,
    // Add others when they are implemented
}

impl StartMatrixInitMethod {
    pub fn get_start_matrix(&self, num_states: usize) -> Result<StartMatrix, HMMInitializerError> {
        if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}
        match self {
            StartMatrixInitMethod::Random => {
                let start_matrix = StartMatrix::new_random(num_states);
                return Ok(start_matrix);
            }

            StartMatrixInitMethod::Balanced => {
                let start_matrix = StartMatrix::new_balanced(num_states);
                return Ok(start_matrix);
            }
        }
    }
}

pub enum TransitionMatrixInitMethod {
    Balanced,
    Random,
    // Add others when they are implemented
}

impl TransitionMatrixInitMethod {
    pub fn get_transition_matrix(&self, num_states: usize) -> Result<TransitionMatrix, HMMInitializerError> {
        if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}
        match self {
            TransitionMatrixInitMethod::Random => {
                let transition_matrix = TransitionMatrix::new_random(num_states);
                return Ok(transition_matrix);
            }

            TransitionMatrixInitMethod::Balanced => {
                let transition_matrix = TransitionMatrix::new_balanced(num_states);
                return Ok(transition_matrix);
            }
        }
    }
}

pub enum NumberOfStatesMethod {
    Manual{num: usize},
    ClustersBayesInformationCriterion,
    BaumWelchBayesInformationCriterion,
}

#[derive(Debug)]
pub enum HMMInitializerError {
    EmptyObservationSequence,
    InvalidObservatonSequence,
    InvalidNumberOfStates,
    HMMInstanceError{err: HMMInstanceError},
    StateError{err: StateError},
    EmptyInitializationCluster,
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_set_methods() {
        let mut initializer = HMMInitializer::new();

        initializer.set_state_values_method(StateValueInitMethod::Random);
        assert!(matches!(initializer.state_values_method, StateValueInitMethod::Random));

        initializer.set_state_noises_method(StateNoiseInitMethod::Sparse { mult: 1.0 });
        assert!(matches!(
            initializer.state_noises_method,
            StateNoiseInitMethod::Sparse { mult: 1.0 }
        ));

        initializer.set_start_matrix_method(StartMatrixInitMethod::Random);
        assert!(matches!(initializer.start_matrix_method, StartMatrixInitMethod::Random));

        initializer.set_transition_matrix_method(TransitionMatrixInitMethod::Random);
        assert!(matches!(
            initializer.transition_matrix_method,
            TransitionMatrixInitMethod::Random
        ));
    }

    #[test]
    fn test_get_initial_state_values() {
        let initializer = HMMInitializer::new();
        let sequence = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let state_values = initializer.get_initial_state_values(3, &sequence);
        assert!(state_values.is_ok());
        assert_eq!(state_values.unwrap().len(), 3);
    }

    #[test]
    fn test_get_initial_state_noises() {
        let initializer = HMMInitializer::new();
        let sequence = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let state_noises = initializer.get_initial_state_noises(3, &sequence);
        assert!(state_noises.is_ok());
        assert_eq!(state_noises.unwrap().len(), 3);
    }

    #[test]
    fn test_get_initial_start_matrix() {
        let initializer = HMMInitializer::new();
        let start_matrix = initializer.get_initial_start_matrix(3, &[]);
        assert!(start_matrix.is_ok());
        assert_eq!(start_matrix.unwrap().matrix.len(), 3);
    }

    #[test]
    fn test_get_initial_transition_matrix() {
        let initializer = HMMInitializer::new();
        let transition_matrix = initializer.get_initial_transition_matrix(3, &[]);
        assert!(transition_matrix.is_ok());
        assert_eq!(transition_matrix.unwrap().matrix.len(), 3);
    }

    #[test]
    fn test_check_validity_empty_sequence() {
        let result = check_validity(3, &[]);
        assert!(matches!(result, Err(HMMInitializerError::EmptyObservationSequence)));
    }

    #[test]
    fn test_check_validity_invalid_num_states() {
        let result = check_validity(0, &[1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(HMMInitializerError::InvalidNumberOfStates)));
    }

    #[test]
    fn test_check_validity_invalid_observation_sequence() {
        let result = check_validity(3, &[1.0, 1.0, 1.0]);
        assert!(matches!(result, Err(HMMInitializerError::InvalidObservatonSequence)));
    }
}