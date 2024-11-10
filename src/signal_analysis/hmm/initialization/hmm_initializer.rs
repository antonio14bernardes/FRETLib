use core::f64;

use nalgebra::{DMatrix, DVector};

use crate::signal_analysis::hmm::amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness};
use crate::signal_analysis::hmm::learning::learner_trait::{LearnerSpecificInitialValues, LearnerSpecificSetup};
use crate::signal_analysis::hmm::{StateError, StartMatrix, TransitionMatrix, hmm_instance::*};

use super::{eval_clusters::*, kmeans::*};

const STATE_NOISE_MULT: f64 = 0.5;
const STATE_NOISE_STD_MULT: f64 = 0.25;
const START_PROB_STD: f64 = 0.25;
const TRANSITION_PROB_STD: f64 = 0.25;


pub struct InitializationMethods {
    pub state_values_method: Option<StateValueInitMethod>,
    pub state_noises_method: Option<StateNoiseInitMethod>,
    pub start_matrix_method: Option<StartMatrixInitMethod>,
    pub transition_matrix_method: Option<TransitionMatrixInitMethod>,
}

impl InitializationMethods {
    pub fn new() -> Self {
        Self {
            state_values_method: Some(StateValueInitMethod::default()),
            state_noises_method: Some(StateNoiseInitMethod::default()),
            start_matrix_method: Some(StartMatrixInitMethod::default()),
            transition_matrix_method: Some(TransitionMatrixInitMethod::default())
        }
    }
}
#[derive(Debug, Clone)]
pub struct HMMInitializer {
    state_values_method: StateValueInitMethod,
    state_noises_method: StateNoiseInitMethod,

    start_matrix_method: StartMatrixInitMethod,
    transition_matrix_method: TransitionMatrixInitMethod,

    learner_setup: LearnerSpecificSetup,
}

impl HMMInitializer {
    pub fn new(learner_setup: &LearnerSpecificSetup) -> Self {
        HMMInitializer{
            state_values_method: StateValueInitMethod::KMeansClustering 
            { max_iters: None, tolerance: None, num_tries: None, eval_method: None },
            state_noises_method: StateNoiseInitMethod::Sparse{mult: None},
            start_matrix_method: StartMatrixInitMethod::Balanced,
            transition_matrix_method: TransitionMatrixInitMethod::Balanced,
            learner_setup: learner_setup.clone(),
        }
    }

    pub fn setup_initializer(&mut self, init_setup: InitializationMethods) -> Result<(), HMMInitializerError> {
        if let Some(method) = init_setup.state_values_method {
            self.set_state_values_method(method)?;
        }
        if let Some(method) = init_setup.state_noises_method {
            self.set_state_noises_method(method)?;
        }
        if let Some(method) = init_setup.start_matrix_method {
            self.set_start_matrix_method(method);
        }
        if let Some(method) = init_setup.transition_matrix_method {
            self.set_transition_matrix_method(method);
        }

        Ok(())
    }

    pub fn set_state_values_method(&mut self, method: StateValueInitMethod) -> Result<(), HMMInitializerError> {
        if let StateValueInitMethod::KMeansClustering { max_iters, num_tries, .. } = method {
            let max_iters_invalid: bool = max_iters.map(|value| value == 0).unwrap_or(false);
            let num_tries_invalid: bool = num_tries.map(|value| value == 0).unwrap_or(false);

            if num_tries_invalid || max_iters_invalid {return Err(HMMInitializerError::InvalidInput)}
        }
        
        self.state_values_method = StateValueInitMethod::handle_nones(method);

        Ok(())
    }

    pub fn set_state_noises_method(&mut self, method: StateNoiseInitMethod) -> Result<(), HMMInitializerError>{
        if let StateNoiseInitMethod::Sparse { mult } = method {
            if mult.map(|value| value <= 0.0).unwrap_or(false) {
                return Err(HMMInitializerError::InvalidInput);
            }
        }
        self.state_noises_method = StateNoiseInitMethod::handle_nones(method);

        Ok(())
    }

    pub fn set_start_matrix_method(&mut self, method: StartMatrixInitMethod) {
        self.start_matrix_method = method;
    }

    pub fn set_transition_matrix_method(&mut self, method:TransitionMatrixInitMethod) {
        self.transition_matrix_method = method;
    }

    pub fn get_intial_values(&self, sequence_set: &Vec<Vec<f64>>, num_states: usize) -> Result<LearnerSpecificInitialValues, HMMInitializerError> {
        let model_dist = self.learner_setup.model_distribution();
        let model_matrix_dist = self.learner_setup.model_matrix_distribution();

        // Get state values.
        let (state_values, state_noise_from_values) = self.state_values_method.get_state_values(num_states, sequence_set, model_dist)?;
        
        
        // If the state values also provides noise, use this instead of the one obtained form the StateNoiseInitMethod struct
        let state_noise: Vec<f64>;
        let mut state_noise_std: Option<Vec<f64>> = None;

        if state_noise_from_values.is_some() {
            state_noise = state_noise_from_values.unwrap();
            if !model_dist {
                // state_noise_std is already None
            } else if state_noise.len() > 1 { // If there is more than 1 state, compute std of noises to model the noise distribution
                let mean_noises: f64 = state_noise.iter().sum::<f64>() / state_noise.len() as f64;
                
                let variance_noises: f64 = 
                state_noise.iter()
                .map(|value| (value - mean_noises).powi(2)).sum::<f64>() / state_noise.len() as f64;
                state_noise_std = Some(vec![variance_noises.sqrt(); num_states]);

            } else if state_noise.len() == 1 {
                state_noise_std = Some(vec![state_noise[0] * STATE_NOISE_STD_MULT; num_states]);
            }
        
        // If the state_value method didnt get us a noise, we use the normal noise init method thing
        } else {
            let (state_noise_from_noises, state_noise_std_from_noises) =
            self.state_noises_method.get_state_noises(num_states, sequence_set, model_dist)?;

            state_noise = state_noise_from_noises;

            if let Some(state_noise_std_value) = state_noise_std_from_noises {
                state_noise_std = Some(vec![state_noise_std_value; num_states]);
            } else {
                // state_noise_std is already None
            }
        }

        // Now get the matrices
        let (start_matrix, start_matrix_cov_diag) = self.start_matrix_method.get_start_matrix(num_states, model_matrix_dist)?;
        let (transition_matrix, transition_matrix_cov_diags) = self.transition_matrix_method.get_transition_matrix(num_states, model_matrix_dist)?;


        // Get the enum
        let initial_values =
        build_learner_setup_enum(&self.learner_setup, state_values, state_noise, state_noise_std, 
            start_matrix, start_matrix_cov_diag, transition_matrix, transition_matrix_cov_diags);

        Ok(initial_values)

    }
}

fn check_validity(num_states: usize, sequence_values:&[f64]) -> Result<(), HMMInitializerError> {
    if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}

    HMMInstance::check_sequence_validity(sequence_values, VALUE_SEQUENCE_THRESHOLD)
    .map_err(|err| HMMInitializerError::HMMInstanceError { err })?;

    Ok(())
}

fn vec_to_means(vec: Vec<f64>) -> DVector<f64> {
    DVector::from_vec(vec)
}

fn vec_to_cov_diag(vec: Vec<f64>) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec))
}

// Convert a StartMatrix to a DVector of means
fn start_matrix_to_means(start_matrix: &StartMatrix) -> DVector<f64> {
    DVector::from_vec(start_matrix.matrix.clone())
}

// Convert a TransitionMatrix to a Vec<DVector<f64>> of means
fn transition_matrix_to_means(transition_matrix: &TransitionMatrix) -> Vec<DVector<f64>> {
    transition_matrix.matrix.iter()
        .map(|row| DVector::from_vec(row.clone()))
        .collect()
}

fn diagonal_to_cov_matrix(diagonals: Vec<Vec<f64>>) -> Vec<DMatrix<f64>> {
    diagonals
        .into_iter()
        .map(|diag| {
            // Create a DVector from the diagonal values
            let diag_vector = DVector::from_vec(diag);
            // Create a DMatrix with the diagonal values set and zeros elsewhere
            DMatrix::from_diagonal(&diag_vector)
        })
        .collect()
}

fn state_and_noise_dists(
    learner_setup: &LearnerSpecificSetup,
    state_values: Vec<f64>,
    state_noise: Vec<f64>,
    state_noise_std: Option<Vec<f64>>,
) -> Option<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)> {

    let learner_setup = LearnerSpecificSetup::handle_nones(learner_setup.clone());

    match learner_setup {
        LearnerSpecificSetup::AmalgamIdea { dependence_type, .. } => {
            let mut means: Vec<DVector<f64>> = Vec::new();
            let mut covs: Vec<DMatrix<f64>> = Vec::new();

            // Check dependency type to see how state values and noises will be set
            match dependence_type.as_ref().unwrap() {
                AmalgamDependencies::AllIndependent => {
                    // First set the states
                    state_values.iter()
                    .for_each(|value| means.push(vec_to_means(vec![*value])));

                    state_noise.iter()
                    .for_each(|std| covs.push(vec_to_cov_diag(vec![*std])));

                    // Then set the noises
                    state_noise.iter()
                    .for_each(|noise| means.push(vec_to_means(vec![*noise])));

                    state_noise_std.unwrap().iter()
                    .for_each(|std| covs.push(vec_to_cov_diag(vec![*std])));
                }  

                AmalgamDependencies::ValuesDependent => {
                    // Set the values
                    means.push(vec_to_means(state_values));
                    covs.push(vec_to_cov_diag(state_noise.clone()));

                    // Set the noise
                    means.push(vec_to_means(state_noise));
                    covs.push(vec_to_cov_diag(state_noise_std.unwrap()));
                }

                AmalgamDependencies::StateCompact => {
                    // Set the states in order
                    for (value, noise) in state_values.iter().zip(&state_noise) {
                        means.push(vec_to_means(vec![*value, *noise]));
                    }
                    for (value_std, noise_std) in state_noise.iter().zip(&state_noise_std.unwrap()) {
                        covs.push(vec_to_cov_diag(vec![*value_std, *noise_std]));
                    }
                }
            }
            return Some((means, covs));
        }
        LearnerSpecificSetup::BaumWelch { .. } => {
            // Only need to set the initial values so this function is not used for BaumWelch
            return None;
        }
    }
}

fn start_and_transition_matrices_dists(
    learner_setup: &LearnerSpecificSetup,
    start_matrix: StartMatrix,
    start_matrix_cov_diag: Option<Vec<f64>>,
    transition_matrix: TransitionMatrix,
    transition_matrix_cov_diags: Option<Vec<Vec<f64>>>,
) -> Option<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)> {

    let learner_setup = LearnerSpecificSetup::handle_nones(learner_setup.clone());

    match learner_setup {
        LearnerSpecificSetup::AmalgamIdea { fitness_type, .. } => {
            let mut means: Vec<DVector<f64>> = Vec::new();
            let mut covs: Vec<DMatrix<f64>> = Vec::new();

            // Check dependency type to see how state values and noises will be set
            match fitness_type.as_ref().unwrap() {
                AmalgamFitness::Direct => {
                    // Set the start matrix means and covs
                    means.push(start_matrix_to_means(&start_matrix));
                    covs.push(vec_to_cov_diag(start_matrix_cov_diag.unwrap()));

                    // Set the transition matrix means and covs
                    means.extend(transition_matrix_to_means(&transition_matrix));
                    covs.extend(diagonal_to_cov_matrix(transition_matrix_cov_diags.unwrap()));
                }

                AmalgamFitness::BaumWelch => {} // No need to add matrix distributions in this case
            }

            return Some((means, covs));
        }
        LearnerSpecificSetup::BaumWelch { .. } => {
            // Only need to set the initial values so this function is not used for BaumWelch
            return None;
        }
    }
}

fn build_learner_setup_enum(
    learner_setup: &LearnerSpecificSetup,
    state_values: Vec<f64>,
    state_noise: Vec<f64>,
    state_noise_std: Option<Vec<f64>>,
    start_matrix: StartMatrix,
    start_matrix_cov_diag: Option<Vec<f64>>,
    transition_matrix: TransitionMatrix,
    transition_matrix_cov_diags: Option<Vec<Vec<f64>>>
) -> LearnerSpecificInitialValues {

    let learner_initial_values: LearnerSpecificInitialValues;

    match learner_setup {
        LearnerSpecificSetup::AmalgamIdea { .. } => {
            let mut means: Vec<DVector<f64>> = Vec::new();
            let mut covs: Vec<DMatrix<f64>> = Vec::new();


            // Add state values and noise dists to the vectors
            let (state_means, state_covs) = 
            state_and_noise_dists(learner_setup, state_values, state_noise, state_noise_std).unwrap();

            means.extend(state_means);
            covs.extend(state_covs);

            // Add start and transition matrices dists to the vectors
            let (matrices_means, matrices_covs) = 
            start_and_transition_matrices_dists(learner_setup, start_matrix, start_matrix_cov_diag, 
                                                transition_matrix, transition_matrix_cov_diags).unwrap();

            means.extend(matrices_means);
            covs.extend(matrices_covs);

            
            // Setup the enum
            learner_initial_values = LearnerSpecificInitialValues::AmalgamIdea { initial_distributions: Some((means, covs)) }
        }

        LearnerSpecificSetup::BaumWelch { .. } => {

            let states = HMMInstance::generate_state_set(&state_values, &state_noise).unwrap();

            learner_initial_values = LearnerSpecificInitialValues::BaumWelch { states, start_matrix: Some(start_matrix), transition_matrix: Some(transition_matrix) }
        }
    }

    learner_initial_values
}

#[derive(Debug, Clone)]
pub enum StateValueInitMethod {
    KMeansClustering{max_iters: Option<usize>, tolerance: Option<f64>, num_tries: Option<usize>, eval_method: Option<ClusterEvaluationMethod>},
    Random,
    Sparse,
    // Add others when they are implemented
}

impl StateValueInitMethod {
    pub fn default() -> Self {
        StateValueInitMethod::KMeansClustering { 
            max_iters: Some(KMEANS_MAX_ITERS_DEFAULT),
            tolerance: Some(KMEANS_TOLERANCE_DEFAULT),
            num_tries: Some(KMEANS_NUM_TRIES_DEFAULT),
            eval_method: Some(KMEANS_EVAL_METHOD_DEFAULT) 
        }
    }

    pub fn handle_nones(self) -> Self {
        match self {
            Self::KMeansClustering { max_iters, tolerance, num_tries, eval_method } => {
                let max_iters_new = max_iters.unwrap_or(KMEANS_MAX_ITERS_DEFAULT);
                let tolerance_new = tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT);
                let num_tries_new = num_tries.unwrap_or(KMEANS_NUM_TRIES_DEFAULT);
                let eval_method_new = eval_method.unwrap_or(KMEANS_EVAL_METHOD_DEFAULT);

                Self::KMeansClustering { 
                    max_iters: Some(max_iters_new),
                    tolerance: Some(tolerance_new),
                    num_tries: Some(num_tries_new),
                    eval_method: Some(eval_method_new) }
            }
            Self::Random => {self}
            Self::Sparse => {self}
        }
    }
    pub fn get_state_values(&self, num_states: usize, sequence_set: &Vec<Vec<f64>>, dist: bool) -> Result<(Vec<f64>, Option<Vec<f64>>), HMMInitializerError>{
        if sequence_set.is_empty() {return Err(HMMInitializerError::EmptyObservationSequence)}
        for sequence_values in sequence_set {check_validity(num_states, sequence_values)?}

        let min_value= sequence_set.iter()
        .map(|sequence| sequence.iter().min_by(|a, b| a.total_cmp(b)).unwrap())
        .min_by(|a, b| a.total_cmp(b)).unwrap();

        let max_value = sequence_set.iter()
        .map(|sequence| sequence.iter().max_by(|a, b| a.total_cmp(b)).unwrap())
        .max_by(|a, b| a.total_cmp(b)).unwrap();      

        // Flatten the set of sequences to one single sequence
        let total_sequence_values = sequence_set.concat();

        match self {
            StateValueInitMethod::KMeansClustering { max_iters, tolerance, num_tries, eval_method } => {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_state_values = vec![0.0; num_states];
                let mut best_assignments = vec![0_usize; total_sequence_values.len()];

                let max_iters = max_iters.unwrap_or(KMEANS_MAX_ITERS_DEFAULT);
                let tolerance = tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT);
                let num_tries = num_tries.unwrap_or(KMEANS_NUM_TRIES_DEFAULT);
                let eval_method = eval_method.as_ref().unwrap_or(&KMEANS_EVAL_METHOD_DEFAULT).clone();
                

                for _ in 0..num_tries {
                    let (state_values, assignments) = k_means_1_d(&total_sequence_values, num_states, max_iters, tolerance);
                    let score: f64;

                    match eval_method {
                        ClusterEvaluationMethod::Silhouette => {
                            score = silhouette_score_1_d(&state_values, &assignments, &total_sequence_values);
                        }

                        ClusterEvaluationMethod::SimplifiedSilhouette => {
                            score = simplified_silhouette_score_1_d(&state_values, &assignments, &total_sequence_values);
                        }
                    }
                    // println!("Before. Num states: {}. Score: {}", num_states, score);
                    if score > best_score {
                        best_score = score;
                        best_state_values = state_values;
                        best_assignments = assignments;
                    }

                }
                
                if best_score == f64::NEG_INFINITY {
                    return Err(HMMInitializerError::EmptyInitializationCluster)
                }

                if dist {
                    // Compute the std per cluster
                    let num_clusters = num_states;
                    let mut cluster_stds = vec![0.0; num_clusters];
                    let mut count_per_cluster = vec![0; num_states];
                    for assignment in best_assignments.iter() {
                        count_per_cluster[*assignment] += 1;
                    }
                    for (value, &assignment) in total_sequence_values.iter().zip(&best_assignments) {
                        cluster_stds[assignment] += (value - best_state_values[assignment]).powi(2);
                    }
                    for i in 0..cluster_stds.len() {
                        cluster_stds[i] = (cluster_stds[i] / count_per_cluster[i] as f64).sqrt();
                    }

                    return Ok((best_state_values, Some(cluster_stds)))

                }


                return Ok((best_state_values, None));
            }

            StateValueInitMethod::Sparse => {
                // Get full states but we only care about values here
                let states = HMMInstance::generate_sparse_states(num_states as u16, *max_value, *min_value, 1.0)
                .map_err(|err| HMMInitializerError::HMMInstanceError { err })?;

                let state_values: Vec<f64> = states.iter().map(|state| state.get_value()).collect();

                return Ok((state_values, None));
            }

            StateValueInitMethod::Random => {
                // Get full states but we only care about values here
                let states = HMMInstance::generate_random_states(num_states as u16, Some(*max_value), Some(*min_value), None, None)
                .map_err(|err| HMMInitializerError::StateError { err })?;

                let state_values: Vec<f64> = states.iter().map(|state| state.get_value()).collect();

                return Ok((state_values, None));
            }
        }

    }
}

#[derive(Debug, Clone)]
pub enum StateNoiseInitMethod {
    Sparse{mult: Option<f64>},
}

impl StateNoiseInitMethod {
    pub fn default() -> Self {
        Self::Sparse { mult: Some(STATE_NOISE_MULT) }
    }

    pub fn handle_nones(self) -> Self {
        match self {
            Self::Sparse { mult } => {
                Self::Sparse { mult: Some(mult.unwrap_or(STATE_NOISE_MULT)) }
            }
        }
    }

    pub fn get_state_noises(&self, num_states: usize, sequence_set: &Vec<Vec<f64>>, dist: bool) -> Result<(Vec<f64>, Option<f64>), HMMInitializerError> {
        if sequence_set.is_empty() {return Err(HMMInitializerError::EmptyObservationSequence)}
        for sequence_values in sequence_set {check_validity(num_states, sequence_values)?}

        let min_value= sequence_set.iter()
        .map(|sequence| sequence.iter().min_by(|a, b| a.total_cmp(b)).unwrap())
        .min_by(|a, b| a.total_cmp(b)).unwrap();

        let max_value = sequence_set.iter()
        .map(|sequence| sequence.iter().max_by(|a, b| a.total_cmp(b)).unwrap())
        .max_by(|a, b| a.total_cmp(b)).unwrap(); 

        let noise: f64;
        let noise_std: Option<f64>;
        match self {
            StateNoiseInitMethod::Sparse { mult } => {
                let mult = mult.unwrap_or(STATE_NOISE_MULT);
                let range = max_value - min_value;
                noise = range / (num_states as f64) * mult; 

                noise_std = if dist {Some(noise * STATE_NOISE_STD_MULT)} else {None}
                
            }
        }
        return Ok((vec![noise; num_states], noise_std));
        
    }
}

#[derive(Debug, Clone)]
pub enum StartMatrixInitMethod {
    Balanced,
    Random,
    // Add others when they are implemented
}

impl StartMatrixInitMethod {
    pub fn default() -> Self  {
        StartMatrixInitMethod::Balanced{}
    }
    pub fn get_start_matrix(&self, num_states: usize, dist: bool) -> Result<(StartMatrix, Option<Vec<f64>>), HMMInitializerError> {
        if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}
        let start_matrix: StartMatrix;
        match self {
            StartMatrixInitMethod::Random => {
                start_matrix = StartMatrix::new_random(num_states);
                
            }

            StartMatrixInitMethod::Balanced => {
                start_matrix = StartMatrix::new_balanced(num_states);
            }
        }

        let diag_vector= 
        if dist {
            Some(vec![START_PROB_STD; num_states])
        } else {
            None
        };
        return Ok((start_matrix, diag_vector));
    }
}

#[derive(Debug, Clone)]
pub enum TransitionMatrixInitMethod {
    Balanced,
    Random,
    // Add others when they are implemented
}

impl TransitionMatrixInitMethod {
    pub fn default() -> Self {
        TransitionMatrixInitMethod::Balanced
    }
    pub fn get_transition_matrix(&self, num_states: usize, dist: bool) -> Result<(TransitionMatrix,Option<Vec<Vec<f64>>>), HMMInitializerError> {
        if num_states == 0 {return Err(HMMInitializerError::InvalidNumberOfStates)}
        let transition_matrix: TransitionMatrix;
        match self {
            TransitionMatrixInitMethod::Random => {
                transition_matrix = TransitionMatrix::new_random(num_states);
            }

            TransitionMatrixInitMethod::Balanced => {
                transition_matrix = TransitionMatrix::new_balanced(num_states);
            }
        }

        let diag_vecs = 
        if dist {
            Some(vec![vec![TRANSITION_PROB_STD; num_states]; num_states])
        } else {
            None
        };

        Ok((transition_matrix, diag_vecs))
    }
}

#[derive(Debug, Clone)]
pub enum HMMInitializerError {
    EmptyObservationSequence,
    InvalidObservatonSequence,
    InvalidNumberOfStates,
    HMMInstanceError{err: HMMInstanceError},
    StateError{err: StateError},
    EmptyInitializationCluster,
    InvalidInput,
}

#[cfg(test)]
mod hmm_initializer_tests {
    use crate::signal_analysis::hmm::State;

    use super::*;

    fn create_test_sequence_set() -> (Vec<State>, StartMatrix, TransitionMatrix, Vec<Vec<f64>>) {
        // Define real states
        let real_state1 = State::new(0, 10.0, 1.0).unwrap();
        let real_state2 = State::new(1, 20.0, 2.0).unwrap();
        let real_state3 = State::new(2, 30.0, 3.0).unwrap();
        let real_states = vec![real_state1, real_state2, real_state3];

        // Define start matrix
        let real_start_matrix = StartMatrix::new(vec![0.1, 0.8, 0.1]);

        // Define transition matrix
        let real_transition_matrix = TransitionMatrix::new(vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.2, 0.05, 0.75],
        ]);

        // Generate multiple sequences as the sequence set
        let sequence_set = (0..5)
            .map(|_| {
                let (_, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 200);
                sequence_values
            })
            .collect();

        (real_states, real_start_matrix, real_transition_matrix, sequence_set)
    }
    
    #[test]
    fn test_amalgam_idea_initialization() {
        // Generate test sequence
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Array of all AmalgamDependencies types
        let dependencies_types = [
            AmalgamDependencies::AllIndependent,
            AmalgamDependencies::StateCompact,
            AmalgamDependencies::ValuesDependent,
        ];

        for dependence_type in dependencies_types.iter() {
            // Define learner setup for each AmalgamDependency type
            let learner_setup = LearnerSpecificSetup::AmalgamIdea {
                iter_memory: Some(true),
                dependence_type: Some(dependence_type.clone()),
                fitness_type: Some(AmalgamFitness::Direct),
                max_iterations: Some(1000),
            };

            // Initialize HMMInitializer with each dependency type
            let initializer = HMMInitializer::new(&learner_setup);

            // Obtain initial values
            let initial_values = initializer.get_intial_values(&sequence_set, num_states).expect("Failed to get initial values");

            // Check that output is of type LearnerSpecificInitialValues::AmalgamIdea
            if let LearnerSpecificInitialValues::AmalgamIdea { initial_distributions: Some((means, covs)) } = initial_values {
                // Set expected means and covs count based on dependence type
                let expected_counts = match dependence_type {
                    AmalgamDependencies::AllIndependent => num_states * 2 + 1 + num_states,
                    AmalgamDependencies::StateCompact => num_states + 1 + num_states,
                    AmalgamDependencies::ValuesDependent => 2 + 1 + num_states,
                };

                // Verify that means and covs match expected counts
                assert_eq!(means.len(), expected_counts, "Unexpected number of mean vectors for {:?}", dependence_type);
                assert_eq!(covs.len(), expected_counts, "Unexpected number of covariance matrices for {:?}", dependence_type);

            } else {
                panic!("Expected LearnerSpecificInitialValues::AmalgamIdea with initial distributions for {:?}", dependence_type);
            }
        }
    }

    #[test]
    fn test_amalgam_idea_baumwelch_initialization() {
        // Generate test sequence
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Array of all AmalgamDependencies types
        let dependencies_types = [
            AmalgamDependencies::AllIndependent,
            AmalgamDependencies::StateCompact,
            AmalgamDependencies::ValuesDependent,
        ];

        for dependence_type in dependencies_types.iter() {
            // Define learner setup for each AmalgamDependency type with BaumWelch fitness
            let learner_setup = LearnerSpecificSetup::AmalgamIdea {
                iter_memory: Some(true),
                dependence_type: Some(dependence_type.clone()),
                fitness_type: Some(AmalgamFitness::BaumWelch),
                max_iterations: Some(1000),
            };

            // Initialize HMMInitializer with each dependency type
            let initializer = HMMInitializer::new(&learner_setup);

            // Obtain initial values
            let initial_values = initializer.get_intial_values(&sequence_set, num_states).expect("Failed to get initial values");

            // Check that output is of type LearnerSpecificInitialValues::AmalgamIdea
            if let LearnerSpecificInitialValues::AmalgamIdea { initial_distributions: Some((means, covs)) } = initial_values {
                // Set expected means and covs count based on dependence type
                let expected_counts = match dependence_type {
                    AmalgamDependencies::AllIndependent => num_states * 2, // Separate values and noises for each state
                    AmalgamDependencies::StateCompact => num_states, // Compact representation with each state having value and noise together
                    AmalgamDependencies::ValuesDependent => 2, // Single vector for values, another for noises
                };

                // Verify that means and covs match expected counts
                assert_eq!(means.len(), expected_counts, "Unexpected number of mean vectors for {:?}", dependence_type);
                assert_eq!(covs.len(), expected_counts, "Unexpected number of covariance matrices for {:?}", dependence_type);

            } else {
                panic!("Expected LearnerSpecificInitialValues::AmalgamIdea with initial distributions for {:?}", dependence_type);
            }
        }
    }

    #[test]
    fn test_baumwelch_initialization() {
        // Generate test sequence
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Define learner setup for BaumWelch
        let learner_setup = LearnerSpecificSetup::BaumWelch {
            termination_criterion: None,
        };

        // Initialize HMMInitializer for BaumWelch
        let initializer = HMMInitializer::new(&learner_setup);

        // Obtain initial values
        let initial_values = initializer.get_intial_values(&sequence_set, num_states).expect("Failed to get initial values");

        // Check that output is of type LearnerSpecificInitialValues::BaumWelch
        if let LearnerSpecificInitialValues::BaumWelch { states, start_matrix, transition_matrix } = initial_values {
            // Verify that states, start_matrix, and transition_matrix match expected sizes and types

            // Check states count
            assert_eq!(states.len(), num_states, "Unexpected number of states");

            // Validate that start and transition matrices match the expected sizes
            assert_eq!(start_matrix.unwrap().matrix.len(), num_states, "Unexpected start matrix size");
            assert_eq!(transition_matrix.as_ref().unwrap().matrix.len(), num_states, "Unexpected transition matrix row count");
            for row in &transition_matrix.unwrap().matrix.raw_matrix {
                assert_eq!(row.len(), num_states, "Unexpected transition matrix column count");
            }
        } else {
            panic!("Expected LearnerSpecificInitialValues::BaumWelch with initial values for BaumWelch optimization");
        }
    }
}