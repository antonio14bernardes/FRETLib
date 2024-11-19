use crate::signal_analysis::hmm::{hmm_instance::HMMInstance, StartMatrix, TransitionMatrix};

use super::{eval_clusters::{silhouette_score_1_d, simplified_silhouette_score_1_d, ClusterEvaluationMethod}, hmm_initializer::{check_validity, HMMInitializerError, START_PROB_STD, STATE_NOISE_MULT, STATE_NOISE_STD_MULT, TRANSITION_PROB_STD}, kmeans::*};

#[derive(Debug, Clone, PartialEq)]
pub enum StateValueInitMethod {
    KMeansClustering{max_iters: Option<usize>, tolerance: Option<f64>, num_tries: Option<usize>, eval_method: Option<ClusterEvaluationMethod>},
    Random,
    Sparse,
    StateHints{state_values: Vec<f64>},
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
            Self::Random => self,
            Self::Sparse => self,
            Self::StateHints { .. } => self,
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

            StateValueInitMethod::StateHints { state_values } => {
                if state_values.len() > num_states {return Err(HMMInitializerError::IncompatibleInputs)}
                
                let mut all_state_values = state_values.clone();
                let num_values_left = num_states - all_state_values.len();

                // Get full states but we only care about values here
                let random_states = HMMInstance::generate_random_states(num_values_left as u16, Some(*max_value), Some(*min_value), None, None)
                .map_err(|err| HMMInitializerError::StateError { err })?;

                let random_state_values: Vec<f64> = random_states.iter().map(|state| state.get_value()).collect();

                all_state_values.extend(random_state_values);


                Ok((all_state_values, None))
            }
        }

    }
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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