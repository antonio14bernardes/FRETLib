use std::fmt;

use crate::signal_analysis::hmm::{
    initialization::{eval_clusters::*, hmm_initializer::{HMMInitializer, HMMInitializerError}, kmeans::{KMEANS_EVAL_METHOD_DEFAULT, KMEANS_MAX_ITERS_DEFAULT, KMEANS_NUM_TRIES_DEFAULT, KMEANS_TOLERANCE_DEFAULT}},
    learning::{hmm_learner::HMMLearner, learner_trait::{HMMLearnerError, HMMLearnerTrait, LearnerSpecificSetup, LearnerType}}};
use super::occams_razor::*;

pub const MAX_K_DEFAULT: usize = 10;
pub const MIN_K_DEFAULT: usize = 1;

#[derive(Debug, Clone)]
pub struct HMMNumStatesFinder {
    strategy: NumStatesFindStrat,
    max_k: usize,
    min_k: usize,
    verbose: bool,
}

impl HMMNumStatesFinder {
    pub fn new() -> Self{
        
        HMMNumStatesFinder{strategy: NumStatesFindStrat::default(), max_k: MAX_K_DEFAULT, min_k: MIN_K_DEFAULT, verbose: true}
    }

    pub fn set_strategy(&mut self, strategy: NumStatesFindStrat) -> Result<(), HMMNumStatesFinderError> {
        if let NumStatesFindStrat::KMeansClustering {  max_iters, .. } = &strategy {
            
            if max_iters.map(|value| value == 0).unwrap_or(false) {
                return Err(HMMNumStatesFinderError::InvalidInput)
            }
           
        }

        self.strategy = strategy;

        Ok(())
    }

    pub fn set_min_max_trial_states(&mut self, max_k: Option<usize>, min_k: Option<usize>) -> Result<(), HMMNumStatesFinderError> {
        let max_k = max_k.unwrap_or(MAX_K_DEFAULT);
        let min_k = min_k.unwrap_or(MIN_K_DEFAULT);

        if max_k == 0 || min_k == 0 {return Err(HMMNumStatesFinderError::InvalidInput)}

        if max_k < min_k {return Err(HMMNumStatesFinderError::IncompatibleInputs)}

        self.max_k = max_k;
        self.min_k = min_k;

        Ok(())
    }

    pub fn set_verbosity(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    pub fn get_number_of_states(&self, sequence_set: &Vec<Vec<f64>>) -> Result<usize, HMMNumStatesFinderError> {
        // Print some stuff for the start of the process
        if self.verbose {self.start_finding_num_states_yap();}

        // Start the process
        let num_states_optimal: usize;

        match &self.strategy {
            NumStatesFindStrat::KMeansClustering { num_tries, max_iters, tolerance, method } => {
                let num_tries = num_tries.unwrap_or(KMEANS_NUM_TRIES_DEFAULT);
                let max_iters = max_iters.unwrap_or(KMEANS_MAX_ITERS_DEFAULT);
                let tolerance = tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT);
                let method = method.as_ref().unwrap_or(&KMEANS_EVAL_METHOD_DEFAULT).clone();

                num_states_optimal = NumStatesFindStrat::kmeans_clustering_strat(sequence_set, self.max_k, self.min_k, num_tries, max_iters, tolerance, &method)?;
            }

            NumStatesFindStrat::BaumWelch => {
                num_states_optimal = NumStatesFindStrat::baum_welch_strat(sequence_set, self.max_k, self.min_k, self.verbose)?;
            }

            NumStatesFindStrat::CurrentSetup { initializer, learner } => {
                num_states_optimal = NumStatesFindStrat::full_setup_strat(sequence_set, self.max_k, self.min_k, &initializer, &learner, self.verbose)?;
            }

        }

        // Print some more stuff
        if self.verbose {self.finish_finding_num_states_yap(num_states_optimal);}
        
        Ok(num_states_optimal)
    }

    pub fn start_finding_num_states_yap(&self) {
        println!("\n\nStarting to Find Number of HMM States");
        println!("---------------------------------\n");
        println!("Number of States Finder Strategy:");
        println!("  Strategy Type: {}", self.strategy);

        // Print common parameters
        println!("  Minimum Number of States (k): {}", self.min_k);
        println!("  Maximum Number of States (k): {}", self.max_k);

        // Match on the strategy to print specific parameters
        match &self.strategy {
            NumStatesFindStrat::KMeansClustering {
                num_tries,
                max_iters,
                tolerance,
                method,
            } => {
                println!("  Parameters for KMeans Clustering:");
                println!(
                    "    Number of Tries: {}",
                    num_tries.unwrap_or(KMEANS_NUM_TRIES_DEFAULT)
                );
                println!(
                    "    Max Iterations: {}",
                    max_iters.unwrap_or(KMEANS_MAX_ITERS_DEFAULT)
                );
                println!(
                    "    Tolerance: {}",
                    tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT)
                );
                println!(
                    "    Evaluation Method: {}",
                    method
                        .as_ref()
                        .unwrap_or(&KMEANS_EVAL_METHOD_DEFAULT)
                );
            }
            NumStatesFindStrat::BaumWelch => {
                println!("  Using the Baum-Welch Method with BIC for model selection.");
            }
            NumStatesFindStrat::CurrentSetup { .. } => {
                println!("  Using the Current Setup Strategy:");
            }
        }
    }

    fn finish_finding_num_states_yap(&self, num_states: usize) {
        println!("\nFinished Finding Number of States. Found {} States", num_states);
    }
}

#[derive(Debug, Clone)]
pub enum NumStatesFindStrat {
    KMeansClustering {num_tries: Option<usize>, max_iters: Option<usize>, tolerance: Option<f64>, method: Option<ClusterEvaluationMethod>},
    BaumWelch,
    CurrentSetup{initializer: HMMInitializer, learner: HMMLearner},
}

impl NumStatesFindStrat {
    pub fn default() -> Self {
        NumStatesFindStrat::KMeansClustering{
            num_tries: Some(KMEANS_NUM_TRIES_DEFAULT),
            max_iters: Some(KMEANS_MAX_ITERS_DEFAULT),
            tolerance: Some(KMEANS_TOLERANCE_DEFAULT),
            method: Some(ClusterEvaluationMethod::default())
        }
    }
    fn kmeans_clustering_strat(
        sequence_set: &Vec<Vec<f64>>,
        max_k: usize,
        min_k: usize,
        num_tries: usize,
        max_iters: usize,
        tolerance: f64,
        method: &ClusterEvaluationMethod
    ) -> Result<usize, HMMNumStatesFinderError> {

        // Flatten the sequence set
        let total_sequence_values = sequence_set.concat();

        let mut best_k: usize = 0;
        let mut best_score = f64::NEG_INFINITY;

        for _ in 0..num_tries {
            let curr_k: usize;
            let curr_score: f64;

            match method {
                ClusterEvaluationMethod::Silhouette => {
                    let simplified = false;
                    (curr_k, curr_score) =
                    silhouette_analysis(&total_sequence_values, min_k, max_k, max_iters, tolerance, simplified)
                    .map_err(|err| HMMNumStatesFinderError::ClusterEvalError { err })?;
    
                }
                ClusterEvaluationMethod::SimplifiedSilhouette => {
                    let simplified = true;
                    (curr_k, curr_score) = 
                    silhouette_analysis(&total_sequence_values, min_k, max_k, max_iters, tolerance, simplified)
                    .map_err(|err| HMMNumStatesFinderError::ClusterEvalError { err })?;
                }
            }

            if curr_score > best_score {
                best_score = curr_score;
                best_k = curr_k;
            }
        }

        
        
        Ok(best_k)
    }

    fn baum_welch_strat(sequence: &Vec<Vec<f64>>, max_k: usize, min_k: usize, verbose: bool) -> Result<usize, HMMNumStatesFinderError> {
        let trial_enclosure = |num_states: usize| {
            let owned_sequence = sequence.to_vec();
            trial_fn_baum_welch(&owned_sequence, num_states)
        };
        let (num_states, _) = bayes_information_criterion_binary_search(
            trial_enclosure, min_k, max_k, verbose
        )
        .map_err(|err| HMMNumStatesFinderError::BayesInformationCriterionError { err })?;

        Ok(num_states)
    }

    fn full_setup_strat(sequence: &Vec<Vec<f64>>, max_k: usize, min_k: usize, initializer: &HMMInitializer, learner: &HMMLearner, verbose: bool) -> Result<usize, HMMNumStatesFinderError> {
        let trial_enclosure = |num_states: usize| {
            let owned_sequence = sequence.to_vec();
            let mut initializer_owned = initializer.clone();
            let mut learner_owned = learner.clone();

            // Make initializer be silent. Learner is set to silent inside the trial function
            initializer_owned.set_verbosity(false);

            learner_owned.reset();
        
            trial_fn_full_setup(&owned_sequence, num_states, initializer_owned, learner_owned)
        };
        let (num_states, _) = bayes_information_criterion_binary_search(
            trial_enclosure, min_k, max_k, verbose
        )
        .map_err(|err| HMMNumStatesFinderError::BayesInformationCriterionError { err })?;

        Ok(num_states)
    }
}

impl fmt::Display for NumStatesFindStrat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumStatesFindStrat::KMeansClustering { .. } => write!(f, "KMeans Clustering"),
            NumStatesFindStrat::BaumWelch => write!(f, "Baum-Welch Method"),
            NumStatesFindStrat::CurrentSetup { .. } => write!(f, "Current Setup with Initializer and Learner"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HMMNumStatesFinderError {
    InvalidInput,
    IncompatibleInputs,
    ClusterEvalError {err: ClusterEvalError},
    BayesInformationCriterionError {err: BayesInformationCriterionError},
    HMMInitializerError {err: HMMInitializerError},
    HMMLearnerError {err: HMMLearnerError},
}

fn trial_fn_baum_welch(sequence_set: &Vec<Vec<f64>>, num_states: usize) -> (f64, usize, usize){

    // Compute the number of parameters
    let num_parameters = 3 * num_states + num_states * num_states;
    
    // Get num_samples as the average
    let num_samples = sequence_set.iter().map(|sequence| sequence.len()).sum::<usize>() / sequence_set.len();

    // Get the baum welch setup stuff
    let baum_welch_setup = LearnerSpecificSetup::default(&LearnerType::BaumWelch);

    // Get the initializer
    let initializer = HMMInitializer::new(&baum_welch_setup);

    // Get the initial values for baum welch
    let init_values_res = initializer.get_intial_values(sequence_set, num_states);
    if init_values_res.is_err() {return (f64::NEG_INFINITY, num_parameters, num_samples)}
    let baum_init_values = init_values_res.unwrap();

    // Get, setup, and initialize the baum welch learner
    let mut baum_learner = HMMLearner::new(LearnerType::BaumWelch);
    baum_learner.setup_learner(baum_welch_setup).unwrap();
    baum_learner.initialize_learner(num_states, sequence_set, baum_init_values).unwrap();

    // Run baum_welch learner
    let baum_output = baum_learner.learn();

    // Retrieve log_likelihood from the baum welch object
    let log_likelihood = if baum_output.is_ok() {
        println!("Was good");
        baum_learner.get_log_likelihood().unwrap_or(f64::NEG_INFINITY)
    } else {
        println!("Error is {:?}", baum_output);
        f64::NEG_INFINITY
    };

    println!("Got fitness");
    
    (log_likelihood, num_parameters, num_samples)

}


fn trial_fn_full_setup(sequence_set: &Vec<Vec<f64>>, num_states: usize, initializer: HMMInitializer, mut learner: HMMLearner)
 -> (f64, usize, usize)
 {

    // Compute the number of parameters
    let num_parameters = 3 * num_states + num_states * num_states;
    
    // Get num_samples as the average
    let num_samples = sequence_set.iter().map(|sequence| sequence.len()).sum::<usize>() / sequence_set.len();

    // Get the initial values for baum welch
    let init_values_res = initializer.get_intial_values(sequence_set, num_states);
    if init_values_res.is_err() {return (f64::NEG_INFINITY, num_parameters, num_samples)}
    let learner_init_values = init_values_res.unwrap();

    // Get, setup, and initialize the baum welch learner
    learner.initialize_learner(num_states, sequence_set, learner_init_values).unwrap();

    // Make the learner be silent
    learner.set_verbosity(false);

    // Run baum_welch learner
    let learner_output = learner.learn();

    // Retrieve log_likelihood from the baum welch object
    let log_likelihood = if learner_output.is_ok() {
        learner.get_log_likelihood().unwrap_or(f64::NEG_INFINITY)
    } else {
        f64::NEG_INFINITY
    };

    
    (log_likelihood, num_parameters, num_samples)

}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_analysis::hmm::{amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness}, hmm_instance::HMMInstance, StartMatrix, State, TransitionMatrix};

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
    fn test_kmeans_clustering_strategy() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_values) = create_test_sequence_set();

        let strategy = NumStatesFindStrat::KMeansClustering {
            num_tries: None,
            max_iters: Some(100),
            tolerance: Some(1e-3),
            method: Some(ClusterEvaluationMethod::Silhouette),
        };

        let mut num_states_finder = HMMNumStatesFinder::new();

        num_states_finder.set_strategy(strategy).expect("Failed to set strategy");

        let num_states_optimal = num_states_finder.get_number_of_states(&sequence_values)
            .expect("Failed to get number of states using KMeansClustering strategy");

        println!("Optimal number of states found by KMeans: {}", num_states_optimal);
        assert_eq!(num_states_optimal, real_states.len(), "Did not find the correct number of states");
    }

    #[test]
    fn test_baum_welch_strategy() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_values) = create_test_sequence_set();

        let strategy = NumStatesFindStrat::BaumWelch;

        let mut num_states_finder = HMMNumStatesFinder::new();

        num_states_finder.set_strategy(strategy).expect("Failed to set strategy");

        let num_states_optimal_res = num_states_finder.get_number_of_states(&sequence_values);
        
            
        let num_states_optimal = num_states_optimal_res.expect("Failed to get number of states using BaumWelch strategy");

        println!("Optimal number of states found by Baum-Welch: {}", num_states_optimal);
        assert_eq!(num_states_optimal, real_states.len(), "Did not find the correct number of states");
    }

    #[test]
    fn test_current_setup_strategy_with_amalgam() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_values) = create_test_sequence_set();

        // Set up Amalgam-specific parameters
        let setup = LearnerSpecificSetup::AmalgamIdea {
            iter_memory: Some(true),
            dependence_type: Some(AmalgamDependencies::AllIndependent),
            fitness_type: Some(AmalgamFitness::Direct),
            max_iterations: Some(100),
        };

        let initializer = HMMInitializer::new(&setup);
        let mut learner = HMMLearner::new(LearnerType::AmalgamIdea);
        learner.setup_learner(setup).unwrap();

        let strategy = NumStatesFindStrat::CurrentSetup {
            initializer,
            learner,
        };

        let mut num_states_finder = HMMNumStatesFinder::new();

        num_states_finder.set_strategy(strategy).expect("Failed to set strategy");

        let num_states_optimal = num_states_finder.get_number_of_states(&sequence_values)
            .expect("Failed to get number of states using CurrentSetup strategy with Amalgam");

        println!("Optimal number of states found by Current Setup with Amalgam: {}", num_states_optimal);
        assert_eq!(num_states_optimal, real_states.len(), "Did not find the correct number of states");
    }
}