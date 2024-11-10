use learning::learner_trait::LearnerType;
use std::fmt::Debug;
use super::{learner_trait::{HMMLearnerError, HMMLearnerTrait, LearnerSpecificInitialValues, LearnerSpecificSetup}, learner_wrappers::{AmalgamHMMWrapper, BaumWelchWrapper}};
use crate::signal_analysis::hmm::*;

pub const LEARNER_TYPE_DEFAULT: LearnerType = LearnerType::BaumWelch;

#[derive(Debug, Clone)]
pub struct HMMLearner {
    learner: Box<dyn HMMLearnerTrait>,
    learner_type: LearnerType,
}

impl HMMLearner {
    pub fn new_default() -> Self {
        Self {
            learner:Box::new(BaumWelchWrapper::new()),
            learner_type: LearnerType::BaumWelch,
        }
    }
    pub fn new(learner_type: LearnerType) -> Self {

        let learner: Box<dyn HMMLearnerTrait>;

        match learner_type {
            LearnerType::AmalgamIdea => {
                learner = Box::new(AmalgamHMMWrapper::new());
            }
            LearnerType::BaumWelch => {
                learner = Box::new(BaumWelchWrapper::new());
            }
        }
        Self{learner, learner_type}}
}

impl HMMLearnerTrait for HMMLearner {
    fn setup_learner(&mut self, specific_setup: LearnerSpecificSetup) -> Result<(), HMMLearnerError> {
        match (&self.learner_type, &specific_setup) {
            (LearnerType::AmalgamIdea, LearnerSpecificSetup::AmalgamIdea { .. }) => {} // It's an ok combination
            (LearnerType::BaumWelch, LearnerSpecificSetup::BaumWelch { .. }) => {} // Also an ok combo
            _ => {return Err(HMMLearnerError::IncompatibleInputs)} // Not ok at all
        }

        self.learner.setup_learner(specific_setup)?;

        Ok(())
    }
    fn initialize_learner(
        &mut self,
        num_states: usize, 
        sequence_set: &Vec<Vec<f64>>,
        specific_initial_values: LearnerSpecificInitialValues
    ) -> Result<(), HMMLearnerError> {

        if num_states == 0 {
            return Err(HMMLearnerError::InvalidNumberOfStates);
        }

        match (&specific_initial_values, &self.learner_type) {
            // All are AmalgamIdea, proceed with AmalgamHMMWrapper
            (LearnerSpecificInitialValues::AmalgamIdea { .. },
             LearnerType::AmalgamIdea) => {} // Good combo

            // All are BaumWelch, proceed with BaumWelchWrapper
            (LearnerSpecificInitialValues::BaumWelch { .. },
             LearnerType::BaumWelch) => {} // Also good combo

            // Mismatched setup/initial values
            _ => return Err(HMMLearnerError::IncompatibleInputs), // Terrible cursed combo
        };

        self.learner.initialize_learner(num_states, sequence_set, specific_initial_values)?;
        Ok(())
    }


    fn learn(&mut self) -> Result<(Vec<State>, StartMatrix, TransitionMatrix), HMMLearnerError> {
        
        self.learner.learn()
    }

    fn get_setup_data(&self) -> Option<&LearnerSpecificSetup> {
        self.learner.get_setup_data()
    }

    fn get_log_likelihood(&self) -> Option<f64> {
        self.learner.get_log_likelihood()
    }

    fn reset(&mut self) {
        self.learner.reset();
    }

    fn full_reset(&mut self) {
        self.learner.full_reset();
    }

    fn clone_box(&self) -> Box<dyn HMMLearnerTrait> {
        Box::new(self.clone())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness};
    use hmm_instance::HMMInstance;
    use initialization::kmeans::k_means_1_d;
    use nalgebra::{DMatrix, DVector};

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
    fn test_amalgam_learner_direct_fit_fn() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Preprocess sequence set to set constraints
        let max_value = sequence_set.iter()
            .flat_map(|sequence| sequence.iter())
            .cloned()
            .fold(f64::NAN, f64::max);
        let min_value = sequence_set.iter()
            .flat_map(|sequence| sequence.iter())
            .cloned()
            .fold(f64::NAN, f64::min);
        
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64 * 2.0);
        let min_noise = max_noise / 1e2;

        // Flattened values for clustering
        let flattened_values: Vec<f64> = sequence_set.iter().flat_map(|seq| seq.clone()).collect();
        let (cluster_means, cluster_assignments) = k_means_1_d(&flattened_values, num_states, 100, 1e-3);

        let mut cluster_stds = vec![0.0; num_states];
        let mut count_per_cluster = vec![0; num_states];
        for &assignment in &cluster_assignments {
            count_per_cluster[assignment] += 1;
        }
        for (value, &assignment) in flattened_values.iter().zip(&cluster_assignments) {
            cluster_stds[assignment] += (value - cluster_means[assignment]).powi(2);
        }
        for i in 0..cluster_stds.len() {
            cluster_stds[i] = (cluster_stds[i] / count_per_cluster[i] as f64).sqrt();
        }

        let mean_of_cluster_stds = cluster_stds.iter().sum::<f64>() / cluster_stds.len() as f64;
        let std_of_cluster_stds = cluster_stds
            .iter()
            .map(|&std| (std - mean_of_cluster_stds).powi(2))
            .sum::<f64>()
            .sqrt()
            / num_states as f64;

        // Define initial values and distributions
        let initial_value_means = cluster_means.clone();
        let initial_value_stds = cluster_stds.clone();
        let initial_noise_means = vec![(max_noise + min_noise) / 2.0; num_states];
        let initial_noise_stds = vec![std_of_cluster_stds; num_states];

        // Initial means and covariances as DVector and DMatrix
        let initial_values_means_vec: Vec<DVector<f64>> = initial_value_means
            .iter()
            .map(|&mean| DVector::from_vec(vec![mean]))
            .collect();

        let initial_values_stds_vec: Vec<DMatrix<f64>> = initial_value_stds
            .iter()
            .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
            .collect();

        let initial_noise_means_vec: Vec<DVector<f64>> = initial_noise_means
            .iter()
            .map(|&mean| DVector::from_vec(vec![mean]))
            .collect();

        let initial_noise_stds_vec: Vec<DMatrix<f64>> = initial_noise_stds
            .iter()
            .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
            .collect();

        // Initial distributions for start and transition matrices
        let initial_prob_mean: f64 = 1.0 / num_states as f64;
        let initial_prob_std: f64 = 0.25;
        let initial_start_probs_means_vec = vec![DVector::from_vec(vec![initial_prob_mean; num_states])];
        let initial_start_probs_cov_vec = vec![DMatrix::from_diagonal(&DVector::from_vec(vec![initial_prob_std.powi(2); num_states]))];
        let initial_transition_probs_means_vec = (0..num_states)
            .map(|_| DVector::from_vec(vec![initial_prob_mean; num_states]))
            .collect::<Vec<DVector<f64>>>();
        let initial_transition_probs_cov_vec: Vec<nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>>> = (0..num_states)
            .map(|_| DMatrix::from_diagonal(&DVector::from_element(num_states, initial_prob_std.powi(2))))
            .collect::<Vec<DMatrix<f64>>>();

        // Combine into single initial means and covariances
        let initial_means: Vec<DVector<f64>> = [
            initial_values_means_vec,
            initial_noise_means_vec,
            initial_start_probs_means_vec,
            initial_transition_probs_means_vec,
        ]
        .concat();

        let initial_covs: Vec<DMatrix<f64>> = [
            initial_values_stds_vec,
            initial_noise_stds_vec,
            initial_start_probs_cov_vec,
            initial_transition_probs_cov_vec,
        ]
        .concat();

        // Define learner with Amalgam setup
        let mut learner = HMMLearner::new(LearnerType::AmalgamIdea);

        // Setup learner with specific AmalgamIdea setup
        learner
            .setup_learner(
                LearnerSpecificSetup::AmalgamIdea {
                    iter_memory: Some(true),
                    dependence_type: Some(AmalgamDependencies::AllIndependent),
                    fitness_type: Some(AmalgamFitness::Direct),
                    max_iterations: Some(1000),
                },
            )
            .expect("Failed to setup Amalgam learner");
        
        // Initialize learner with the sequence set
        learner
            .initialize_learner(
                num_states,
                &sequence_set,  // Pass sequence_set here
                LearnerSpecificInitialValues::AmalgamIdea {
                    initial_distributions: Some((initial_means, initial_covs)),
                },
            )
            .expect("Failed to initialize Amalgam learner");
        
        // Run learning process and check output
        let (mut states, start_matrix, transition_matrix) = learner
            .learn()
            .expect("Failed to learn using Amalgam");

        // Sort real and learned states by value for comparison
        let mut real_states_sorted = real_states.clone();
        real_states_sorted.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());
        states.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());

        // Define acceptable error threshold (e.g., 5% difference allowed)
        let threshold_relative = 0.05;

        // Compare real and learned states within threshold
        for (real_state, learned_state) in real_states_sorted.iter().zip(&states) {
            let real_value = real_state.get_value();
            let learned_value = learned_state.get_value();
            let error_relative = ((real_value - learned_value).abs() / real_value).abs();

            println!(
                "Real value: {:.3}, Learned value: {:.3}, Error: {:.3}%",
                real_value, learned_value, error_relative * 100.0
            );

            assert!(
                error_relative <= threshold_relative,
                "Learned state value {:.3} is outside the allowed {:.2}% error threshold for real value {:.3}",
                learned_value,
                threshold_relative,
                real_value
            );
        }

        println!("Start Matrix: {:?}", start_matrix);
        println!("Transition Matrix: {:?}", transition_matrix);
    }

    #[test]
    fn test_amalgam_learner_baum_welch_fit_fn() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Preprocess sequence set to set constraints
        let max_value = sequence_set.iter()
            .flat_map(|sequence| sequence.iter())
            .cloned()
            .fold(f64::NAN, f64::max);
        let min_value = sequence_set.iter()
            .flat_map(|sequence| sequence.iter())
            .cloned()
            .fold(f64::NAN, f64::min);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64 * 2.0);
        let min_noise = max_noise / 1e2;

        // Flattened values for clustering
        let flattened_values: Vec<f64> = sequence_set.iter().flat_map(|seq| seq.clone()).collect();
        let (cluster_means, cluster_assignments) = k_means_1_d(&flattened_values, num_states, 100, 1e-3);

        let mut cluster_stds = vec![0.0; num_states];
        let mut count_per_cluster = vec![0; num_states];
        for &assignment in &cluster_assignments {
            count_per_cluster[assignment] += 1;
        }
        for (value, &assignment) in flattened_values.iter().zip(&cluster_assignments) {
            cluster_stds[assignment] += (value - cluster_means[assignment]).powi(2);
        }
        for i in 0..cluster_stds.len() {
            cluster_stds[i] = (cluster_stds[i] / count_per_cluster[i] as f64).sqrt();
        }

        let mean_of_cluster_stds = cluster_stds.iter().sum::<f64>() / cluster_stds.len() as f64;
        let std_of_cluster_stds = cluster_stds
            .iter()
            .map(|&std| (std - mean_of_cluster_stds).powi(2))
            .sum::<f64>()
            .sqrt()
            / num_states as f64;

        // Define initial values and distributions
        let initial_value_means = cluster_means.clone();
        let initial_value_stds = cluster_stds.clone();
        let initial_noise_means = vec![(max_noise + min_noise) / 2.0; num_states];
        let initial_noise_stds = vec![std_of_cluster_stds; num_states];

        // Initial means and covariances as DVector and DMatrix
        let initial_values_means_vec: Vec<DVector<f64>> = initial_value_means
            .iter()
            .map(|&mean| DVector::from_vec(vec![mean]))
            .collect();

        let initial_values_stds_vec: Vec<DMatrix<f64>> = initial_value_stds
            .iter()
            .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
            .collect();

        let initial_noise_means_vec: Vec<DVector<f64>> = initial_noise_means
            .iter()
            .map(|&mean| DVector::from_vec(vec![mean]))
            .collect();

        let initial_noise_stds_vec: Vec<DMatrix<f64>> = initial_noise_stds
            .iter()
            .map(|&std| DMatrix::from_diagonal(&DVector::from_vec(vec![std.powi(2)])))
            .collect();

        // Combine into single initial means and covariances
        let initial_means: Vec<DVector<f64>> = [
            initial_values_means_vec,
            initial_noise_means_vec,
        ]
        .concat();

        let initial_covs: Vec<DMatrix<f64>> = [
            initial_values_stds_vec,
            initial_noise_stds_vec,
        ]
        .concat();

        // Define learner with Amalgam setup
        let mut learner = HMMLearner::new(LearnerType::AmalgamIdea);

        // Setup learner with specific AmalgamIdea setup
        learner
            .setup_learner(
                LearnerSpecificSetup::AmalgamIdea {
                    iter_memory: Some(true),
                    dependence_type: Some(AmalgamDependencies::AllIndependent),
                    fitness_type: Some(AmalgamFitness::BaumWelch),
                    max_iterations: Some(500),
                },
            )
            .expect("Failed to setup Amalgam learner");

        // Initialize learner with sequence set
        learner
            .initialize_learner(
                num_states,
                &sequence_set,  // Pass the sequence set here
                LearnerSpecificInitialValues::AmalgamIdea {
                    initial_distributions: Some((initial_means, initial_covs)),
                },
            )
            .expect("Failed to initialize Amalgam learner");

        // Run learning process and check output
        let (mut states, start_matrix, transition_matrix) = learner
            .learn()
            .expect("Failed to learn using Amalgam");

        // Sort real and learned states by value for comparison
        let mut real_states_sorted = real_states.clone();
        real_states_sorted.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());
        states.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());

        // Define acceptable error threshold (e.g., 5% difference allowed)
        let threshold_relative = 0.05;

        // Compare real and learned states within threshold
        for (real_state, learned_state) in real_states_sorted.iter().zip(&states) {
            let real_value = real_state.get_value();
            let learned_value = learned_state.get_value();
            let error_relative = ((real_value - learned_value).abs() / real_value).abs();

            println!(
                "Real value: {:.3}, Learned value: {:.3}, Error: {:.3}%",
                real_value, learned_value, error_relative * 100.0
            );

            assert!(
                error_relative <= threshold_relative,
                "Learned state value {:.3} is outside the allowed {:.2}% error threshold for real value {:.3}",
                learned_value,
                threshold_relative,
                real_value
            );
        }

        println!("Start Matrix: {:?}", start_matrix);
        println!("Transition Matrix: {:?}", transition_matrix);
    }


    #[test]
    fn test_baum_welch_learner() {
        let (real_states, _real_start_matrix, _real_transition_matrix, sequence_set) = create_test_sequence_set();
        let num_states = real_states.len();

        // Flatten sequence set for clustering
        let flattened_values: Vec<f64> = sequence_set.iter().flat_map(|seq| seq.clone()).collect();

        // Cluster sequence values to get initial distributions
        let num_clusters = num_states;
        let (cluster_means, _cluster_assignments) = k_means_1_d(&flattened_values, num_clusters, 100, 1e-3);

        // Noise calculation
        let min_value = flattened_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_value = flattened_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let noise = range / (num_states as f64 * 0.5);

        // Define initial values for Baum-Welch
        let initial_states = HMMInstance::generate_state_set(&cluster_means, &vec![noise; num_states])
            .expect("Failed to generate initial state set for Baum-Welch");

        // Define learner with Baum-Welch setup
        let mut learner = HMMLearner::new(LearnerType::BaumWelch);

        // Setup learner with specific Baum-Welch setup
        learner
            .setup_learner(
                LearnerSpecificSetup::BaumWelch {
                    termination_criterion: None,
                },
            )
            .expect("Failed to setup Baum-Welch learner");

        // Initialize learner with the sequence set
        learner
            .initialize_learner(
                num_states,
                &sequence_set,
                LearnerSpecificInitialValues::BaumWelch {
                    states: initial_states,
                    start_matrix: None,
                    transition_matrix: None,
                },
            )
            .expect("Failed to setup Baum-Welch learner");

        // Run learning process and check output
        let (mut states, start_matrix, transition_matrix) = learner
            .learn()
            .expect("Failed to learn using Baum-Welch");

        // Sort real and learned states by value
        let mut real_states_sorted = real_states.clone();
        real_states_sorted.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());
        states.sort_by(|a, b| a.get_value().partial_cmp(&b.get_value()).unwrap());

        // Define acceptable error threshold (e.g., 5% difference allowed)
        let threshold_relative = 0.05;

        // Compare real and learned states within threshold
        for (real_state, learned_state) in real_states_sorted.iter().zip(&states) {
            let real_value = real_state.get_value();
            let learned_value = learned_state.get_value();
            let error_relative = ((real_value - learned_value).abs() / real_value).abs();

            println!(
                "Real value: {:.3}, Learned value: {:.3}, Error: {:.3}%",
                real_value, learned_value, error_relative * 100.0
            );

            assert!(
                error_relative <= threshold_relative,
                "Learned state value {:.3} is outside the allowed {:.2}% error threshold for real value {:.3}",
                learned_value,
                threshold_relative,
                real_value
            );
        }

        println!("Start Matrix: {:?}", start_matrix);
        println!("Transition Matrix: {:?}", transition_matrix);
    }
}