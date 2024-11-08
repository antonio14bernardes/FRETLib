use super::analysis::hmm_analyzer::{HMMAnalyzer, HMMAnalyzerError};
use super::hmm_initializer::InitializationMethods;
use super::learning::learner_trait::HMMLearnerTrait;
use super::{HMMInitializer, HMMLearner, HMMNumStatesFinder, StartMatrix, State, TransitionMatrix};
use super::{HMMNumStatesFinderError, HMMInitializerError, HMMLearnerError};
use super::{LearnerSpecificInitialValues, LearnerSpecificSetup, LearnerType, NumStatesFindStrat};

#[derive(Debug)]
pub struct HMM {
    num_states_finder: Option<HMMNumStatesFinder>,
    initializer: Option<HMMInitializer>,
    learner: Option<HMMLearner>,
    analyzer: HMMAnalyzer,
    first_component: HMMComponent,
}

impl HMM {
    pub fn new() -> Self {
        Self {
            num_states_finder: None,
            initializer: None,
            learner: None,
            analyzer: HMMAnalyzer::new(),
            first_component: HMMComponent::Analyzer,
        }
    }

    fn run_analyzer(&mut self, input: HMMInput) -> Result<(), HMMError> {

        // If input is compatible:
        if let HMMInput::Analyzer
        { sequence_values, states, start_matrix, transition_matrix } = input {
            // Setup Analyzer
            self.analyzer.setup(&sequence_values, states, start_matrix, transition_matrix)
            .map_err(|err| HMMError::AnalyzerError { err })?;
            
            // Run Analyzer
            self.analyzer.run()
            .map_err(|err| HMMError::AnalyzerError { err })?;
            

            Ok(())

        } else {
            return Err(HMMError::IncompatibleInput { expected: HMMComponent::Analyzer });
        }
    }

    pub fn add_learner(&mut self) {
        self.learner = Some(HMMLearner::new_default());
        self.first_component = HMMComponent::Learner;

        // Reset components that would run before to None
        self.num_states_finder = None;
        self.initializer = None;   
    }

    pub fn set_learner_type(&mut self, learner_type: LearnerType) -> Result<(), HMMError> {
        // Check if learner has been added
        if self.learner.is_none() {return Err(HMMError::LearnerNotDefined)}

        self.learner = Some(HMMLearner::new(learner_type));
        self.first_component = HMMComponent::Learner;

        // Reset components that would run before to None
        self.num_states_finder = None;
        self.initializer = None;  

        Ok(())
    }

    pub fn setup_learner(&mut self, learner_setup: LearnerSpecificSetup) -> Result<(), HMMError> {
        // Check if learner has been added
        if self.learner.is_none() {return Err(HMMError::LearnerNotDefined)}

        // If learner has been added, set it up
        let learner = self.learner.as_mut().unwrap();
        learner.setup_learner(learner_setup)
        .map_err(|err| HMMError::LearnerError { err })?;

        // Reset previous components to None
        self.num_states_finder = None;
        self.initializer = None;

        Ok(())
    }

    fn run_learner(&mut self, input: HMMInput) -> Result<((Vec<State>, StartMatrix, TransitionMatrix), Vec<f64>), HMMError> {
        // Check if learner has been added
        if self.learner.is_none() {return Err(HMMError::LearnerNotDefined)}
        let learner = self.learner.as_mut().unwrap();

        // If input is compatible:
        if let HMMInput::Learner 
        { num_states, sequence_values, learner_init } = input {
            // Initialize learner
            learner.initialize_learner(num_states, &sequence_values, learner_init)
            .map_err(|err| HMMError::LearnerError { err })?;

            // Run learner
            let learned_output = 
            learner.learn().map_err(|err| HMMError::LearnerError { err })?;

            Ok((learned_output, sequence_values))

        } else {
            return Err(HMMError::IncompatibleInput { expected: HMMComponent::Learner });
        }
    }

    pub fn add_initializer(&mut self) -> Result<(), HMMError> {
        // Check if previous component is set 
        if self.learner.is_none() {return Err(HMMError::LearnerNotDefined)}

        // Get the learner's setup data
        let learner_setup = self.learner.as_ref().unwrap().get_setup_data()
        .ok_or(HMMError::LearnerNotSetup)?;

        // Add the initializer
        self.initializer = Some(HMMInitializer::new(learner_setup));
        self.first_component = HMMComponent::Initializer;

        // Reset components that would run before to None
        self.num_states_finder = None;

        Ok(())
    }

    pub fn set_initialization_method(&mut self, initializer_setup: InitializationMethods) -> Result<(), HMMError> {
        if self.initializer.is_none() { return Err(HMMError::InitializerNotDefined) };

        // If the initializer has been added, set it up
        let initializer = self.initializer.as_mut().unwrap();
        initializer.setup_initializer(initializer_setup)
        .map_err(|err| HMMError::InitializerError { err })?;

        // Reset components that would run before to None
        self.num_states_finder = None;

        Ok(())
    }

    fn run_initializer(&mut self, input: HMMInput) -> Result<(LearnerSpecificInitialValues, usize, Vec<f64>), HMMError> {
        // Check if learner has been added
        if self.initializer.is_none() {return Err(HMMError::InitializerNotDefined)}
        let initializer = self.initializer.as_mut().unwrap();

        // If input is compatible:
        if let HMMInput::Initializer { num_states, sequence_values }= input {
            // Run Initializer
            let init_output = initializer.get_intial_values(&sequence_values, num_states)
            .map_err(|err| HMMError::InitializerError { err })?;

            Ok((init_output, num_states, sequence_values))

        } else {
            return Err(HMMError::IncompatibleInput { expected: HMMComponent::Initializer });
        }
    }

    pub fn add_number_of_states_finder(&mut self) -> Result<(), HMMError> {
        if self.initializer.is_none() {return Err(HMMError::InitializerNotDefined)}

        // Add the number of states finder
        self.num_states_finder = Some(HMMNumStatesFinder::new());
        self.first_component = HMMComponent::NumStatesFinder;

        Ok(())
    }

    pub fn set_state_number_finder_strategy(&mut self, strategy: NumStatesFindStrat) -> Result<(), HMMError> {
        if self.num_states_finder.is_none() {return Err(HMMError::NumStatesFinderNotDefined)}

        // If the number of states finder has been added, set it up
        let num_states_finder = self.num_states_finder.as_mut().unwrap();
        num_states_finder.set_strategy(strategy)
        .map_err(|err| HMMError::NumStatesFinderError { err })?;

        Ok(())
    }

    fn run_num_states_finder(&mut self, input: HMMInput) -> Result<(usize, Vec<f64>), HMMError> {
        // Check if learner has been added
        if self.num_states_finder.is_none() {return Err(HMMError::NumStatesFinderNotDefined)}
        let num_states_finder = self.num_states_finder.as_mut().unwrap();

        // If input is compatible:
        if let HMMInput::NumStatesFinder{sequence_values} = input {
            // Run NumStatesFinder
            let num_states = num_states_finder.get_number_of_states(&sequence_values)
            .map_err(|err| HMMError::NumStatesFinderError { err})?;

            Ok((num_states, sequence_values))

        } else {
            return Err(HMMError::IncompatibleInput { expected: HMMComponent::NumStatesFinder });
        }
    }

    pub fn run(&mut self, input: HMMInput) -> Result<(), HMMError> {
        let mut current_input = input;
        if self.num_states_finder.is_some() {
            let (num_states, sequence_values) = self.run_num_states_finder(current_input)?;

            current_input = HMMInput::Initializer { num_states, sequence_values};
            println!("Ran Num states finder");

        }
        if self.initializer.is_some(){
            let (learner_init, num_states, sequence_values) = 
            self.run_initializer(current_input)?;

            current_input = HMMInput::Learner { num_states, sequence_values, learner_init };
            println!("Ran Initializer");

        }
        if self.learner.is_some() {
            let ((states, start_matrix, transition_matrix), sequence_values) = 
            self.run_learner(current_input)?;

            current_input = HMMInput::Analyzer { sequence_values, states, start_matrix, transition_matrix };

            println!("Ran Learner");
        }

        self.run_analyzer(current_input)?;
        println!("Ran Analyzer");

        Ok(())
    }

    pub fn get_analyzer(&self) -> &HMMAnalyzer {
        &self.analyzer
    }
}


#[derive(Debug)]
pub enum HMMInput {
    NumStatesFinder{sequence_values: Vec<f64>},
    Initializer{num_states: usize, sequence_values: Vec<f64>},
    Learner{num_states: usize, sequence_values: Vec<f64>, learner_init: LearnerSpecificInitialValues},
    Analyzer{sequence_values: Vec<f64>, states: Vec<State>, start_matrix: StartMatrix, transition_matrix: TransitionMatrix},
}

#[derive(Debug)]
pub enum HMMComponent {
    NumStatesFinder,
    Initializer,
    Learner,
    Analyzer,
}

#[derive(Debug)]
pub enum HMMError {
    AnalyzerError{err: HMMAnalyzerError},
    LearnerError{err: HMMLearnerError},
    InitializerError{err: HMMInitializerError},
    NumStatesFinderError{err: HMMNumStatesFinderError},

    LearnerNotDefined,
    LearnerNotSetup,

    InitializerNotDefined,

    NumStatesFinderNotDefined,

    IncompatibleInput{expected: HMMComponent},
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::signal_analysis::hmm::hmm_instance::HMMInstance;

    use super::*;
    
    fn create_test_sequence() -> (Vec<State>, StartMatrix, TransitionMatrix, Vec<f64>, Vec<usize>) {
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

        // Generate the sequence
        let (sequence_ids, sequence_values) = HMMInstance::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 1000);

        (real_states, real_start_matrix, real_transition_matrix, sequence_values, sequence_ids)
    }

    fn remap_states(predicted_states: &Vec<State>, real_states: &Vec<State>) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
    
        // Iterate over each predicted state and find the closest real state
        for (pred_index, pred_state) in predicted_states.iter().enumerate() {
            let mut min_distance = f64::MAX;
            let mut best_match = 0;
    
            for (real_index, real_state) in real_states.iter().enumerate() {
                // Calculate distance between predicted and real state (e.g., Euclidean distance of means)
                let distance = (pred_state.get_value() - real_state.get_value()).abs()
                    + (pred_state.get_noise_std() - real_state.get_noise_std()).abs();
    
                if distance < min_distance {
                    min_distance = distance;
                    best_match = real_index;
                }
            }
            mapping.insert(pred_index, best_match);
        }
    
        mapping
    }
    
    #[test]
    fn test_analysis_only() {
        // Generate test sequence
        let (states, start_matrix, transition_matrix, sequence_values, sequence_ids) = create_test_sequence();

        // Create an HMM instance and run the analysis
        let mut hmm = HMM::new();
        
        // Run the analyzer
        let input = HMMInput::Analyzer { 
            sequence_values: sequence_values.clone(), 
            states: states.clone(), 
            start_matrix: start_matrix.clone(), 
            transition_matrix: transition_matrix.clone() 
        };
        let result = hmm.run(input);

        // Assert the result is Ok
        assert!(result.is_ok(), "Analysis-only setup failed: {:?}", result);
        
        // Check if analysis is correct
        let analyzer = hmm.get_analyzer();
        let analyzed_sequence_states = analyzer.get_states_sequence().expect("No state sequence in analyzer");


        // Calculate the percentage match for state sequences
        let state_matches = sequence_ids.iter().zip(analyzed_sequence_states.iter())
            .filter(|(original, analyzed)| original == analyzed)
            .count();
        let state_match_percentage = (state_matches as f64 / sequence_ids.len() as f64) * 100.0;
        println!("State match percentage: {:.2}%", state_match_percentage);

        // Define acceptable match thresholds
        let acceptable_state_match_threshold = 95.0; // 95% match

        // Assert that match percentages are above acceptable thresholds
        assert!(state_match_percentage >= acceptable_state_match_threshold, "State sequence match percentage below threshold: {:.2}%", state_match_percentage);
    }

    #[test]
    fn test_learning_analysis() {
        // Generate test sequence
        let (states, _start_matrix, _transition_matrix, sequence_values, sequence_ids) = create_test_sequence();

        let mut fake_states = Vec::new();
        for (i,state) in states.iter().enumerate() {
            fake_states.push(State::new(i, state.get_value() * 0.9, state.get_noise_std()*0.9).unwrap());
        }

        // Create an HMM instance and run the analysis
        let mut hmm = HMM::new();

        // Add the learner
        hmm.add_learner();
        
        // Run the analyzer
        let input = HMMInput::Learner 
        { 
            num_states: states.len(),
            sequence_values: sequence_values.clone(),
            learner_init: LearnerSpecificInitialValues::BaumWelch { states: fake_states, start_matrix: None, transition_matrix: None } };

        let result = hmm.run(input);

        // Assert the result is Ok
        assert!(result.is_ok(), "Analysis-only setup failed: {:?}", result);
        
        // Check if analysis is correct
        let analyzer = hmm.get_analyzer();
        let analyzed_sequence_states = analyzer.get_states_sequence().expect("No state sequence in analyzer");


        // Calculate the percentage match for state sequences
        let state_matches = sequence_ids.iter().zip(analyzed_sequence_states.iter())
            .filter(|(original, analyzed)| original == analyzed)
            .count();
        let state_match_percentage = (state_matches as f64 / sequence_ids.len() as f64) * 100.0;
        println!("State match percentage: {:.2}%", state_match_percentage);

        // Define acceptable match thresholds
        let acceptable_state_match_threshold = 95.0; // 95% match

        // Assert that match percentages are above acceptable thresholds
        assert!(state_match_percentage >= acceptable_state_match_threshold, "State sequence match percentage below threshold: {:.2}%", state_match_percentage);
    }

    #[test]
    fn test_initialization_learning_analysis() {
        // Generate test sequence
        let (states, _start_matrix, _transition_matrix, sequence_values, sequence_ids) = create_test_sequence();

        // Create an HMM instance
        let mut hmm = HMM::new();

        // Add the learner
        hmm.add_learner();
        
        // Add the initializer
        hmm.add_initializer().unwrap();

        // Input to the initializer with a specified number of states
        let input = HMMInput::Initializer { 
            num_states: states.len(), 
            sequence_values: sequence_values.clone() 
        };

        // Run the initializer step
        let result = hmm.run(input);
        assert!(result.is_ok(), "Initializer setup failed: {:?}", result);

        // Check if analysis is correct
        let analyzer = hmm.get_analyzer();
        let analyzed_sequence_states = analyzer.get_states_sequence().expect("No state sequence in analyzer");

        // Get the remapped state indices based on closest matching states
        let predicted_states = analyzer.get_states().unwrap();
        let state_mapping = remap_states(predicted_states, &states);

        // Apply the mapping to the analyzed sequence states
        let remapped_analyzed_states: Vec<usize> = analyzed_sequence_states
            .iter()
            .map(|&pred_index| *state_mapping.get(&pred_index).unwrap_or(&pred_index)) // Default to the same index if no mapping found
            .collect();

        // Calculate the percentage match for remapped state sequences
        let state_matches = sequence_ids.iter().zip(remapped_analyzed_states.iter())
            .filter(|(original, remapped)| original == remapped)
            .count();
        let state_match_percentage = (state_matches as f64 / sequence_ids.len() as f64) * 100.0;
        println!("State match percentage after remapping: {:.2}%", state_match_percentage);

        // Define acceptable match thresholds
        let acceptable_state_match_threshold = 95.0; // 95% match

        // Assert that match percentages are above acceptable thresholds
        assert!(state_match_percentage >= acceptable_state_match_threshold, "State sequence match percentage below threshold: {:.2}%", state_match_percentage);
    }

    #[test]
    fn test_nsfinder_initialization_learning_analysis() {
        // Generate test sequence
        let (states, _start_matrix, _transition_matrix, sequence_values, sequence_ids) = create_test_sequence();

        // Create an HMM instance
        let mut hmm = HMM::new();

        // Add the learner
        hmm.add_learner();
        
        // Add the initializer
        hmm.add_initializer().unwrap();

        // Add the num state finder
        hmm.add_number_of_states_finder().unwrap();

        // Input to the initializer with a specified number of states
        let input = HMMInput::NumStatesFinder { sequence_values };

        // Run the initializer step
        let result = hmm.run(input);
        assert!(result.is_ok(), "Initializer setup failed: {:?}", result);

        // Check if analysis is correct
        let analyzer = hmm.get_analyzer();
        let analyzed_sequence_states = analyzer.get_states_sequence().expect("No state sequence in analyzer");

        // Get the remapped state indices based on closest matching states
        let predicted_states = analyzer.get_states().unwrap();
        let state_mapping = remap_states(predicted_states, &states);

        // Apply the mapping to the analyzed sequence states
        let remapped_analyzed_states: Vec<usize> = analyzed_sequence_states
            .iter()
            .map(|&pred_index| *state_mapping.get(&pred_index).unwrap_or(&pred_index)) // Default to the same index if no mapping found
            .collect();

        // Calculate the percentage match for remapped state sequences
        let state_matches = sequence_ids.iter().zip(remapped_analyzed_states.iter())
            .filter(|(original, remapped)| original == remapped)
            .count();
        let state_match_percentage = (state_matches as f64 / sequence_ids.len() as f64) * 100.0;
        println!("State match percentage after remapping: {:.2}%", state_match_percentage);

        // Define acceptable match thresholds
        let acceptable_state_match_threshold = 95.0; // 95% match

        // Assert that match percentages are above acceptable thresholds
        assert!(state_match_percentage >= acceptable_state_match_threshold, "State sequence match percentage below threshold: {:.2}%", state_match_percentage);
    }
}