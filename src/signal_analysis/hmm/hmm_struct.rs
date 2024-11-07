use super::hmm_initializer::InitializationMethods;
use super::learning::learner_trait::HMMLearnerTrait;
use super::{HMMNumStatesFinder, HMMInitializer, HMMLearner};
use super::{HMMNumStatesFinderError, HMMInitializerError, HMMLearnerError};
use super::{LearnerSpecificInitialValues, LearnerSpecificSetup, LearnerType, NumStatesFindStrat};

pub struct HMM {
    num_states_finder: Option<HMMNumStatesFinder>,
    initializer: Option<HMMInitializer>,
    learner: Option<HMMLearner>,
}

impl HMM {
    pub fn new() -> Self {
        Self {
            num_states_finder: None,
            initializer: None,
            learner: None,
        }
    }

    pub fn add_learner(&mut self, learner_type: LearnerType) {
        self.learner = Some(HMMLearner::new(learner_type));

        // Reset components that would run before to None
        self.num_states_finder = None;
        self.initializer = None;   
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

    pub fn add_initializer(&mut self) -> Result<(), HMMError> {
        // Check if previous component is set 
        if self.learner.is_none() {return Err(HMMError::LearnerNotDefined)}

        // Get the learner's setup data
        let learner_setup = self.learner.as_ref().unwrap().get_setup_data()
        .ok_or(HMMError::LearnerNotSetup)?;

        // Add the initializer
        self.initializer = Some(HMMInitializer::new(learner_setup));

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

    pub fn add_number_of_states_finder(&mut self) -> Result<(), HMMError> {
        if self.initializer.is_none() {return Err(HMMError::InitializerNotDefined)}

        // Add the number of states finder
        self.num_states_finder = Some(HMMNumStatesFinder::new());

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
}


pub enum HMMError {
    LearnerError{err: HMMLearnerError},
    InitializerError{err: HMMInitializerError},
    NumStatesFinderError{err: HMMNumStatesFinderError},

    LearnerNotDefined,
    LearnerNotSetup,

    InitializerNotDefined,

    NumStatesFinderNotDefined,
}