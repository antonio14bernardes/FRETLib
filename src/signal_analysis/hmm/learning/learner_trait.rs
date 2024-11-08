use nalgebra::{DMatrix, DVector};
use std::fmt::Debug;

use crate::{optimization::amalgam_idea::AmalgamIdeaError, signal_analysis::hmm::{amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness, AMALGAM_FITNESS_DEFAULT}, baum_welch::BaumWelchError, optimization_tracker::TerminationCriterium, StartMatrix, State, TransitionMatrix}};

pub trait HMMLearnerTrait: Debug{
    fn setup_learner(&mut self, specific_setup: LearnerSpecificSetup) -> Result<(), HMMLearnerError>;
    fn initialize_learner(
        &mut self,
        num_states: usize, 
        sequence_values: &[f64],
        specific_initial_values: LearnerSpecificInitialValues) -> Result<(), HMMLearnerError>;

    fn learn(&mut self) -> Result<(Vec<State>, StartMatrix, TransitionMatrix), HMMLearnerError>;

    fn get_setup_data(&self) -> Option<&LearnerSpecificSetup>;

    fn get_log_likelihood(&self) -> Option<f64>;

    fn reset(&mut self); // reset up to the setup step

    fn full_reset(&mut self);// reset everything

    fn clone_box(&self) -> Box<dyn HMMLearnerTrait>;

}

impl Clone for Box<dyn HMMLearnerTrait> {
    fn clone(&self) -> Box<dyn HMMLearnerTrait> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub enum LearnerType {
    AmalgamIdea,
    BaumWelch,
}

#[derive(Debug, Clone)]
pub enum LearnerSpecificSetup {
    AmalgamIdea{iter_memory: Option<bool>, dependence_type: Option<AmalgamDependencies>, fitness_type: Option<AmalgamFitness>, max_iterations: Option<usize>},
    BaumWelch{termination_criterion: Option<TerminationCriterium>},
}

impl LearnerSpecificSetup {
    pub fn model_distribution(&self) -> bool {
        match &self {
            LearnerSpecificSetup::AmalgamIdea { .. } => true,
            LearnerSpecificSetup::BaumWelch { .. } => false,
        }
    }

    pub fn model_matrix_distribution(&self) -> bool {
        match &self {
            LearnerSpecificSetup::AmalgamIdea { fitness_type, .. } => {
                match fitness_type.as_ref().unwrap_or(&AMALGAM_FITNESS_DEFAULT) {
                    AmalgamFitness::Direct => true,
                    AmalgamFitness::BaumWelch => false,
                }
            },
            LearnerSpecificSetup::BaumWelch { .. } => false,
        }
    }
}

#[derive(Debug)]
pub enum LearnerSpecificInitialValues {
    AmalgamIdea {initial_distributions: Option<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)>},
    BaumWelch {states: Vec<State>, start_matrix: Option<StartMatrix>, transition_matrix: Option<TransitionMatrix>}
}

#[derive(Debug, Clone)]
pub enum HMMLearnerError {
    SetupNotPerformed,
    InitializationNotPerformed,

    InvalidSetupType,
    InvalidStartValuesType,
    IncompatibleInputs,
    
    AmalgamIdeaError{err: AmalgamIdeaError},
    BaumWelchError{err: BaumWelchError},

    InvalidNumberOfStates,
    InvalidSequence,

    OptimizationOutputNotFound,
}