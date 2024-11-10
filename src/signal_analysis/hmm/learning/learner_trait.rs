use nalgebra::{DMatrix, DVector};
use std::fmt::Debug;

use crate::{optimization::amalgam_idea::AmalgamIdeaError, signal_analysis::hmm::{amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness, AMALGAM_DEPENDENCY_DEFAULT, AMALGAM_FITNESS_DEFAULT, AMALGAM_ITER_MEMORY_DEFAULT, AMALGAM_MAX_ITERS_DEFAULT}, baum_welch::BaumWelchError, optimization_tracker::TerminationCriterium, StartMatrix, State, TransitionMatrix}};

pub trait HMMLearnerTrait: Debug{
    fn setup_learner(&mut self, specific_setup: LearnerSpecificSetup) -> Result<(), HMMLearnerError>;
    fn initialize_learner(
        &mut self,
        num_states: usize, 
        sequence_values: &Vec<Vec<f64>>,
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

impl LearnerType {
    pub fn default() -> Self {
        LearnerType::BaumWelch
    }
}

#[derive(Debug, Clone)]
pub enum LearnerSpecificSetup {
    AmalgamIdea{iter_memory: Option<bool>, dependence_type: Option<AmalgamDependencies>, fitness_type: Option<AmalgamFitness>, max_iterations: Option<usize>},
    BaumWelch{termination_criterion: Option<TerminationCriterium>},
}

impl LearnerSpecificSetup {
    pub fn default(learner_type: &LearnerType) -> LearnerSpecificSetup {
        match learner_type {
            LearnerType::BaumWelch => {
                LearnerSpecificSetup::BaumWelch { termination_criterion: Some(TerminationCriterium::default()) }
            }
            LearnerType::AmalgamIdea => {
                LearnerSpecificSetup::AmalgamIdea { 
                    iter_memory: Some(AMALGAM_ITER_MEMORY_DEFAULT),
                    dependence_type: Some(AMALGAM_DEPENDENCY_DEFAULT),
                    fitness_type: Some(AMALGAM_FITNESS_DEFAULT),
                    max_iterations: Some(AMALGAM_MAX_ITERS_DEFAULT)}
            }
        }
    }

    pub fn handle_nones(self) -> Self{
        match self {
            LearnerSpecificSetup::AmalgamIdea { 
                iter_memory,
                dependence_type,
                fitness_type,
                max_iterations } =>
                {

                let default = Self::default(&LearnerType::AmalgamIdea);
                
                if let LearnerSpecificSetup::AmalgamIdea {
                    iter_memory: iter_memory_default,
                    dependence_type: dependence_type_default,
                    fitness_type: fitness_type_default,
                    max_iterations: max_iterations_default } = default {

                        let iter_memory_new = iter_memory.unwrap_or(iter_memory_default.unwrap());
                        let dependence_type_new = dependence_type.unwrap_or(dependence_type_default.unwrap());
                        let fitness_type_new = fitness_type.unwrap_or(fitness_type_default.unwrap());
                        let max_iteration_new = max_iterations.unwrap_or(max_iterations_default.unwrap());

                        LearnerSpecificSetup::AmalgamIdea { 
                            iter_memory: Some(iter_memory_new),
                            dependence_type: Some(dependence_type_new),
                            fitness_type: Some(fitness_type_new),
                            max_iterations: Some(max_iteration_new)
                        }

                } else {
                    return default;
                }
            }

            LearnerSpecificSetup::BaumWelch { termination_criterion } => {
                let default = LearnerSpecificSetup::default(&LearnerType::BaumWelch);

                if let LearnerSpecificSetup::BaumWelch { termination_criterion: termination_criterion_default } = default {
                    let termination_criterion_new = termination_criterion.unwrap_or(termination_criterion_default.unwrap());

                    LearnerSpecificSetup::BaumWelch { termination_criterion: Some(termination_criterion_new) }
                } else {
                    return default;
                }
            }
        }
    }
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