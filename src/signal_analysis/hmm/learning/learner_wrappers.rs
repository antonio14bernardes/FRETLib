use crate::optimization::amalgam_idea::AmalgamIdea;
use crate::signal_analysis::hmm::amalgam_integration::amalgam_fitness_functions::{AmalgamHMMFitness, AmalgamHMMFitnessFunction};
use crate::signal_analysis::hmm::amalgam_integration::amalgam_modes::get_amalgam_object;
use crate::signal_analysis::hmm::baum_welch::{run_baum_welch_on_sequence_set, BAUM_WELCH_TERMINATION_DEFAULT};
use crate::signal_analysis::hmm::optimization_tracker::TerminationCriterium;
use crate::signal_analysis::hmm::{StartMatrix, State, TransitionMatrix};
use crate::optimization::optimizer::{OptimizationFitness, Optimizer};
use crate::optimization::amalgam_idea::YapLevel;
use super::learner_trait::{HMMLearnerError, HMMLearnerTrait, LearnerSpecificInitialValues, LearnerSpecificSetup, LearnerType};

#[derive(Debug, Clone)]
pub struct AmalgamHMMWrapper {
    amalgam: Option<AmalgamIdea<AmalgamHMMFitnessFunction, AmalgamHMMFitness>>,
    setup_data: Option<LearnerSpecificSetup>,
}

impl AmalgamHMMWrapper {
    pub fn new() -> Self { 
        let mut out = Self{amalgam: None, setup_data: None};
        out.setup_learner(LearnerSpecificSetup::default(&LearnerType::AmalgamIdea)).unwrap();

        out
    }
}

impl HMMLearnerTrait for AmalgamHMMWrapper {

    fn setup_learner(&mut self, specific_setup: LearnerSpecificSetup) -> Result<(), HMMLearnerError> {
        match specific_setup {
            LearnerSpecificSetup::AmalgamIdea { .. } => {}

            _ => {return Err(HMMLearnerError::InvalidSetupType)}
        }
        
        self.setup_data = Some(specific_setup.handle_nones()); // Removes any nones from the setup
        
        Ok(())
    }
    fn initialize_learner(
            &mut self,
            num_states: usize,
            sequence_values: &Vec<Vec<f64>>,
            specific_initial_values: LearnerSpecificInitialValues) -> Result<(), HMMLearnerError> {

        

        if self.setup_data.is_none(){
            return Err(HMMLearnerError::SetupNotPerformed)
        }

        if num_states == 0 {
            return Err(HMMLearnerError::InvalidNumberOfStates)
        }
        
        let mut amalgam: AmalgamIdea<AmalgamHMMFitnessFunction, AmalgamHMMFitness>;

        match self.setup_data.as_ref().unwrap() {
            LearnerSpecificSetup::AmalgamIdea { iter_memory, dependence_type, fitness_type, max_iterations } => {
                let iter_memory = iter_memory.unwrap();
                let dependence_type = dependence_type.as_ref().unwrap().clone();
                let fitness_type = fitness_type.as_ref().unwrap().clone();
                let max_iters = max_iterations.unwrap();

                // Get the object
                amalgam = get_amalgam_object(iter_memory, num_states, sequence_values, &dependence_type, &fitness_type)?;
                
                // If max_iters is not usize::max, set the max_iters
                if max_iters != usize::MAX {
                    amalgam.set_max_iterations(max_iters);
                }
                
                
            }

            _ => {return Err(HMMLearnerError::InvalidSetupType)}
        }

        match specific_initial_values {
            LearnerSpecificInitialValues::AmalgamIdea { initial_distributions } => {
                if let Some((initial_means, initial_covs)) = initial_distributions {
                    amalgam.set_initial_distribution(&initial_means, &initial_covs)
                    .map_err(|err| HMMLearnerError::AmalgamIdeaError { err })?;
                }
            }

            _ => {return Err(HMMLearnerError::InvalidStartValuesType)}
        }

        self.amalgam = Some(amalgam);

        Ok(())

    }

    fn learn(&mut self) -> Result<(Vec<State>, StartMatrix, TransitionMatrix), HMMLearnerError> {
        // Unwrap the amalgam object
        let amalgam = self.amalgam.as_mut().ok_or(HMMLearnerError::SetupNotPerformed)?;
        
        // Run the amalgam optimization
        let _ = amalgam.run()
        .map_err(|err| HMMLearnerError::AmalgamIdeaError { err });

        let (_ind, best_fitness_ref) = amalgam.get_best_solution().unwrap();
        let best_fitness = best_fitness_ref.clone();

        let states = best_fitness.get_states().ok_or(HMMLearnerError::OptimizationOutputNotFound)?.clone();
        let start_matrix = best_fitness.get_start_matrix().ok_or(HMMLearnerError::OptimizationOutputNotFound)?.clone();
        let transition_matrix = best_fitness.get_transition_matrix().ok_or(HMMLearnerError::OptimizationOutputNotFound)?.clone();

        Ok((states, start_matrix, transition_matrix))

        
    }

    fn get_setup_data(&self) -> Option<&LearnerSpecificSetup> {
        self.setup_data.as_ref()
    }

    fn get_learner_type(&self) -> &LearnerType {
        &LearnerType::AmalgamIdea
    }

    fn get_log_likelihood(&self) -> Option<f64> {
        if let Some(amalgam) = self.amalgam.as_ref() {
            let best_sol_output = amalgam.get_best_solution();
            if best_sol_output.is_none() {return None}

            let (_best_ind, best_fitness) = best_sol_output.unwrap();

            return Some(best_fitness.get_fitness());
        }

        None
    }

    fn reset(&mut self) {
        self.amalgam = None;
    }

    fn full_reset(&mut self) {
        self.amalgam = None;
        self.setup_data = None;
    }

    fn clone_box(&self) -> Box<dyn HMMLearnerTrait> {
        Box::new(self.clone())
    }

    fn set_verbosity(&mut self, verbose: bool) {
        if let Some(amalgam) = self.amalgam.as_mut() {
            let verbosity_level = if verbose {YapLevel::ALot} else {YapLevel::ALittle};

            amalgam.set_verbosity(verbosity_level);
        }
    }
}



pub const BAUM_WELCH_SETUP_DEFAULT: LearnerSpecificSetup =
LearnerSpecificSetup::BaumWelch { termination_criterion: None };

#[derive(Debug, Clone)]
pub struct BaumWelchWrapper {
    sequence_set: Option<Vec<Vec<f64>>>,
    initial_states: Option<Vec<State>>,
    initial_start_matrix: Option<StartMatrix>,
    initial_transition_matrix: Option<TransitionMatrix>,
    termination_criterion: Option<TerminationCriterium>,

    log_likelihood: Option<f64>,

    setup_data: Option<LearnerSpecificSetup>,

}

impl BaumWelchWrapper {
    pub fn new() -> Self { 
        let mut out = Self {
            sequence_set: None, 
            initial_states: None,
            initial_start_matrix: None,
            initial_transition_matrix: None,
            termination_criterion: None,
            log_likelihood: None,
            setup_data: None,
        };

        out.setup_learner(BAUM_WELCH_SETUP_DEFAULT).unwrap();

        out
    }
}
    

impl HMMLearnerTrait for BaumWelchWrapper {
    fn setup_learner(&mut self, specific_setup: LearnerSpecificSetup) -> Result<(), HMMLearnerError> {
        
        match specific_setup {
            LearnerSpecificSetup::BaumWelch {..} => {}

            _ => {return Err(HMMLearnerError::InvalidSetupType)}
        }

        self.setup_data = Some(specific_setup);

        Ok(())
    }
    
    fn initialize_learner(
            &mut self,
            num_states: usize,
            sequence_set: &Vec<Vec<f64>>,
            specific_initial_values: LearnerSpecificInitialValues) -> Result<(), HMMLearnerError> {

        if self.setup_data.is_none() {
            return Err(HMMLearnerError::SetupNotPerformed)
        }

        if num_states == 0 {
            return Err(HMMLearnerError::InvalidNumberOfStates);
        }

        // Set the default maximum iterations
        let termination: TerminationCriterium;

        match self.setup_data.as_ref().unwrap() {
            LearnerSpecificSetup::BaumWelch {termination_criterion} => {
                termination = termination_criterion.as_ref().unwrap_or(&BAUM_WELCH_TERMINATION_DEFAULT).clone();
            }

            _ => {return Err(HMMLearnerError::InvalidSetupType)}
        }

        match specific_initial_values {
            LearnerSpecificInitialValues::BaumWelch { states, start_matrix, transition_matrix } => {
                self.initial_states = Some(states);
                self.initial_start_matrix = Some(start_matrix.unwrap_or(StartMatrix::new_balanced(num_states)));
                self.initial_transition_matrix = Some(transition_matrix.unwrap_or(TransitionMatrix::new_balanced(num_states)));
            }

            _ => {return Err(HMMLearnerError::InvalidStartValuesType)}
        }
        
        self.sequence_set = Some(sequence_set.clone());
        self.termination_criterion = Some(termination);

        Ok(())

    }

    fn learn(&mut self) -> Result<(Vec<State>, StartMatrix, TransitionMatrix), HMMLearnerError> {
        // Unwrap the object
        let initial_states = self.initial_states.as_ref().ok_or(HMMLearnerError::SetupNotPerformed)?;
        let sequence_set = self.sequence_set.as_ref().ok_or(HMMLearnerError::InitializationNotPerformed)?;
        let termination_criterion = self.termination_criterion.as_ref().unwrap().clone();
        let initial_start_matrix = self.initial_start_matrix.as_ref().unwrap();
        let initial_transition_matrix = self.initial_transition_matrix.as_ref().unwrap();

        // Run optimization
        let (log_likelihood, states, start_matrix, transition_matrix) = run_baum_welch_on_sequence_set(sequence_set, initial_states.clone(), initial_start_matrix, initial_transition_matrix, &termination_criterion)
        .map_err(|err| HMMLearnerError::BaumWelchError { err })?;

        self.log_likelihood = Some(log_likelihood);

        Ok((states, start_matrix, transition_matrix))

    }

    fn get_setup_data(&self) -> Option<&LearnerSpecificSetup> {
        self.setup_data.as_ref()
    }

    fn get_learner_type(&self) -> &LearnerType {
        &LearnerType::BaumWelch
    }

    fn get_log_likelihood(&self) -> Option<f64> {
        self.log_likelihood
    }

    fn reset(&mut self) {
        self.sequence_set = None;
        self.termination_criterion = None
    }

    fn full_reset(&mut self) {
        self.setup_data = None;
        self.sequence_set = None;
        self.termination_criterion = None
    }

    fn clone_box(&self) -> Box<dyn HMMLearnerTrait> {
        Box::new(self.clone())
    }

    fn set_verbosity(&mut self, verbose: bool) {
        let _ = verbose; // Just to get rid of the warning
        // Nothing to see here -_-
    }

}