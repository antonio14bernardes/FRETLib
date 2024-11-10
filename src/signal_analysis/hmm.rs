/********** Hidden Markov Model (HMM) Module for FRET Analysis **********
* This module is designed based on the methodology and algorithm described in the paper:
*
* "Analysis of Single-Molecule FRET Trajectories Using Hidden Markov Modeling"**
*
* Authors: Sean A. McKinney, Chirlmin Joo, and Taekjip Ha  
* Department of Physics and Howard Hughes Medical Institute, University of Illinois at Urbana-Champaign  
* Published in: *Biophysical Journal*, Volume 91, September 2006, Pages 1941–1951  
* DOI: [10.1529/biophysj.106.082487](https://doi.org/10.1529/biophysj.106.082487)
**********/

pub mod hmm_tools;
pub mod state;
pub mod hmm_matrices;
pub mod viterbi;
pub mod hmm_instance;
pub mod baum_welch;
pub mod optimization_tracker;
pub mod probability_matrices;
pub mod initialization;
pub mod amalgam_integration;
pub mod number_states_finder;
pub mod learning;
pub mod analysis;
pub mod hmm_struct;

use state::*;
use hmm_matrices::*; 

use initialization::hmm_initializer::{HMMInitializer, HMMInitializerError};
use learning::{hmm_learner::HMMLearner, learner_trait::HMMLearnerError};
use number_states_finder::hmm_num_states_finder::{HMMNumStatesFinder, HMMNumStatesFinderError};

pub use initialization::hmm_initializer::{self, StateValueInitMethod, StateNoiseInitMethod, StartMatrixInitMethod,TransitionMatrixInitMethod, InitializationMethods};
pub use learning::{hmm_learner, learner_trait::{LearnerSpecificInitialValues, LearnerType, LearnerSpecificSetup}};
pub use number_states_finder::hmm_num_states_finder::NumStatesFindStrat;


pub use amalgam_integration::amalgam_modes::{AmalgamDependencies, AmalgamFitness};