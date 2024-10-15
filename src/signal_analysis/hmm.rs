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

use rand::Rng;
use rand_distr::{Distribution, Normal};


pub mod hmm_tools;
pub mod state;
pub mod hmm_matrices;
pub mod viterbi;
pub mod hmm_instance;
pub mod baum_welch;
pub mod optimization_tracker;
pub mod probability_matrices;
pub mod occams_razor;
use state::*;
use hmm_matrices::*; 



pub struct HMM {

}

impl HMM {
    pub fn gen_sequence(
        states: &[State], 
        start_matrix: &StartMatrix, 
        transition_matrix: &TransitionMatrix, 
        time_steps: usize
    ) -> (Vec<usize>, Vec<f64>) {
        let mut rng = rand::thread_rng();
        let mut sequence = Vec::with_capacity(time_steps);
        let mut values = Vec::with_capacity(time_steps);
    
        // Start by choosing the initial state based on the start matrix
        let mut cumulative_prob = 0.0;
        let random_value = rng.gen_range(0.0..1.0);
        let mut current_state = 0;
    
        for (state, &start_prob) in start_matrix.matrix.iter().enumerate() {
            cumulative_prob += start_prob;
            if random_value < cumulative_prob {
                current_state = state;
                break;
            }
        }
        sequence.push(current_state);
    
        // Get theb objects for the Normal dist of each state
        let normal_distributions: Vec<Normal<f64>> = states
            .iter()
            .map(|state| Normal::new(state.value, state.noise_std).unwrap())
            .collect();
    
        // Sample the value for the initial state
        let initial_value = normal_distributions[current_state].sample(&mut rng);
        values.push(initial_value);
    
        // Generate the remaining states based on the transition matrix
        for _ in 1..time_steps {
            cumulative_prob = 0.0;
            let random_value = rng.gen_range(0.0..1.0);
            let transition_probs = &transition_matrix.matrix[current_state];
    
            for (next_state, &trans_prob) in transition_probs.iter().enumerate() {
                cumulative_prob += trans_prob;
                if random_value < cumulative_prob {
                    current_state = next_state;
                    break;
                }
            }
    
            sequence.push(current_state);
    
            // Sample from this state's dist to get the current value
            let state_value = normal_distributions[current_state].sample(&mut rng);
            values.push(state_value);
        }
    
        (sequence, values)
    }
}