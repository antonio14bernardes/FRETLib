use core::f64;

use super::hmm_tools::{StateMatrix2D, StateMatrix3D};
use super::state::*;
use super::hmm_matrices::*;


fn compute_alpha_i_t(
    current_state: &State, time_step: usize, observation_value: &f64,
    start_matrix: &StartMatrix, transition_matrix: &TransitionMatrix, 
    alphas_matrix: &mut StateMatrix2D<f64>) {

    if time_step == 0 {
        // Base case: alpha at time step 0 is just the start probability * emission probability
        let start_prob = start_matrix[current_state];
        let emission_prob = current_state.standard_gaussian_emission_probability(*observation_value);
        alphas_matrix[current_state.get_id()][time_step] = start_prob * emission_prob;
        
    } else {
        let mut new_value: f64 = 0.;

        // Sum over all previous states, following the transition probabilities
        for previous_state_id in 0..start_matrix.matrix.len() {
            let previous_alpha = alphas_matrix[previous_state_id][time_step - 1];

            let transition_prob = transition_matrix[(previous_state_id, current_state.id)];

            new_value += previous_alpha * transition_prob;
        }

        

        new_value *= current_state.standard_gaussian_emission_probability(*observation_value);

        alphas_matrix[current_state.get_id()][time_step] = new_value;

    }
}

pub fn compute_alphas(
    states: &[State], observations: &[f64],
    start_matrix: &StartMatrix, transition_matrix: &TransitionMatrix,
    alphas_matrix: &mut StateMatrix2D<f64>){

    
    // Now exponentiate to get the actual alphas 
    for t in 0..observations.len() {
        for state in states {
            compute_alpha_i_t(
                state,
                t,
                &observations[t],
                start_matrix, transition_matrix, alphas_matrix);
        }
    }

}

pub fn compute_scaled_alphas(
    states: &[State], observations: &[f64],
    start_matrix: &StartMatrix, transition_matrix: &TransitionMatrix,
    alphas_matrix: &mut StateMatrix2D<f64>) -> Vec<f64>{
    
    let mut scaling_factors: Vec<f64> = Vec::with_capacity(observations.len());
    
    // Now exponentiate to get the actual alphas 
    for t in 0..observations.len() {
        let mut normalization  = 0.0;
        for state in states {
            compute_alpha_i_t(
                state,
                t,
                &observations[t],
                start_matrix, transition_matrix, alphas_matrix);

            normalization += alphas_matrix[state][t];
        }

        if normalization == 0.0 {normalization = f64::EPSILON}

        for state in states {
            alphas_matrix[state][t] /= normalization;
        }
        
        scaling_factors.push(normalization);

    }

    scaling_factors
}

fn compute_beta_i_t(
    current_state: &State, time_step: usize, observations: &[f64],
    transition_matrix: &TransitionMatrix, states: &[State],
    betas_matrix: &mut StateMatrix2D<f64>) {

    if time_step == observations.len()-1 {
        // Base case: beta at the last time step is 1
        betas_matrix[current_state.get_id()][time_step] = 1.0;
    } else {
        let mut new_value: f64 = 0.;

        // Sum over all previous states, following the transition probabilities
        for next_state in states {
            let next_beta = betas_matrix[next_state.id][time_step + 1];
            let transition_prob = transition_matrix[(current_state.id, next_state.id)];
            let emission_prob = next_state.standard_gaussian_emission_probability(observations[time_step + 1]);

            new_value += next_beta * transition_prob * emission_prob;
        }

        betas_matrix[current_state.get_id()][time_step] = new_value;

    }
}

pub fn compute_betas(
    states: &[State], observations: &[f64],
    transition_matrix: &TransitionMatrix, betas_matrix: &mut StateMatrix2D<f64>
) {

    // Now exponentiate to get the actual alphas 
    for t in (0..observations.len()).rev() {
        for state in states {
            compute_beta_i_t(state, t, observations, transition_matrix, states, betas_matrix);
        }
    }

}

pub fn compute_scaled_betas(
    states: &[State], observations: &[f64],
    transition_matrix: &TransitionMatrix, betas_matrix: &mut StateMatrix2D<f64>,
    scaling_factors: &[f64]
) {

    // Compute the betas for the last timestep (these do not need scaling?)
    for state in states {
        compute_beta_i_t(state, observations.len()-1, observations, transition_matrix, states, betas_matrix);
    }
    for t in (0..observations.len()-1).rev() {
        for state in states {
            compute_beta_i_t(state, t, observations, transition_matrix, states, betas_matrix);
            betas_matrix[state][t] /= scaling_factors[t+1];
        }
    }
}

pub fn compute_gammas(
    states: &[State], 
    observations: &[f64],
    alphas: &StateMatrix2D<f64>, 
    betas: &StateMatrix2D<f64>, 
    gammas: &mut StateMatrix2D<f64>
) {
    for t in 0..observations.len() {
        let mut normalization_factor = 0.0;

        // Compute unnormalized gamma values and accumulate their sum for normalization
        for state in states {
            gammas[state][t] = alphas[state][t] * betas[state][t];
            normalization_factor += gammas[state][t];
        }

        // Normalize gamma values to ensure they sum to 1 at each time step
        for state in states {
            gammas[state][t] /= normalization_factor;
        }
    }
}

pub fn compute_gammas_with_scaled(
    states: &[State], 
    observations: &[f64],
    scaled_alphas: &StateMatrix2D<f64>, 
    scaled_betas: &StateMatrix2D<f64>, 
    gammas: &mut StateMatrix2D<f64>
) {
    for t in 0..observations.len() {
        for state in states {
            gammas[state][t] = scaled_alphas[state][t] * scaled_betas[state][t];
        }
    }

}

pub fn compute_xis(states: &[State], observations: &[f64], transition_matrix: &TransitionMatrix,
    alphas: &StateMatrix2D<f64>, betas: &StateMatrix2D<f64>, xis: &mut StateMatrix3D<f64>
) {

    for t in 0..observations.len() - 1 {
        let mut normalization_value = 0.0;

        for state_from in states {
            for state_to in states {
                let curr_alpha = alphas[state_from][t];
                let transition_prob = transition_matrix[(state_from, state_to)];
                let emission_prob = state_to.standard_gaussian_emission_probability(observations[t + 1]);
                let next_beta = betas[state_to][t + 1];

                let curr_prob = curr_alpha * transition_prob * emission_prob * next_beta;

                xis[(state_from, state_to, t)] = curr_prob;
                normalization_value += curr_prob;
            }
        }

        for state_from in states {
            for state_to in states {
                xis[(state_from, state_to, t)] /= normalization_value;
            }
        }
    }
    
}

pub fn compute_xis_with_scaled(
    states: &[State], observations: &[f64], transition_matrix: &TransitionMatrix,
    scaled_alphas: &StateMatrix2D<f64>, scaled_betas: &StateMatrix2D<f64>,
    xis: &mut StateMatrix3D<f64>, scaling_factors: &[f64],
) {

    for t in 0..observations.len() - 1 {

        for state_from in states {
            for state_to in states {
                let curr_alpha = scaled_alphas[state_from][t];
                let transition_prob = transition_matrix[(state_from, state_to)];
                let emission_prob = state_to.standard_gaussian_emission_probability(observations[t + 1]);
                let next_beta = scaled_betas[state_to][t + 1];

                let curr_prob = curr_alpha * transition_prob * emission_prob * next_beta;
                let scaling_factor = scaling_factors[t+1];

                xis[(state_from, state_to, t)] = curr_prob / scaling_factor;
            }
        }
    }
}