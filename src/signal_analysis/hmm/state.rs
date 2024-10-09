use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct State {
    pub id: usize,
    pub value: f64,
    pub noise_std: f64,
    pub name: Option<String>,
}

impl State {
    // Constructor to create a new State.
    // Name is always set to None when isntantiating
    // Use method set_name to give the state a cool name
    pub fn new(id: usize, value: f64, noise_std: f64) -> Result<Self, StateError> {
        if noise_std <= 0.0 {
            return Err(StateError::InvalidNoiseInput { input: noise_std });
        }

        Ok(State {
            id,
            value,
            noise_std,
            name: None,
        })
    }

    pub fn new_random(
        id: usize, 
        max_value: Option<f64>, 
        min_value: Option<f64>,
        max_noise: Option<f64>, 
        min_noise: Option<f64>
    ) -> Result<Self, StateError> {

        // Set default values if None is provided
        let min_value = min_value.unwrap_or(0.0); // Min value defaults to 0
        let max_value = max_value.unwrap_or(1.0); // max value defaults to 1
        let value_range = max_value - min_value;

        if value_range <= 0.0 { return Err(StateError::InvalidValueLimitInput { max: max_value, min: min_value })}
    
        let min_noise = min_noise.unwrap_or(value_range / 20.0); // Min noise defaults to 1/20 of the value_range
        let max_noise = max_noise.unwrap_or(value_range / 5.0);  // Max noise defaults to 1/5 of the value range
        let noise_range = max_noise - min_noise;
        if noise_range <= 0.0 || max_noise <= 0.0 || min_noise <= 0.0 {
            return Err(StateError::InvalidNoiseLimitInput { max: max_noise, min: min_noise });
        }

        let value = rand::random::<f64>() * value_range + min_value;  // Random value between min_value and max_value
        let noise_std = rand::random::<f64>() * noise_range + min_noise;  // Random noise between min_noise and max_noise

    
        Ok(State {
            id,
            value,
            noise_std,
            name: None,
        })
    }

    // Method to baptize the state with a legit name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }

    pub fn get_noise_std(&self) -> f64 {
        self.noise_std
    }

    // Method to calculate the  emission probability based on an observed FRET value using
    // the standard Gaussian
    pub fn standard_gaussian_emission_probability(&self, observed_fret: f64) -> f64 {
        let variance = self.noise_std * self.noise_std;
        let exponent = -(observed_fret - self.value).powi(2) / (2.0 * variance);
        (1.0 / (self.noise_std * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    // Method to calculate the  emission probability based on an observed FRET value using
    // the simplified Gaussian discussed in the paper (see hmm module root file)
    pub fn paper_gaussian_emission_probability(&self, observed_fret: f64) -> f64 {
        let scaled_diff = (observed_fret - self.value) / self.noise_std;
        (-2.0 * scaled_diff.powi(2)).exp()
    }
}

pub trait IDTarget {
    fn get_id(&self) -> usize;
}

impl IDTarget for usize {
    fn get_id(&self) -> usize {
        *self  // Simply return the value of usize
    }
}

impl IDTarget for State {
    fn get_id(&self) -> usize {
        self.id  // Return the ID field from the State struct
    }
}

// Implement IDTarget for &State
impl IDTarget for &State {
    fn get_id(&self) -> usize {
        self.id
    }
}

#[derive(Debug)]
pub enum StateError {
    InvalidNoiseInput {input: f64},
    InvalidNoiseLimitInput {max: f64, min: f64},
    InvalidValueLimitInput {max: f64, min: f64},
}

pub fn remove_from_state_vec<T: IDTarget>(states: &Vec<State>, to_remove: &[T]) -> Vec<State> {
    let remove_set: HashSet<usize> = to_remove.iter().map(|state| state.get_id()).collect();

    let mut new = states.clone();
    new.retain(|state| !remove_set.contains(&state.get_id()));

    new
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test State creation with default values
    #[test]
    fn test_state_creation() {
        let state = State::new(1, 0.5, 0.1).unwrap();

        assert_eq!(state.id, 1);
        assert_eq!(state.value, 0.5);
        assert_eq!(state.noise_std, 0.1);
        assert_eq!(state.name, None);
    }

    // Test setting the name for a State
    #[test]
    fn test_set_name() {
        let mut state = State::new(1, 0.5, 0.1).unwrap();
        state.set_name("TestState".to_string());

        assert_eq!(state.name, Some("TestState".to_string()));
    }

    // Test IDTarget trait for State
    #[test]
    fn test_idtarget_for_state() {
        let state = State::new(2, 0.7, 0.15).unwrap();
        assert_eq!(state.get_id(), 2);
    }

    // Test IDTarget trait for usize
    #[test]
    fn test_idtarget_for_usize() {
        let id: usize = 42;
        assert_eq!(id.get_id(), 42);
    }

    // Test removing state
    #[test]
    fn test_remove_from_state_vec() {
        let mut states = vec![
            State::new(0, 0.5, 0.1).unwrap(),
            State::new(1, 0.7, 0.2).unwrap(),
            State::new(2, 0.9, 0.3).unwrap(),
            State::new(3, 1.1, 0.4).unwrap(),
        ];

        // Define a list of states to remove based on their IDs
        let states_to_remove = vec![states[1].clone(), states[3].clone()]; // Remove states with IDs 1 and 3

        // Call the function to remove states
        states = remove_from_state_vec(&states, &states_to_remove);

        // The resulting state vector should only have the states with IDs 0 and 2
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].id, 0);
        assert_eq!(states[1].id, 2);
    }
}