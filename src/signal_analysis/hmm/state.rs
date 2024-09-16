#[derive(Debug)]
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
    pub fn new(id: usize, value: f64, noise_std: f64) -> Self {
        State {
            id,
            value,
            noise_std,
            name: None,
        }
    }

    // Method to baptize the state with a legit name
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
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



#[cfg(test)]
mod tests {
    use super::*;

    // Test State creation with default values
    #[test]
    fn test_state_creation() {
        let state = State::new(1, 0.5, 0.1);

        assert_eq!(state.id, 1);
        assert_eq!(state.value, 0.5);
        assert_eq!(state.noise_std, 0.1);
        assert_eq!(state.name, None);
    }

    // Test setting the name for a State
    #[test]
    fn test_set_name() {
        let mut state = State::new(1, 0.5, 0.1);
        state.set_name("TestState".to_string());

        assert_eq!(state.name, Some("TestState".to_string()));
    }

    // Test standard Gaussian emission probability calculation
    // #[test]
    // fn test_standard_gaussian_emission_probability() {
    //     let state = State::new(1, 0.5, 0.1);
        
    // }

    // // Test paper Gaussian emission probability calculation
    // #[test]
    // fn test_paper_gaussian_emission_probability() {
    //     let state = State::new(1, 0.5, 0.1);
        
    // }

    // Test IDTarget trait for State
    #[test]
    fn test_idtarget_for_state() {
        let state = State::new(2, 0.7, 0.15);
        assert_eq!(state.get_id(), 2);
    }

    // Test IDTarget trait for usize
    #[test]
    fn test_idtarget_for_usize() {
        let id: usize = 42;
        assert_eq!(id.get_id(), 42);
    }
}