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