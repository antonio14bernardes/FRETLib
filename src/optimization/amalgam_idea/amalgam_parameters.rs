use super::super::multivariate_gaussian::CovMatrixType;

#[derive(Debug, Clone)]
pub struct AmalgamIdeaParameters {

    pub population_size: usize,

    // Selection parameter
    pub tau: f64, // Fraction of population to be selected for new distribution computation

    // c_mult parameters
    pub c_mult_inc: f64, // Factor to increase c_mult
    pub c_mult_dec: f64, // Factor to decrease c_mult 
    pub c_mult_min: f64, // If c_mult < c_mult_min stop optimization

    // Memory parameters
    pub eta_cov: f64, // Memory in updating the cov matrix
    pub eta_shift: f64, // Memory in updating the mean

    // Mean shift parameters
    pub alpha_shift: f64, // Fraction of selected individuals to shift
    pub gamma_shift: f64, // Factor for shift amount

    // Stagnant Iterations Threshold
    pub stagnant_iterations_threshold: usize,
}

impl AmalgamIdeaParameters {
    pub fn new
    (
    population_size: usize,
    tau: f64,
    c_mult_inc: f64, 
    c_mult_dec: f64,
    c_mult_min: f64,
    eta_cov: f64,
    eta_shift: f64,
    alpha_shift: f64,
    gamma_shift: f64,
    stagnant_iterations_threshold: usize,
    ) -> Self {

        Self{ population_size, tau, c_mult_inc, c_mult_dec, c_mult_min, eta_cov, eta_shift, alpha_shift, gamma_shift, stagnant_iterations_threshold }
    }

    pub fn new_auto(prob_size: usize, cov_mat_type: &CovMatrixType, factorized: bool, memory: bool) -> Self {

        let tau = 0.35; // Independent of everything else
        let gamma_shift = 2.0; // Independent of everything else
        
        let c_mult_min = 1e-4; // Independent of everything else. 

        // If subset indices only has one entry, then the optimal subset division hasnt been performed
        // As such, check if cov_mat_type is Full or Diagonal and compute parameters accordingly
        // Also, check if there is memory - if we're using baseline Amalgam or iAmalgam

        let mut alphas_cov = [0.0_f64; 3];
        let mut alphas_shift = [0.0_f64; 3];
        let alphas_population_size: [f64; 3];

        if !factorized {
            match cov_mat_type {
                CovMatrixType::Full => {
                    if memory {
                        alphas_cov = [-1.1, 1.2, 1.6];
                        alphas_shift = [-1.2, 0.31, 0.50];

                        alphas_population_size = [0.0, 10.0, 0.5];

                    } else {
                        alphas_population_size = [17.0, 3.0, 1.5];
                    }
                    
                }
                CovMatrixType::Diagonal => {
                    if memory {
                        alphas_cov = [-0.40, 0.15, -0.034];
                        alphas_shift = [-0.31, 0.70, 0.65];

                        alphas_population_size = [0.0, 4.0, 0.5];
                    } else {
                        alphas_population_size = [0.0, 10.0, 0.5];
                    }
                    
                }
            }
        } else {
            if memory {
                alphas_cov = [-0.33, 1.5,1.1];
                alphas_shift = [-0.52, 0.70, 0.65];

                alphas_population_size = [0.0, 7.0, 0.5];
            } else {
                alphas_population_size = [12.0, 8.0, 0.7];
            }
            
        }


        let population_size = (alphas_population_size[0] + alphas_population_size[1] * (prob_size as f64).powf(alphas_population_size[2])) as usize;

        let eta_cov = if memory {
            1.0 - (alphas_cov[0] * (population_size as f64).powf(alphas_cov[1]) / (prob_size as f64).powf(alphas_cov[2])).exp()
        } else {
            1.0
        };

        let eta_shift = if memory {
            1.0 - (alphas_shift[0] * (population_size as f64).powf(alphas_shift[1]) / (prob_size as f64).powf(alphas_shift[2])).exp()
        } else {
            1.0
        };




        let alpha_shift = 0.5 * tau * population_size as f64 / (population_size + 1) as f64;
        
        let c_mult_dec = 0.9;
        let c_mult_inc = 1.0 / c_mult_dec;

        let stagnant_iterations_threshold: usize = 25 + prob_size;


        Self {
            population_size,
            tau,
            c_mult_inc,
            c_mult_dec,
            c_mult_min,
            eta_cov,
            eta_shift,
            alpha_shift,
            gamma_shift,
            stagnant_iterations_threshold,
        }

    }
}
