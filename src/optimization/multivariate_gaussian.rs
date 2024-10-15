use nalgebra::{DMatrix, DVector, Cholesky};
use rand::{distributions, Rng};
use rand_distr::{Normal, Distribution};

#[derive(Debug)]
pub struct MultivariateGaussian {
    mean: DVector<f64>,     // Mean vector
    cov: DMatrix<f64>,      // Covariance matrix
    cholesky: Cholesky<f64, nalgebra::Dyn>,  // Cholesky decomposition for sampling

}

impl MultivariateGaussian {
    // Constructor: Create a new multivariate normal distribution with a mean vector and covariance matrix
    pub fn new(mean: DVector<f64>, cov: DMatrix<f64>) -> Result<Self, MultivariateGaussianError> {
        let cholesky = Cholesky::new(cov.clone())
        .ok_or(MultivariateGaussianError::InvalidCovMatrix)?;  // Cholesky decomposition of covariance matrix

        Ok(MultivariateGaussian {
            mean,
            cov,
            cholesky,
        })
    }

    pub fn new_from_vecs(mean: Vec<f64>, flat_cov: Vec<f64>) -> Result<Self, MultivariateGaussianError> {
        let mean = DVector::from_vec(mean);

        let cov_vec = Self::unflatten_cov(&flat_cov)?;

        let n = cov_vec.len();
        let cov = DMatrix::from_fn(n, n, |i, j| cov_vec[i][j]);

        let cholesky = Cholesky::new(cov.clone())
            .ok_or(MultivariateGaussianError::InvalidCovMatrix)?;

        Ok(MultivariateGaussian {
            mean,
            cov,
            cholesky,
        })
    }

    pub fn from_observations(observations: &[&[f64]], cov_type: &CovMatrixType) -> Result<Self, MultivariateGaussianError> {

        
        let num_observations = observations.len();
        let num_dimensions = observations[0].len();

        if num_observations == 0 || num_dimensions == 0 {
            return Err(MultivariateGaussianError::EmptyObservationSet);
        }
        
        // Compute the mean vector
        let mut mean = vec![0.0; num_dimensions];
        for observation in observations {
            for (i, value) in observation.iter().enumerate() {
                mean[i] += value;
            }
        }
        for m in mean.iter_mut() {
            *m /= num_observations as f64;
        }

        // Convert mean vector to DVector
        let mean = DVector::from_vec(mean);

        // Compute the covariance matrix
    let mut cov_matrix = DMatrix::zeros(num_dimensions, num_dimensions);
    
    match cov_type {
        CovMatrixType::Full => {
            // Compute the full covariance matrix
            for observation in observations {
                let diff = DVector::from_vec(observation.to_vec().clone()) - &mean;
                cov_matrix += &diff * diff.transpose();
            }
        }
        CovMatrixType::Diagonal => {
            // Compute only the diagonal elements of the covariance matrix
            for observation in observations {
                let diff = DVector::from_vec(observation.to_vec().clone()) - &mean;
                for i in 0..num_dimensions {
                    cov_matrix[(i, i)] += diff[i] * diff[i];  // Only update the diagonal elements
                }
            }
        }
    }
    
    cov_matrix /= num_observations as f64;

        // Try to perform Cholesky decomposition
        match Cholesky::new(cov_matrix.clone()) {
            Some(cholesky) => {
                Ok(MultivariateGaussian {
                    mean,
                    cov: cov_matrix,
                    cholesky,
                })
            },
            None => {
                // If Cholesky fails, add jitter and try again
                let jitter = 1e-6; // You can adjust this value as needed
                Self::add_jitter_to_cov_matrix(&mut cov_matrix, jitter);

                // Retry Cholesky decomposition with the modified matrix
                let cholesky = Cholesky::new(cov_matrix.clone())
                    .ok_or(MultivariateGaussianError::InvalidCovMatrix)?;

                Ok(MultivariateGaussian {
                    mean,
                    cov: cov_matrix,
                    cholesky,
                })
            }
        }
    }

    // Sample from the multivariate normal distribution using Cholesky decomp
    pub fn sample(&self, rng: &mut impl Rng) -> DVector<f64> {

        // Make a standard normal distribution
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Sample from independent standard normal distribution
        let z: DVector<f64> = DVector::from_fn(self.mean.len(), |_, _| normal.sample(rng));

        // Transform using the Cholesky factor
        let transformed_z = self.cholesky.l() * z;

        // Add the mean to the result
        self.mean.clone() + transformed_z
    }

    pub fn get_mean_cov(&self) -> (&DVector<f64>, &DMatrix<f64>) {
        (&self.mean, &self.cov)
    }

    fn flatten_cov(cov: &Vec<Vec<f64>>) -> Vec<f64> {
        let n = cov.len();
        let len = n * (n + 1) / 2;  // Length of flattened covariance matrix
    
        let mut flat_cov: Vec<f64> = Vec::with_capacity(len);
    
        // Iterate over the rows (i)
        for i in 0..n {
            // For each row, iterate over the columns (j) where j >= i
            for j in i..n {
                flat_cov.push(cov[i][j]);  // Push upper triangular and diagonal elements
            }
        }
    
        flat_cov
    }

    fn unflatten_cov(flat_cov: &Vec<f64>) -> Result<Vec<Vec<f64>>, MultivariateGaussianError> {
        let len = flat_cov.len();
        let n_float = (-1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0;
    
        // Check if n is an integer
        if n_float.fract() != 0.0 {
            return Err(MultivariateGaussianError::InvalidFlatCovMatrix);
        }
    
        let n = n_float as usize;
    
        let mut unflat_cov: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            unflat_cov.push(vec![0.0; n]);
        }
    
        let mut curr_offset: usize = 0;
    
        for i in 0..n {
            for j in i..n {
                let element = flat_cov[curr_offset];
                unflat_cov[i][j] = element;   // Fill upper triangular and diagonal
                unflat_cov[j][i] = element;   // Mirror to lower triangular
                curr_offset += 1;
            }
        }
    
        Ok(unflat_cov)
    }

    // Helper function to add jitter to the diagonal of a covariance matrix
    fn add_jitter_to_cov_matrix(cov: &mut DMatrix<f64>, jitter: f64) {
        for i in 0..cov.nrows() {
            cov[(i, i)] += jitter;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CovMatrixType {
    Full,
    Diagonal,
}

#[derive(Debug)]
pub enum MultivariateGaussianError {
    InvalidCovMatrix,
    InvalidFlatCovMatrix,
    EmptyObservationSet,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_cov() {
        let cov = vec![
            vec![1.0, 0.8, 0.5],
            vec![0.8, 1.0, 0.3],
            vec![0.5, 0.3, 1.0]
        ];

        // Expected flattened result (upper triangular part)
        let expected_flattened = vec![1.0, 0.8, 0.5, 1.0, 0.3, 1.0];

        let flat_cov = MultivariateGaussian::flatten_cov(&cov);
        assert_eq!(flat_cov, expected_flattened, "Flattened covariance matrix is incorrect");
    }

    #[test]
    fn test_unflatten_cov() {
        let flat_cov = vec![1.0, 0.8, 0.5, 1.0, 0.3, 1.0];

        // Expected unflattened covariance matrix
        let expected_cov = vec![
            vec![1.0, 0.8, 0.5],
            vec![0.8, 1.0, 0.3],
            vec![0.5, 0.3, 1.0]
        ];

        let unflat_cov = MultivariateGaussian::unflatten_cov(&flat_cov).unwrap();
        println!("{:?}",unflat_cov);
        assert_eq!(unflat_cov, expected_cov, "Unflattened covariance matrix is incorrect");
    }

    #[test]
    fn test_unflatten_cov_invalid_length() {
        let invalid_flat_cov = vec![1.0, 0.8, 0.5, 0.1];  // Invalid length
    
        match MultivariateGaussian::unflatten_cov(&invalid_flat_cov) {
            Ok(_) => panic!("Expected error for invalid flat_cov length"),
            Err(MultivariateGaussianError::InvalidFlatCovMatrix) => {
                // Correct error is caught, the test passes
            },
            Err(_) => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_compute_distribution_and_sampling() {
        // Define some sample observations (2D multivariate Gaussian with 3 points)
        let observations: Vec<&[f64]> = vec![
            &[1.0, 2.0],
            &[1.5, 2.5],
            &[0.5, 1.5],
        ];

        // Compute the distribution from the observations
        let gaussian = MultivariateGaussian::from_observations(&observations, &CovMatrixType::Full).unwrap();

        // Ensure the mean and covariance are correct
        let expected_mean = DVector::from_vec(vec![1.0, 2.0]);  // This is the mean of the given data points
        let (mean, cov) = gaussian.get_mean_cov();

        assert_eq!(*mean, expected_mean, "Computed mean is incorrect");

        // Check if we can sample from the computed distribution
        let mut rng = rand::thread_rng();
        let sample = gaussian.sample(&mut rng);

        // The sample should be a DVector with the same dimension as the mean (2D here)
        assert_eq!(sample.len(), 2, "Sampled individual should have the same dimensionality as the mean");
        println!("Sampled individual: {:?}", sample);

    }

    #[test]
    fn test_diagonal_covariance() {
        // Define some sample observations (3D multivariate Gaussian with 3 points)
        let observations: Vec<&[f64]> = vec![
            &[1.0, 2.0, 3.0],
            &[1.5, 2.5, 3.5],
            &[0.5, 1.5, 2.5],
        ];

        // Compute the distribution from the observations using Diagonal covariance matrix type
        let gaussian = MultivariateGaussian::from_observations(&observations, &CovMatrixType::Diagonal).unwrap();

        // Get the covariance matrix
        let (_, cov) = gaussian.get_mean_cov();

        // Check that only the diagonal elements are non-zero
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                if i == j {
                    assert!(cov[(i, j)] > 0.0, "Diagonal elements should be greater than zero");
                } else {
                    assert_eq!(cov[(i, j)], 0.0, "Off-diagonal elements should be zero");
                }
            }
        }

        println!("Covariance matrix (diagonal):\n{}", cov);
    }
}