use nalgebra::{DMatrix, DVector, Cholesky};
use rand::Rng;
use rand_distr::{Normal, Distribution};

#[derive(Debug, Clone)]
pub struct MultivariateGaussian {
    mean: DVector<f64>,     // Mean vector
    cov: DMatrix<f64>,      // Covariance matrix
    cholesky: DMatrix<f64>,  // Cholesky decomposition for sampling (Lower triangular)
    cholesky_inv: DMatrix<f64>,  // Inverse of the Cholesky mat

}

impl MultivariateGaussian {
    // Create a new multivariate normal distribution with a mean vector and covariance matrix
    pub fn new(mean: DVector<f64>, cov: DMatrix<f64>) -> Result<Self, MultivariateGaussianError> {
        let [cholesky, cholesky_inv] = get_cholesky_and_inv_stable(&mut cov.clone())?;

        Ok(MultivariateGaussian {
            mean,
            cov,
            cholesky,
            cholesky_inv
        })
    }

    pub fn new_from_vecs(mean: Vec<f64>, flat_cov: Vec<f64>) -> Result<Self, MultivariateGaussianError> {
        let mean = DVector::from_vec(mean);

        let cov_vec = Self::unflatten_cov(&flat_cov)?;

        let n = cov_vec.len();
        let mut cov = DMatrix::from_fn(n, n, |i, j| cov_vec[i][j]);

        let [cholesky, cholesky_inv] = get_cholesky_and_inv_stable(&mut cov)?;

        Ok(MultivariateGaussian {
            mean,
            cov,
            cholesky,
            cholesky_inv
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

        let [cholesky, cholesky_inv] = get_cholesky_and_inv_stable(&mut cov_matrix)?;
        Ok(MultivariateGaussian {
            mean,
            cov: cov_matrix,
            cholesky,
            cholesky_inv
        })


        // let res = get_cholesky_and_inv_stable(&mut cov_matrix);
        // // Attempt to get the Cholesky decomposition and its inverse
        // match get_cholesky_and_inv_stable(&mut cov_matrix) {
        //     Ok([cholesky, cholesky_inv]) => Ok(MultivariateGaussian {
        //         mean,
        //         cov: cov_matrix,
        //         cholesky,
        //         cholesky_inv,
        //     }),
        //     Err(e) => {
        //         println!("Problematic population: {:?}", observations);
        //         Err(e)
        //     }
        // }
    }

    // Sample from the multivariate normal distribution using Cholesky decomp
    pub fn sample(&self, rng: &mut impl Rng) -> DVector<f64> {

        // Make a standard normal distribution
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Sample from independent standard normal distribution
        let z: DVector<f64> = DVector::from_fn(self.mean.len(), |_, _| normal.sample(rng));

        // Transform using the Cholesky factor
        let transformed_z = &self.cholesky * z;

        // Add the mean to the result
        self.mean.clone() + transformed_z
    }

    pub fn sample_n(&self, n: usize, rng: &mut impl Rng) -> DMatrix<f64> {
        let dimension = self.mean.len();
        let mut samples = DMatrix::zeros(dimension, n);
    
        for i in 0..n {
            let sample = self.sample(rng);
            samples.set_column(i, &sample);
        }
    
        samples
    }

    pub fn get_mean_cov(&self) -> (&DVector<f64>, &DMatrix<f64>) {
        (&self.mean, &self.cov)
    }

    pub fn get_cholesky(&self) -> &DMatrix<f64> {
        &self.cholesky
    }
    
    pub fn get_cholesky_inv(&self) -> &DMatrix<f64> {
        &self.cholesky_inv
    }

    pub fn flatten_cov(cov: &Vec<Vec<f64>>) -> Vec<f64> {
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

    pub fn unflatten_cov(flat_cov: &Vec<f64>) -> Result<Vec<Vec<f64>>, MultivariateGaussianError> {
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum CovMatrixType {
    Full,
    Diagonal,
}

#[derive(Debug, Clone)]
pub enum MultivariateGaussianError {
    InvalidCovMatrix,
    InvalidFlatCovMatrix,
    EmptyObservationSet,
}

fn add_jitter_to_cov_matrix(cov: &mut DMatrix<f64>, jitter: f64) {
    for i in 0..cov.nrows() {
        cov[(i, i)] += jitter;
    }
}

fn scale_matrix(matrix: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    // Compute a scale factor based on the norm of the covariance matrix
    let scale_factor = matrix.norm();
    
    // Scale the covariance matrix
    let scaled_matrix = matrix / scale_factor;

    (scaled_matrix, scale_factor)
}

fn rescale_matrix(matrix: &DMatrix<f64>, scaling_factor: f64) -> DMatrix<f64> {
    matrix * scaling_factor
}

fn rescale_cholesky(cholesky: &DMatrix<f64>, cov_scale: f64) -> DMatrix<f64> {
    // Extract the lower triangular matrix `L`
    let mut rescaled_l = cholesky.clone();
    
    // Scale Lower triangular by the square root of the scaling factor
    rescaled_l *= cov_scale.sqrt();
    
    rescaled_l

}

fn get_cholesky_stable(cov: &mut DMatrix<f64>) -> Result<DMatrix<f64>, MultivariateGaussianError> {
    let (scaled_cov, scaling_factor) = scale_matrix(cov);
    
    match Cholesky::new(scaled_cov.clone()) {
        Some(cholesky) => {

            // Get the scaled version of the cholesky lower triangular matrix
            let scaled_cholesky = cholesky.l();

            // Rescale to the original scale
            return Ok(rescale_cholesky(&scaled_cholesky, scaling_factor));
            
        },
        None => {
            // If Cholesky fails, add jitter and try again
            let jitter = 1e-6;
            add_jitter_to_cov_matrix(cov, jitter); 
            let (scaled_cov_with_jitter, _scaling_factor_jitter) = scale_matrix(cov);

            let scaled_cholesky = Cholesky::new(scaled_cov_with_jitter.clone())
            .ok_or(MultivariateGaussianError::InvalidCovMatrix)?.l();

            // Rescale to the original scale
            return Ok(rescale_cholesky(&scaled_cholesky, scaling_factor));
        }
    }
}

fn get_cholesky_and_inv_stable(cov: &mut DMatrix<f64>) -> Result<[DMatrix<f64>;2], MultivariateGaussianError> {
    // Try to perform Cholesky decomposition
    let cholesky = get_cholesky_stable(cov)?;

    // Get scaled colesky to then get the inverse in a stable way
    let (scaled_cholesky, scaling_factor) = scale_matrix(&cholesky);

    // Try to get the inverse
    match scaled_cholesky.clone().try_inverse() {
        Some(scaled_inv) => {

            // Rescale to the original scale
            let cholesky_inv = rescale_matrix(&scaled_inv, 1.0/scaling_factor);

            return Ok([cholesky, cholesky_inv]);
        },
        None => {
            // If Cholesky fails, add jitter and try again
            let jitter = 1e-6;
            add_jitter_to_cov_matrix(cov, jitter);

            // Since we changed the cov matrix, let's obtain the cholesky factor again for the new matrix
            let cholesky = get_cholesky_stable(cov)?;
            let (scaled_cholesky, scaling_factor) = scale_matrix(&cholesky);

            // Compute the inverse of the Cholesky factor
            let scaled_inv = scaled_cholesky.clone().try_inverse()
            .ok_or(MultivariateGaussianError::InvalidCovMatrix)?;

            // Rescale to the original scale
            let cholesky_inv = rescale_cholesky(&scaled_inv, 1.0/scaling_factor);
            
            return Ok([cholesky, cholesky_inv]);
        }
    }         
}


#[cfg(test)]
mod tests {
    use rand::thread_rng;

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
        let (mean, _cov) = gaussian.get_mean_cov();

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


    #[test]
    fn test_cholesky_inverse() {
        // Define a simple covariance matrix for testing
        let cov = DMatrix::from_row_slice(3, 3, &[
            4.0, 2.0, 0.6,
            2.0, 5.0, 1.5,
            0.6, 1.5, 3.0,
        ]);

        // Define a mean vector
        let mean = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        // Create a MultivariateGaussian object
        let gaussian = MultivariateGaussian::new(mean, cov).unwrap();

        // Retrieve the Cholesky decomposition and its inverse
        let cholesky = gaussian.cholesky;
        let cholesky_inv = &gaussian.cholesky_inv;

        // Compute the product of the Cholesky factor and its inverse
        let identity_approx = cholesky * cholesky_inv;

        // Check if the result is approximately an identity matrix
        let identity = DMatrix::identity(3, 3);
        let tolerance = 1e-6;
        assert!(
            (&identity_approx - identity).abs().max() < tolerance,
            "Cholesky inverse computation is incorrect, result: \n{:?}",
            identity_approx
        );
    }

    #[test]
    fn test_distribution_sampling() {
        let cov = DMatrix::from_row_slice(3, 3, &[
            4.0, 2.0, 0.6,
            2.0, 5.0, 1.5,
            0.6, 1.5, 3.0,
        ]);

        // Define a mean vector
        let mean = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        // Create a MultivariateGaussian object
        let gaussian = MultivariateGaussian::new(mean.clone(), cov.clone()).unwrap();

        // Get the random number generator
        let mut rng = thread_rng();
        
        // Get new sampled
        let n = 10000;
        let mut sampled = Vec::new();
        for _ in 0..n {
            let new_sample: Vec<f64> = gaussian.sample(&mut rng).iter().map(|a| *a).collect();
            sampled.push(new_sample);
        }

        let as_refs: Vec<&[f64]> = sampled.iter().map(|vec| vec.as_slice()).collect();

        // Compute distribution based on the samples
        let new_gaussian = MultivariateGaussian::from_observations(&as_refs, &CovMatrixType::Full).unwrap();

        let (sampled_mean, sampled_cov) = new_gaussian.get_mean_cov();

        // Define a tolerance for mean and covariance comparison
        let tolerance = 1e-1;

        // Check that the empirical mean is close to the specified mean
        for (i, &value) in sampled_mean.iter().enumerate() {
            assert!(
                (value - mean[i]).abs() < tolerance,
                "Empirical mean at index {} is incorrect. Expected: {}, Got: {}",
                i, mean[i], value
            );
        }

        // Check that the empirical covariance is close to the specified covariance
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert!(
                    (sampled_cov[(i, j)] - cov[(i, j)]).abs() < tolerance,
                    "Empirical covariance at index ({}, {}) is incorrect. Expected: {}, Got: {}",
                    i, j, cov[(i, j)], sampled_cov[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_get_cholesky_stable() {
        // Define a numerically stable covariance matrix
        let mut cov = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.2, 0.5,
            1.2, 3.0, 0.8,
            0.5, 0.8, 2.0,
        ]);

        // Get Cholesky and inverse using `get_cholesky_stable`
        let [cholesky_mat, cholesky_inv_mat] = get_cholesky_and_inv_stable(&mut cov)
            .expect("Cholesky decomposition should succeed for a stable matrix");

        // Obtain expected Cholesky and inverse using nalgebra's Cholesky
        let cholesky = Cholesky::new(cov.clone()).expect("Cholesky decomposition should succeed");
        let expected_cholesky = cholesky.l();
        let expected_cholesky_inv = expected_cholesky.clone().try_inverse()
            .expect("Inverse of Cholesky factor should exist");

        // Set a tolerance for floating-point comparisons
        let tolerance = 1e-6;

        // Check if `cholesky_mat` is close to `expected_cholesky`
        assert!(
            (cholesky_mat - expected_cholesky).abs().max() < tolerance,
            "Computed Cholesky factor does not match expected result."
        );

        // Check if `cholesky_inv_mat` is close to `expected_cholesky_inv`
        assert!(
            (cholesky_inv_mat - expected_cholesky_inv).abs().max() < tolerance,
            "Computed inverse of Cholesky factor does not match expected result."
        );
    }
}