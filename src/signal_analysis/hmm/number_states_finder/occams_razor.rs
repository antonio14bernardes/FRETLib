pub fn bayes_information_criterion_binary_search<F>
(test_function: F,  min_n: usize, max_n: usize) -> Result<(usize, f64), BayesInformationCriterionError>
where
    F: Fn(usize) -> (f64, usize, usize), // log-likelihood, total num parameters (not necessarily the value we give the function), num samples
{
    let mut left = min_n;
    let mut right = max_n;
    let mut best_n = 1;
    let mut best_bic = f64::MAX;

    while left <= right {
        let mid = (left + right) / 2;

        // Evaluate BIC at mid and mid+1
        let (ll_mid, num_parameters_mid, samples_mid) = test_function(mid);
        let (ll_next, num_parameters_next, samples_next) = test_function(mid + 1);
        let bic_mid = compute_bic(ll_mid, num_parameters_mid, samples_mid);
        let bic_next = compute_bic(ll_next, num_parameters_next, samples_next);

        if bic_mid < best_bic {
            best_bic = bic_mid;
            best_n = mid;
        }

        // Find direction in which bic is improving
        if bic_mid > bic_next {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if best_bic == f64::MAX {return Err(BayesInformationCriterionError::FailedToExecuteTrialFunction)}

    Ok((best_n, best_bic))
}

fn compute_bic(log_likelihood: f64, k: usize, n_samples: usize) -> f64 {
    -2.0 * log_likelihood + (k as f64) * (n_samples as f64).ln()
}

#[derive(Debug, Clone)]
pub enum BayesInformationCriterionError {
    FailedToExecuteTrialFunction,
}