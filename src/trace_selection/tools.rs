pub fn median_filter(values: &[f64], window_size: usize) -> Vec<f64> {
    let mut filtered = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(values.len());

        let mut window: Vec<f64> = values[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if window.len() == 1 {
            window[0]
        } else if window.len() % 2 == 0 {
            (window[window.len() / 2 - 1] + window[window.len() / 2]) / 2.0
        } else {
            window[window.len() / 2]
        };
        
        filtered.push(median);
    }
    filtered
}

pub fn compute_gradient(filtered_values: &[f64]) -> Vec<f64> {
    let mut gradient = Vec::with_capacity(filtered_values.len() - 1);
    for i in 1..filtered_values.len() {
        gradient.push(filtered_values[i] - filtered_values[i - 1]);
    }
    gradient
}

pub fn compute_mean_and_std(values: &[f64]) -> [f64; 2] {
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    let std_dev = variance.sqrt();
    [mean, std_dev]
}

pub fn compute_snr(values: &[f64]) -> f64 {
    let [mean, std] = compute_mean_and_std(values);

    mean / std
}