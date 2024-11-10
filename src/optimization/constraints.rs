use std::ops::{Add, Sub, Div, Mul};
use std::cmp::PartialOrd;

#[derive(Debug, Clone)]
pub enum OptimizationConstraint<T> {
    SumTo {sum: T},
    PositiveSumTo{sum: T},
    MaxValue {max: Vec<T>},
    MinValue {min: Vec<T>},
    MaxMinValue {max: Vec<T>, min: Vec<T>},
    None,
}

impl<T> OptimizationConstraint<T>
where
    T: Default + Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + From<f64> + Into<f64>,
{
    pub fn repair(&self, values: &mut [T]) {
        match self {

            // Warning: Problematic if current sum happens to be 0
            OptimizationConstraint::SumTo { sum } => {
                let mut current_sum = T::from(0.0);
                for value in values.iter().copied() {
                    current_sum = current_sum + value;
                }

                // If current sum is 0, set to the sum / num values I guess
                if current_sum == T::from(0.0) {
                    for i in 0..values.len() {
                        values[i] = *sum / (T::from(values.len() as f64));
                    }
                } else if current_sum != *sum {
                    let factor = *sum / current_sum;
                    for value in values.iter_mut() {
                        *value = *value * factor;
                    }
                }
            }

            // Warning: Problematic if current sum happens to be 0
            OptimizationConstraint::PositiveSumTo{sum} => {
                let mut current_sum = T::from(0.0);
                for value in values.iter_mut() {
                    *value = if *value >= T::default() {*value} else {T::default()};

                    current_sum = current_sum + value.clone();
                }
                // If current sum is 0, set to the sum / num values I guess
                if current_sum == T::from(0.0) {
                    for i in 0..values.len() {
                        values[i] = *sum / (T::from(values.len() as f64));
                    }
                } else if current_sum != *sum {
                    let factor = *sum / current_sum;
                    for value in values.iter_mut() {
                        *value = *value * factor;
                    }
                }
            }

            OptimizationConstraint::MaxValue { max } => {
                for (i, value) in values.iter_mut().enumerate() {
                    if max.len() == 1 {
                        if *value > max[0] {
                            *value = max[0];
                        }
                    } else {
                        if *value > max[i] {
                            *value = max[i];
                        }
                    }
                }
            }

            OptimizationConstraint::MinValue { min } => {
                for (i, value) in values.iter_mut().enumerate() {
                    if min.len() == 1 {
                        if *value < min[0] {
                            *value = min[0];
                        }
                    } else {
                        if *value < min[i] {
                            *value = min[i];
                        }
                    }
                }
            }

            OptimizationConstraint::MaxMinValue { max, min } => {
                
                for (i, value) in values.iter_mut().enumerate() {
                    if max.len() == 1 {
                        if *value > max[0] {
                            *value = max[0];
                        }
                    } else {
                        if *value > max[i] {
                            *value = max[i];
                        }
                    }

                    if min.len() == 1 {
                        if *value < min[0] {
                            *value = min[0];
                        }
                    } else {
                        if *value < min[i] {
                            *value = min[i];
                        }
                    }
                }
            }

            OptimizationConstraint::None => {},
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_to_constraint() {
        let mut values = vec![1.0, 2.0, 3.0];
        let constraint = OptimizationConstraint::SumTo { sum: 12.0 };
        constraint.repair(&mut values);
        
        let sum: f64 = values.iter().sum();
        assert_eq!(sum, 12.0, "Sum should equal the target value of 12.0");

        // Ensure proportionality is maintained
        assert!((values[0] - 2.0).abs() < 1e-6);
        assert!((values[1] - 4.0).abs() < 1e-6);
        assert!((values[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_value_constraint() {
        let mut values = vec![1.0, 2.0, 5.0, 8.0];
        let constraint = OptimizationConstraint::MaxValue { max: vec![3.0, 3.0, 3.0, 3.0] };
        constraint.repair(&mut values);
        
        assert_eq!(values, vec![1.0, 2.0, 3.0, 3.0], "All values should be less than or equal to their corresponding max values.");
    }

    #[test]
    fn test_min_value_constraint() {
        let mut values = vec![1.0, 2.0, 0.5, -1.0];
        let constraint = OptimizationConstraint::MinValue { min: vec![0.0, 1.0, 1.0, 1.0] };
        constraint.repair(&mut values);
        
        assert_eq!(values, vec![1.0, 2.0, 1.0, 1.0], "All values should be greater than or equal to their corresponding min values.");
    }

    #[test]
    fn test_max_min_value_constraint() {
        let mut values = vec![1.0, 5.0, 0.5, -3.0, 7.0];
        let constraint = OptimizationConstraint::MaxMinValue { max: vec![4.0, 4.0, 4.0, 4.0, 4.0], min: vec![0.0, 0.0, 0.0, 0.0, 0.0] };
        constraint.repair(&mut values);

        assert_eq!(values, vec![1.0, 4.0, 0.5, 0.0, 4.0], "Values should be between the corresponding min and max values.");
    }

    #[test]
    fn test_sum_to_constraint_edge_case_zero_sum() {
        let mut values = vec![0.0, 0.0, 0.0];
        let constraint = OptimizationConstraint::SumTo { sum: 0.0 };
        constraint.repair(&mut values);
        
        let sum: f64 = values.iter().sum();
        assert_eq!(sum, 0.0, "Sum should equal the target value of 0.0");
    }

    #[test]
    fn test_sum_to_constraint_negative_sum() {
        let mut values = vec![-1.0, -2.0, -3.0];
        let constraint = OptimizationConstraint::SumTo { sum: -12.0 };
        constraint.repair(&mut values);
        
        let sum: f64 = values.iter().sum();
        assert_eq!(sum, -12.0, "Sum should equal the target value of -12.0");
    }

    #[test]
    fn test_positive_sum_to_constraint() {
        let mut values = vec![-1.0, 2.0, 3.0, -4.0, 5.0];
        let constraint = OptimizationConstraint::PositiveSumTo { sum: 10.0 };
        constraint.repair(&mut values);

        // Check that all values are non-negative
        assert!(values.iter().all(|&x| x >= 0.0), "All values should be non-negative.");

        // Check that the sum matches the target
        let sum: f64 = values.iter().sum();
        assert!((sum - 10.0).abs() < 1e-6, "Sum should equal the target value of 10.0");

        // Ensure the proportionality of non-zero values is maintained
        let original_values = vec![0.0, 2.0, 3.0, 0.0, 5.0]; // Negative values replaced by 0 for comparison
        let adjusted_sum: f64 = original_values.iter().sum();

        for (original, adjusted) in original_values.iter().zip(values.iter()) {
            if *original > 0.0 && adjusted_sum > 0.0 {
                let expected_value = (*original / adjusted_sum) * 10.0;
                assert!((adjusted - expected_value).abs() < 1e-6, "Values should be proportional to the target sum.");
            }
        }
    }
}