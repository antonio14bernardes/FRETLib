use super::amalgam_fitness_functions::*;
use crate::{optimization::{amalgam_idea::AmalgamIdea, constraints::OptimizationConstraint}, signal_analysis::hmm::learning::learner_trait::HMMLearnerError};
use std::fmt;

// Max noise constraits are obtained by dividing the range of values by the number of states
// and then multiplying by max noise multiplier. The min noise is obtained by multiplying the max noise by the min noise mult
const MAX_NOISE_MULT: f64 = 0.5;
const MIN_NOISE_MULT: f64 = 1e-2;

pub const AMALGAM_ITER_MEMORY_DEFAULT: bool = true;
pub const AMALGAM_FITNESS_DEFAULT: AmalgamFitness = AmalgamFitness::Direct;
pub const AMALGAM_DEPENDENCY_DEFAULT: AmalgamDependencies = AmalgamDependencies::AllIndependent;
pub const AMALGAM_MAX_ITERS_DEFAULT: usize = 1000;

#[derive(Debug, Clone)]
pub enum AmalgamDependencies {
    AllIndependent, // Recommended
    StateCompact,
    ValuesDependent,
}

#[derive(Debug, Clone)]
pub enum AmalgamFitness {
    Direct, // Recommended
    BaumWelch
}

impl fmt::Display for AmalgamDependencies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AmalgamDependencies::AllIndependent => write!(f, "All Independent"),
            AmalgamDependencies::StateCompact => write!(f, "State Compact"),
            AmalgamDependencies::ValuesDependent => write!(f, "Values Dependent"),
        }
    }
}

impl fmt::Display for AmalgamFitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AmalgamFitness::Direct => write!(f, "Direct"),
            AmalgamFitness::BaumWelch => write!(f, "Baum-Welch"),
        }
    }
}

pub fn get_variable_subsets(num_states: usize, dependence_type: &AmalgamDependencies, fitness_type: &AmalgamFitness) -> Result<Vec<Vec<usize>>, HMMLearnerError> {
    if num_states == 0 {return Err(HMMLearnerError::InvalidNumberOfStates)};

    // All state noises and values are independent. Start and transition matrix are still row-wise dependent
    let mut dependencies:Vec<Vec<usize>> = Vec::new();

    match dependence_type {
        AmalgamDependencies::AllIndependent => {        
            // Add the state values
            (0..num_states).for_each(|idx| dependencies.push(vec![idx]));

            // Add the state noises
            (0..num_states).for_each(|idx| dependencies.push(vec![idx + num_states]));
        }

        AmalgamDependencies::StateCompact => {
            // Set state value and noise
            (0..num_states).for_each(|idx| dependencies.push(vec![idx, idx + num_states]));
        }

        AmalgamDependencies::ValuesDependent => {
            // Add the states
            dependencies.push((0..num_states).collect());

            // Add the noises
            dependencies.push((num_states..2*num_states).collect());
        }
    }

    match fitness_type {
        AmalgamFitness::Direct => {
            let mut idx_offset = num_states * 2;
            
            // Add the start_matrix
            dependencies.push((idx_offset..idx_offset + num_states).collect());
            idx_offset += num_states;
            
            // Add the transition_matrix
            (0..num_states).for_each(|row_idx| {
                let lower_lim = idx_offset + row_idx * num_states;
                let upper_lim = idx_offset + (row_idx + 1) * num_states;
                let new_vec = (lower_lim .. upper_lim).collect();
                dependencies.push(new_vec);
            });
        }
        AmalgamFitness::BaumWelch => {}
    }

    Ok(dependencies)
}

pub fn get_constraints(num_states: usize, sequence_set: &Vec<Vec<f64>>, dependence_type: &AmalgamDependencies, fitness_type: &AmalgamFitness) -> Result<Vec<OptimizationConstraint<f64>>, HMMLearnerError> {
    if num_states == 0 {return Err(HMMLearnerError::InvalidNumberOfStates)};
    let mut constraints: Vec<OptimizationConstraint<f64>> = Vec::new();

    let min_value= sequence_set.iter().map(|sequence| sequence.iter().min_by(|a, b| a.total_cmp(b)).unwrap()).min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_value = sequence_set.iter().map(|sequence| sequence.iter().max_by(|a, b| a.total_cmp(b)).unwrap()).max_by(|a, b| a.total_cmp(b)).unwrap();
    let range = max_value - min_value;
    // Define the noise limits as such:
    let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
    let min_noise = max_noise * MIN_NOISE_MULT;    

    match dependence_type {
        AmalgamDependencies::AllIndependent => {
            // Set the state value constraints
            (0..num_states).for_each(|_| constraints.push(OptimizationConstraint::MaxMinValue { max: vec![*max_value], min: vec![*min_value] }));

            // Set the state noise constraints
            (0..num_states).for_each(|_| constraints.push(OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] }));
        }

        AmalgamDependencies::StateCompact => {
            // Set the state value and noise constraints
            (0..num_states).for_each(|_| constraints.push(OptimizationConstraint::MaxMinValue { max: vec![*max_value, max_noise], min: vec![*min_value, min_noise] }));
        }

        AmalgamDependencies::ValuesDependent => {
            // Set the state value constraints
            constraints.push(OptimizationConstraint::MaxMinValue { max: vec![*max_value], min: vec![*min_value] });

            // Set the state noise constraints
            constraints.push(OptimizationConstraint::MaxMinValue { max: vec![max_noise], min: vec![min_noise] });
        }
    }

    match fitness_type {
        AmalgamFitness::Direct => {
            // Set the start matrix constraint
            constraints.push(OptimizationConstraint::PositiveSumTo { sum: 1.0 });

            // Set the transition matrix constraints
            (0..num_states).for_each(|_| constraints.push(OptimizationConstraint::PositiveSumTo { sum: 1.0 }));
        }
        AmalgamFitness::BaumWelch => {}
    }

    Ok(constraints)
}

pub fn get_amalgam_object(
    iter_memory: bool,
    num_states: usize,
    sequence_set: &Vec<Vec<f64>>,
    dependence_type: &AmalgamDependencies,
    fitness_type: &AmalgamFitness
) -> Result<AmalgamIdea<AmalgamHMMFitnessFunction, AmalgamHMMFitness>, HMMLearnerError>

{
    

    let prob_size = match fitness_type {
        AmalgamFitness::BaumWelch => 2 * num_states,
        AmalgamFitness::Direct => 3 * num_states + num_states * num_states,
    };

    let mut amalgam: AmalgamIdea<AmalgamHMMFitnessFunction, AmalgamHMMFitness> = AmalgamIdea::new(prob_size, iter_memory);

    let fitness_fn = AmalgamHMMFitnessFunction::new(num_states, sequence_set, fitness_type);
    amalgam.set_fitness_function(fitness_fn);

    // Get the subsets
    let subsets= get_variable_subsets(num_states, dependence_type, fitness_type)?;
    amalgam.set_dependency_subsets(subsets)
    .map_err(|err| HMMLearnerError::AmalgamIdeaError { err })?;

    // Get the constraints
    let constraints = get_constraints(num_states, sequence_set, dependence_type, fitness_type)?;
    amalgam.set_constraints(constraints)
    .map_err(|err| HMMLearnerError::AmalgamIdeaError { err })?;

    Ok(amalgam)
}







#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_variable_subsets_all_independent() {
        let num_states = 3;
        let dependence_type = AmalgamDependencies::AllIndependent;

        // For direct fit function
        let fitness_type = AmalgamFitness::Direct;
        let expected_subsets = vec![
            vec![0], vec![1], vec![2],
            vec![3], vec![4], vec![5],
            vec![6, 7, 8],
            vec![9, 10, 11], vec![12, 13, 14], vec![15, 16, 17]
        ];

        let subsets = get_variable_subsets(num_states, &dependence_type, &fitness_type).unwrap();

        assert_eq!(subsets, expected_subsets, "Obtained subsets do not equal the expected subsets for Direct");

        // For baum welch fit function
        let fitness_type = AmalgamFitness::BaumWelch;
        let expected_subsets = vec![
            vec![0], vec![1], vec![2],
            vec![3], vec![4], vec![5],
        ];

        let subsets = get_variable_subsets(num_states, &dependence_type, &fitness_type).unwrap();

        assert_eq!(subsets, expected_subsets, "Obtained subsets do not equal the expected subsets for BaumWelch");
    }

    #[test]
    fn test_get_constraints_all_independent_direct() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::AllIndependent;
        let fitness_type = AmalgamFitness::Direct;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();

        let expected_count = 2 * num_states + 1 + num_states;
        assert_eq!(constraints.len(), expected_count);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let min_noise = max_noise * MIN_NOISE_MULT;

        // Check state values constraints
        for i in 0..num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_value, "Expected max state value {}", max_value);
                    assert_eq!(min[0], min_value, "Expected min state value {}", min_value);
                },
                _ => panic!("Expected MaxMinValue constraint for state values"),
            }
        }

        // Check noise constraints
        for i in num_states..2 * num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_noise, "Expected max noise value {}", max_noise);
                    assert_eq!(min[0], min_noise, "Expected min noise value {}", min_noise);
                },
                _ => panic!("Expected MaxMinValue constraint for noise values"),
            }
        }

        // Start matrix constraint
        match &constraints[2 * num_states] {
            OptimizationConstraint::PositiveSumTo { sum } => {
                assert_eq!(*sum, 1.0, "Expected start matrix sum to be 1.0");
            },
            _ => panic!("Expected PositiveSumTo constraint for start matrix"),
        }

        // Transition matrix constraints
        for i in (2 * num_states + 1)..constraints.len() {
            match &constraints[i] {
                OptimizationConstraint::PositiveSumTo { sum } => {
                    assert_eq!(*sum, 1.0, "Expected transition matrix row sum to be 1.0");
                },
                _ => panic!("Expected PositiveSumTo constraint for transition matrix row"),
            }
        }
    }
    
    #[test]
    fn test_get_variable_subsets_values_together() {
        let num_states = 3;
        let dependence_type = AmalgamDependencies::ValuesDependent;

        // For direct fit function
        let fitness_type = AmalgamFitness::Direct;
        let expected_subsets = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
            vec![9, 10, 11], vec![12, 13, 14], vec![15, 16, 17]
        ];

        let subsets = get_variable_subsets(num_states, &dependence_type, &fitness_type).unwrap();

        assert_eq!(subsets, expected_subsets, "Obtained subsets do not equal the expected subsets for Direct");

        // For baum welch fit function
        let fitness_type = AmalgamFitness::BaumWelch;
        let expected_subsets = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
        ];

        let subsets = get_variable_subsets(num_states, &dependence_type, &fitness_type).unwrap();

        assert_eq!(subsets, expected_subsets, "Obtained subsets do not equal the expected subsets for BaumWelch");
    }

    #[test]
    fn test_get_constraints_state_compact_direct() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::StateCompact;
        let fitness_type = AmalgamFitness::Direct;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();

        // Expected constraint count: num_states for combined state/noise + 1 for start matrix + num_states for transition matrix
        let expected_count = num_states + 1 + num_states;
        assert_eq!(constraints.len(), expected_count);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let min_noise = max_noise * MIN_NOISE_MULT;

        // Check the structure of the combined state value and noise constraints
        for i in 0..num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_value, "Expected max value for state values to be {}", max_value);
                    assert_eq!(min[0], min_value, "Expected min value for state values to be {}", min_value);
                    assert_eq!(max[1], max_noise, "Expected max noise for state noises to be {}", max_noise);
                    assert_eq!(min[1], min_noise, "Expected min noise for state noises to be {}", min_noise);
                },
                _ => panic!("Expected MaxMinValue constraint for combined state/noise values"),
            }
        }

        // Check the start matrix constraint
        match &constraints[num_states] {
            OptimizationConstraint::PositiveSumTo { sum } => {
                assert_eq!(*sum, 1.0, "Expected start matrix to have PositiveSumTo constraint with sum 1.0");
            },
            _ => panic!("Expected PositiveSumTo constraint for start matrix"),
        }

        // Check the transition matrix constraints
        for i in (num_states + 1)..constraints.len() {
            match &constraints[i] {
                OptimizationConstraint::PositiveSumTo { sum } => {
                    assert_eq!(*sum, 1.0, "Expected transition matrix row to have PositiveSumTo constraint with sum 1.0");
                },
                _ => panic!("Expected PositiveSumTo constraint for transition matrix row"),
            }
        }
    }

    #[test]
    fn test_get_constraints_values_dependent_direct() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::ValuesDependent;
        let fitness_type = AmalgamFitness::Direct;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();
        // Expected constraints for ValuesDependent with Direct fitness:
        // A single constraint for state values and one for state noises, followed by constraints for start and transition matrices.
        assert_eq!(constraints.len(), 6);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let _min_noise = max_noise * MIN_NOISE_MULT;
        // Check state values and noises constraints
        match &constraints[0] {
            OptimizationConstraint::MaxMinValue { max, min } => {
                assert_eq!(max[0], max_value);
                assert_eq!(min[0], min_value);
            },
            _ => panic!("Expected MaxMinValue constraint"),
        }
        match &constraints[1] {
            OptimizationConstraint::MaxMinValue { max, min } => {
                assert!(max[0] > min[0]); // Noise constraints
            },
            _ => panic!("Expected MaxMinValue constraint"),
        }

        // Check for start matrix constraints
        if let OptimizationConstraint::PositiveSumTo { sum } = &constraints[2] {
            assert_eq!(*sum, 1.0, "Expected start matrix constraint to have PositiveSumTo with sum 1.0");
        } else {
            panic!("Expected PositiveSumTo constraint for start matrix");
        }

        // Check for transition matrix constraints
        if let OptimizationConstraint::PositiveSumTo { sum } = &constraints[3] {
            assert_eq!(*sum, 1.0, "Expected transition matrix constraint to have PositiveSumTo with sum 1.0");
        } else {
            panic!("Expected PositiveSumTo constraint for transition matrix");
        }
    }

    #[test]
    fn test_get_constraints_all_independent_baumwelch() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::AllIndependent;
        let fitness_type = AmalgamFitness::BaumWelch;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();

        // Expected constraint count: 2 * num_states for states and noises only (no start or transition matrix constraints)
        let expected_count = 2 * num_states;
        assert_eq!(constraints.len(), expected_count);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let min_noise = max_noise * MIN_NOISE_MULT;

        // Check the structure of the state value constraints
        for i in 0..num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_value, "Expected max value for state values to be {}", max_value);
                    assert_eq!(min[0], min_value, "Expected min value for state values to be {}", min_value);
                },
                _ => panic!("Expected MaxMinValue constraint for state values"),
            }
        }

        // Check the structure of the noise constraints
        for i in num_states..2 * num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_noise, "Expected max value for noise to be {}", max_noise);
                    assert_eq!(min[0], min_noise, "Expected min value for noise to be {}", min_noise);
                },
                _ => panic!("Expected MaxMinValue constraint for state noises"),
            }
        }
    }

    #[test]
    fn test_get_constraints_state_compact_baumwelch() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::StateCompact;
        let fitness_type = AmalgamFitness::BaumWelch;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();

        // Expected constraint count: num_states for combined state/noise only (no start or transition matrix constraints)
        let expected_count = num_states;
        assert_eq!(constraints.len(), expected_count);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let min_noise = max_noise * MIN_NOISE_MULT;

        // Check the structure of the combined state value and noise constraints
        for i in 0..num_states {
            match &constraints[i] {
                OptimizationConstraint::MaxMinValue { max, min } => {
                    assert_eq!(max[0], max_value, "Expected max value for state values to be {}", max_value);
                    assert_eq!(min[0], min_value, "Expected min value for state values to be {}", min_value);
                    assert_eq!(max[1], max_noise, "Expected max noise for state noises to be {}", max_noise);
                    assert_eq!(min[1], min_noise, "Expected min noise for state noises to be {}", min_noise);
                },
                _ => panic!("Expected MaxMinValue constraint for combined state/noise values"),
            }
        }
    }

    #[test]
    fn test_get_constraints_values_dependent_baumwelch() {
        let num_states = 3;
        let sequence_set = vec![vec![0.1, 1.0, 0.5, 2.0], vec![0.3, 1.5, 0.7, 2.5]];
        let dependence_type = AmalgamDependencies::ValuesDependent;
        let fitness_type = AmalgamFitness::BaumWelch;

        let constraints = get_constraints(num_states, &sequence_set, &dependence_type, &fitness_type).unwrap();

        // Expected constraints for ValuesDependent with BaumWelch fitness:
        // A single constraint for state values and one for state noises only (no start or transition matrix constraints)
        assert_eq!(constraints.len(), 2);

        let min_value = sequence_set.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let max_value = sequence_set.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_value - min_value;
        let max_noise = range / (num_states as f64) * MAX_NOISE_MULT;
        let min_noise = max_noise * MIN_NOISE_MULT;

        // Check state values constraint
        match &constraints[0] {
            OptimizationConstraint::MaxMinValue { max, min } => {
                assert_eq!(max[0], max_value);
                assert_eq!(min[0], min_value);
            },
            _ => panic!("Expected MaxMinValue constraint for state values"),
        }

        // Check state noises constraint
        match &constraints[1] {
            OptimizationConstraint::MaxMinValue { max, min } => {
                assert_eq!(max[0], max_noise, "Expected max noise value to be {}", max_noise);
                assert_eq!(min[0], min_noise, "Expected min noise value to be {}", min_noise);
            },
            _ => panic!("Expected MaxMinValue constraint for state noises"),
        }
    }
    
}