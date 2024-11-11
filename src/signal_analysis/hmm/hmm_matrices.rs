use rand::Rng;
use std::fmt;

use super::state::*;
use super::hmm_tools::*;
use std::ops::{Index, IndexMut};
use std::collections::HashSet;
pub trait ProbabilityMatrix {

    // Check if the matrix is valid
    fn validate(&self) -> Result<(), MatrixValidationError>;
}

// Custom error for matrix value validation
#[derive(Debug, Clone)]
pub enum MatrixValidationError {
    IncorrectShape,                                                    // If 2D, num of rows and columns must be the same
    VectorIncorrectValues { values: Vec<f64>},                         // If 1D, the vector/row matrix values dont sum to 1
    RowsIncorrectValues { rows: Vec<usize>, values: Vec<Vec<f64>> },   // Rows are incorrect, doesn't sum to 1
    MatrixEmpty,                                                       // Entire matrix sums to 0
    RowsEmpty { rows: Vec<usize> },                                    // Rows are empty 
    InvalidValue                                                      // There is at least one value outside of [0.0, 1.0]
}

#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    pub matrix: StateMatrix2D<f64>,  // Use StateMatrix2D instead of Vec<Vec<f64>>
}

impl TransitionMatrix {
    // Create a new transition matrix from full matrix input
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        Self {
            matrix: StateMatrix2D::new(matrix),  
        }
    }

    // Create a new transition matrix from StateMatrix2D input
    pub fn from_state_matrix(matrix: StateMatrix2D<f64>) -> Self {
        Self {
            matrix,
        }
    }

    // Generate a random transition matrix
    pub fn new_random(num_states: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(num_states);

        for _ in 0..num_states {
            let mut row: Vec<f64> = Vec::with_capacity(num_states);
            let mut total: f64 = 1.0;

            for _ in 0..(num_states - 1) {
                let val = rng.gen_range(0.0..total);
                row.push(val);
                total -= val;
            }
            
            row.push(total);
            matrix.push(row);
        }

        Self {
            matrix: StateMatrix2D::new(matrix)
        }
    }

    // Generate a transition matrix with all equal probabilities
    pub fn new_balanced(num_states: usize) -> Self {
        let value = 1.0 / (num_states as f64);
        let matrix = vec![vec![value; num_states]; num_states];

        Self {
            matrix: StateMatrix2D::new(matrix)
        }
    }

    // Create an empty transition matrix
    pub fn empty(size: usize) -> Self {
        Self {
            matrix: StateMatrix2D::empty((size, size)),  // Create an empty StateMatrix2D
        }
    }

    // Get probability for a given transition
    pub fn get<T: IDTarget>(&self, from: T, to: T) -> f64 {
        self.matrix.get((from.get_id(), to.get_id())).clone()  // Access StateMatrix2D
    }

    // Set the probability of transitioning from one state to another
    pub fn set<T: IDTarget>(&mut self, from_state: T, to_state: T, prob: f64) {
        self.matrix.set((from_state.get_id(), to_state.get_id()), prob);  // Modify StateMatrix2D
    }

    // Remove a state from the matrix
    pub fn remove_states<T: IDTarget>(&mut self, states: &[T]) {
        let curr_num = self.matrix.shape().0;
        let new_num = curr_num - states.len();
        let mut new_mat: StateMatrix2D<f64> = StateMatrix2D::empty((new_num, new_num));
        
        // Set for fast lookup of states to be removed
        let states_set: HashSet<usize> = states.iter().map(|s| s.get_id()).collect();

        let mut i = 0_usize;
        for state_from in 0..curr_num {
            if states_set.contains(&state_from) {
                continue; // If this state is to be removed, just go to the next one
            }

            let mut j = 0_usize;
            for state_to in 0..curr_num {
                if states_set.contains(&state_to) {
                    continue; // If this state is to be removed, just go to the next one
                }

                // Fill in the new reduced matrix
                new_mat[(i, j)] = self.matrix[(state_from, state_to)];
                j += 1;
            }
            i += 1;
        }

        // Normalize rows
        for row in 0..new_num {
            let sum: f64 = new_mat[row].iter().sum();
            if sum > 0.0 {
                for val in new_mat[row].iter_mut() {
                    *val /= sum;
                }
            }
        }

        self.matrix = new_mat;
    }
}

impl ProbabilityMatrix for TransitionMatrix {

    // Check if the matrix is valid
    fn validate(&self) -> Result<(), MatrixValidationError> {
        if self.matrix.len() != self.matrix[0].len() {
            return Err(MatrixValidationError::IncorrectShape)
        }
        let mut empty_rows = Vec::<usize>::new();
        let mut incorrect_rows_id = Vec::<usize>::new();
        let mut incorrect_rows_values = Vec::<Vec<f64>>::new();
    
        for (i, row) in self.matrix.iter().enumerate() {

            let margin = 1e-4;
            // Check if values are valid
            if row.iter().any(|&val| val < 0.0 || val > 1.0 + margin) {
                return Err(MatrixValidationError::InvalidValue);
            }

            // Check if the row is empty
            if row.iter().all(|&val| val == 0.0) {
                empty_rows.push(i);
            }
            // Check if the row doesn't sum to 1.0
            else {
                let sum: f64 = row.iter().sum();
                if (sum - 1.0).abs() > margin {
                    incorrect_rows_id.push(i);
                    incorrect_rows_values.push(row.clone());
                }
            }
        }
    
        // Entire matrix is empty
        if empty_rows.len() == self.matrix.len() {
            return Err(MatrixValidationError::MatrixEmpty);
        }
        // Some rows are empty
        else if !empty_rows.is_empty() {
            return Err(MatrixValidationError::RowsEmpty { rows: empty_rows });
        }
        // Some rows have incorrect values
        else if !incorrect_rows_id.is_empty() {
            return Err(MatrixValidationError::RowsIncorrectValues {
                rows: incorrect_rows_id,
                values: incorrect_rows_values,
            });
        }
    
        // Matrix is valid
        Ok(())
    }
}

// Implement Index trait for not mut
impl<T: IDTarget> Index<(T, T)> for TransitionMatrix {
    type Output = f64;

    fn index(&self, index: (T, T)) -> &Self::Output {
        let (from, to) = (index.0.get_id(), index.1.get_id());
        &self.matrix[from][to]
    }
}

// Implement IndexMut trait for mut
impl<T: IDTarget> IndexMut<(T, T)> for TransitionMatrix {
    fn index_mut(&mut self, index: (T, T)) -> &mut Self::Output {
        let (from, to) = (index.0.get_id(), index.1.get_id());
        &mut self.matrix[from][to]
    }
}


impl fmt::Display for TransitionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_states = self.matrix.len();

        // Print header
        write!(f, "{:<10}", "From/To")?;
        for to_state_id in 0..num_states {
            write!(f, "{:<12}", format!("State {}", to_state_id))?;
        }
        writeln!(f)?;
        
        // Rows
        for (from_state_id, row) in self.matrix.iter().enumerate() {
            write!(f, "{:<10}", format!("State {}", from_state_id))?;
            for &prob in row {
                write!(f, "{:<12.6}", prob)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct StartMatrix {
    pub matrix: Vec<f64>,
}

impl StartMatrix {
    // Create a new start matrix from full matrix input
    pub fn new(matrix: Vec<f64>) -> Self {
        Self { matrix }
    }

    // Create a new random start matrix
    pub fn new_random(num_states: usize) -> Self {

        let mut rng = rand::thread_rng();
        let mut raw_matrix = Vec::with_capacity(num_states);

        let mut total = 1.0;

        for _ in 0..(num_states - 1) {
            let val = rng.gen_range(0.0..total);
            raw_matrix.push(val);
            total -= val;
        }

        raw_matrix.push(total);

        Self { matrix: raw_matrix }
    }

    // Create a new start matrix with all equal start probabilities
    pub fn new_balanced(num_states: usize) -> Self {

        let value = 1.0 / (num_states as f64);
        let matrix = vec![value; num_states];

        Self { matrix }
    }

    // Create an empty start matrix
    pub fn empty(size: usize) -> Self {
        Self {
            matrix: vec![0.0; size],
        }
    }

    // Get probability for a given transition
    pub fn get<T: IDTarget>(&self, state: T) -> f64 {
        self.matrix[state.get_id()]
    }

    // Set the probability of transitioning from one state to another
    pub fn set<T: IDTarget>(&mut self, state: T, prob: f64) {
        self.matrix[state.get_id()] = prob;
    }

    // Remove a state
    pub fn remove_states<T: IDTarget>(&mut self, states: &[T]) {
        let curr_num = self.matrix.len();
        let new_num = curr_num - states.len();
        let mut new_mat = Vec::with_capacity(new_num);

        // Create a HashSet for fast lookup of states to be removed
        let states_set: HashSet<usize> = states.iter().map(|s| s.get_id()).collect();

        // Fill in remaining elements into new matrix
        for (i, value) in self.matrix.iter().enumerate() {
            if !states_set.contains(&i) {
                new_mat.push(*value);
            }
        }

        // Normalize the new matrix
        let sum: f64 = new_mat.iter().sum();
        if sum > 0.0 {
            for value in new_mat.iter_mut() {
                *value /= sum;
            }
        }

        // Replace the old matrix with the new one
        self.matrix = new_mat;
    }
    
}


impl ProbabilityMatrix for StartMatrix {
    // Validate the 1D start matrix
    fn validate(&self) -> Result<(), MatrixValidationError> {

        let margin = 1e-4;
        // Check if all values are valid
        if self.matrix.iter().any(|&val| val < 0.0 || val > 1.0 + margin) {
            return Err(MatrixValidationError::InvalidValue);
        }

        // Check if all elements are zero (i.e., the matrix is empty)
        if self.matrix.iter().all(|&val| val == 0.0) {
            return Err(MatrixValidationError::MatrixEmpty);
        }

        // Check if the values in the matrix sum to 1.0 (with some tolerance for floating-point errors)
        let total_sum: f64 = self.matrix.iter().sum();
        if (total_sum - 1.0).abs() > margin {
            return Err(MatrixValidationError::VectorIncorrectValues {
                values: self.matrix.clone(),
            });
        }

        Ok(())
    }

    
}

// Implement Index trait for not mut
impl<T: IDTarget> Index<T> for StartMatrix {
    type Output = f64;

    fn index(&self, index: T) -> &Self::Output {
        &self.matrix[index.get_id()]
    }
}

// Implement IndexMut trait for mut
impl<T: IDTarget> IndexMut<T> for StartMatrix {
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.matrix[index.get_id()]
    }
}

impl fmt::Display for StartMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start Matrix (Initial State Probabilities):")?;
        for (state_id, &prob) in self.matrix.iter().enumerate() {
            writeln!(f, "  State {:<3}: Probability = {:.6}", state_id, prob)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_transition_matrix {
    use super::*;

    // Test for creating a new transition matrix from a full matrix
    #[test]
    fn test_transition_matrix_new() {
        let matrix = vec![
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ];
        let transition_matrix = TransitionMatrix::new(matrix.clone());
        assert_eq!(transition_matrix.matrix.raw_matrix, matrix);
    }

    // Test for creating an empty transition matrix
    #[test]
    fn test_transition_matrix_empty() {
        let transition_matrix = TransitionMatrix::empty(3);
        assert_eq!(transition_matrix.matrix.len(), 3);
        assert_eq!(transition_matrix.matrix[0].len(), 3);
        assert!(transition_matrix.matrix.iter().all(|row| row.iter().all(|&val| val == 0.0)));
    }

    // Test for setting and getting values using get and set with usize
    #[test]
    fn test_transition_matrix_get_set_usize() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        transition_matrix.set(0, 1, 0.7);
        assert_eq!(transition_matrix.get(0, 1), 0.7);
    }

    // Test for setting and getting values using get and set with State
    #[test]
    fn test_transition_matrix_get_set_state() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        let state1 = State::new(0, 0.5, 0.1).unwrap();
        let state2 = State::new(1, 0.7, 0.2).unwrap();

        transition_matrix.set(&state1, &state2, 0.6);
        assert_eq!(transition_matrix.get(&state1, &state2), 0.6);
    }

    // Test for validation - matrix is not square (IncorrectShape error)
    #[test]
    fn test_transition_matrix_validation_incorrect_shape() {
        let matrix = vec![
            vec![0.5, 0.5],
            vec![0.3, 0.7],
            vec![0.1, 0.9]
        ];
        let transition_matrix = TransitionMatrix::new(matrix);

        match transition_matrix.validate() {
            Err(MatrixValidationError::IncorrectShape) => (),
            _ => panic!("Expected IncorrectShape error"),
        }
    }

    // Test for validation - valid matrix
    #[test]
    fn test_transition_matrix_validation_valid() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        transition_matrix.set(0, 0, 0.5);
        transition_matrix.set(0, 1, 0.5);
        transition_matrix.set(1, 0, 0.3);
        transition_matrix.set(1, 1, 0.7);
        transition_matrix.set(2, 0, 1.0);

        assert!(transition_matrix.validate().is_ok());
    }

    // Test for validation - matrix empty (MatrixEmpty error)
    #[test]
    fn test_transition_matrix_validation_matrix_empty() {
        let transition_matrix = TransitionMatrix::empty(3);

        match transition_matrix.validate() {
            Err(MatrixValidationError::MatrixEmpty) => (),
            _ => panic!("Expected MatrixEmpty error"),
        }
    }

    // Test for validation - rows empty (RowsEmpty error)
    #[test]
    fn test_transition_matrix_validation_rows_empty() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        transition_matrix.set(0, 0, 0.5);

        match transition_matrix.validate() {
            Err(MatrixValidationError::RowsEmpty { rows }) => {
                assert_eq!(rows, vec![1, 2]);
            },
            _ => panic!("Expected RowsEmpty error"),
        }
    }

    // Test for validation - incorrect row values (RowsIncorrectValues error)
    #[test]
    fn test_transition_matrix_validation_incorrect_rows() {
        let mut transition_matrix = TransitionMatrix::empty(2);
        transition_matrix.set(0, 0, 0.7); 
        transition_matrix.set(0, 1, 0.7); // Incorrect sum in row 0
        transition_matrix.set(1, 0, 0.5);
        transition_matrix.set(1, 1, 0.5);

        match transition_matrix.validate() {
            Err(MatrixValidationError::RowsIncorrectValues { rows, values }) => {
                assert_eq!(rows, vec![0]);
                assert_eq!(values, vec![vec![0.7, 0.7]]); // Expected incorrect row
            },
            _ => panic!("Expected RowsIncorrectValues error"),
        }
    }

    // Test for validation - invalid values (InvalidValue error)
    #[test]
    fn test_transition_matrix_validation_invalid_value() {
        let mut transition_matrix = TransitionMatrix::empty(2);
        
        // Set an invalid value (outside the range 0.0 - 1.0)
        transition_matrix.set(0, 0, 1.5);  // Invalid value > 1.0
        transition_matrix.set(1, 1, -0.2); // Invalid value < 0.0

        match transition_matrix.validate() {
            Err(MatrixValidationError::InvalidValue) => (),
            _ => panic!("Expected InvalidValue error"),
        }
    }

    // Test for indexing using (usize, usize)
    #[test]
    fn test_transition_matrix_indexing_usize() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        transition_matrix[(0, 1)] = 0.7;
        assert_eq!(transition_matrix[(0, 1)], 0.7);
    }

    // Test for indexing using (&State, &State)
    #[test]
    fn test_transition_matrix_indexing_state() {
        let mut transition_matrix = TransitionMatrix::empty(3);
        let state1 = State::new(0, 0.5, 0.1).unwrap();
        let state2 = State::new(1, 0.7, 0.2).unwrap();

        transition_matrix[(&state1, &state2)] = 0.6;
        assert_eq!(transition_matrix[(&state1, &state2)], 0.6);
    }

    // Test the generate random matrix method
    #[test]
    fn test_random_transition_matrix_valid() {
        for _ in 0..1000 {
            let num_states = 3; 
            let transition_matrix = TransitionMatrix::new_random(num_states);
            
            assert!(transition_matrix.validate().is_ok());
        }
    }

    // Test for removing states from matrix
    #[test]
    fn test_transition_matrix_remove_states() {
        let mut transition_matrix = TransitionMatrix::new(vec![
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.8, 0.1],
            vec![0.25, 0.25, 0.5],
        ]);

        let indices_to_remove = vec![1, 2];

        transition_matrix.remove_states(&indices_to_remove);

        
        assert_eq!(transition_matrix.matrix.len(), 1);
        assert_eq!(transition_matrix.matrix[0].len(), 1);
        assert_eq!(transition_matrix.matrix[0][0], 1.0);

        assert!(transition_matrix.validate().is_ok());
    }

}

#[cfg(test)]
mod tests_start_matrix {
    use super::*;

    // Test for creating a new start matrix from a full matrix
    #[test]
    fn test_start_matrix_new() {
        let matrix = vec![0.5, 0.3, 0.2];
        let start_matrix = StartMatrix::new(matrix.clone());
        assert_eq!(start_matrix.matrix, matrix);
    }

    // Test for creating an empty start matrix
    #[test]
    fn test_start_matrix_empty() {
        let start_matrix = StartMatrix::empty(3);
        assert_eq!(start_matrix.matrix.len(), 3);
        assert!(start_matrix.matrix.iter().all(|&val| val == 0.0));
    }

    // Test for setting and getting values using get and set with usize
    #[test]
    fn test_start_matrix_get_set_usize() {
        let mut start_matrix = StartMatrix::empty(3);
        start_matrix.set(0, 0.7);
        assert_eq!(start_matrix.get(0), 0.7);
    }

    // Test for setting and getting values using get and set with State
    #[test]
    fn test_start_matrix_get_set_state() {
        let mut start_matrix = StartMatrix::empty(3);
        let state1 = State::new(0, 0.5, 0.1).unwrap();
        let state2 = State::new(1, 0.7, 0.2).unwrap();

        start_matrix.set(&state1, 0.6);
        assert_eq!(start_matrix.get(&state1), 0.6);

        start_matrix.set(&state2, 0.4);
        assert_eq!(start_matrix.get(&state2), 0.4);
    }

    // Test for validation - valid start matrix
    #[test]
    fn test_start_matrix_validation_valid() {
        let mut start_matrix = StartMatrix::empty(3);
        start_matrix.set(0, 0.5);
        start_matrix.set(1, 0.3);
        start_matrix.set(2, 0.2);

        assert!(start_matrix.validate().is_ok());
    }

    // Test for validation - matrix empty (MatrixEmpty error)
    #[test]
    fn test_start_matrix_validation_matrix_empty() {
        let start_matrix = StartMatrix::empty(3);

        match start_matrix.validate() {
            Err(MatrixValidationError::MatrixEmpty) => (),
            _ => panic!("Expected MatrixEmpty error"),
        }
    }

    // Test for validation - vector incorrect values (VectorIncorrectValues error)
    #[test]
    fn test_start_matrix_validation_incorrect_values() {
        let start_matrix = StartMatrix::new(vec![0.5, 0.4, 0.4]); // Sum is greater than 1.0

        match start_matrix.validate() {
            Err(MatrixValidationError::VectorIncorrectValues { values }) => {
                assert_eq!(values, vec![0.5, 0.4, 0.4]);
            },
            _ => panic!("Expected VectorIncorrectValues error"),
        }
    }

    // Test for validation - invalid values (InvalidValue error)
    #[test]
    fn test_start_matrix_validation_invalid_value() {
        let start_matrix = StartMatrix::new(vec![1.5, 0.5, -0.5]); // Invalid values: > 1.0 and < 0.0

        match start_matrix.validate() {
            Err(MatrixValidationError::InvalidValue) => (),
            _ => panic!("Expected InvalidValue error"),
        }
    }

    // Test for indexing using usize
    #[test]
    fn test_start_matrix_indexing_usize() {
        let mut start_matrix = StartMatrix::empty(3);
        start_matrix[0] = 0.7;
        assert_eq!(start_matrix[0], 0.7);
    }

    // Test for indexing using (&State)
    #[test]
    fn test_start_matrix_indexing_state() {
        let mut start_matrix = StartMatrix::empty(3);
        let state = State::new(0, 0.5, 0.1).unwrap();

        start_matrix[&state] = 0.6;
        assert_eq!(start_matrix[&state], 0.6);
    }


    // Test for removing states from matrix
    #[test]
    fn test_remove_states_from_start_matrix() {
        let mut start_matrix = StartMatrix::new(vec![0.4, 0.3, 0.2, 0.1]);

        // Remove states at indices 1 and 2
        let indices_to_remove = vec![1, 2];

        start_matrix.remove_states(&indices_to_remove);

        // After removing states 1 and 2, we should have [0.4, 0.1]
        // After normalization, it should become [0.8, 0.2]
        assert_eq!(start_matrix.matrix.len(), 2);
        assert!((start_matrix.matrix[0] - 0.8).abs() < 1e-4);
        assert!((start_matrix.matrix[1] - 0.2).abs() < 1e-4);

        // Validate the new start matrix
        assert!(start_matrix.validate().is_ok());
    }
}