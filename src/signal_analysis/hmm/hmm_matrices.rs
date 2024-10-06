use rand::Rng;

use super::state::*;
use super::hmm_tools::*;
use std::ops::{Index, IndexMut};

pub trait ProbabilityMatrix {

    // Check if the matrix is valid
    fn validate(&self) -> Result<(), MatrixValidationError>;
}

// Custom error for matrix value validation
#[derive(Debug)]
pub enum MatrixValidationError {
    IncorrectShape,                                                    // If 2D, num of rows and columns must be the same
    VectorIncorrectValues { values: Vec<f64>},                         // If 1D, the vector/row matrix values dont sum to 1
    RowsIncorrectValues { rows: Vec<usize>, values: Vec<Vec<f64>> },   // Rows are incorrect, doesn't sum to 1
    MatrixEmpty,                                                       // Entire matrix sums to 0
    RowsEmpty { rows: Vec<usize> },                                    // Rows are empty 
    InvalidValue                                                       // There is at least one value outside of [0.0, 1.0]
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
impl Index<(usize, usize)> for TransitionMatrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (from, to) = index;
        &self.matrix[from][to]
    }
}

impl Index<(&State, &State)> for TransitionMatrix {
    type Output = f64;

    fn index(&self, index: (&State, &State)) -> &Self::Output {
        let (from, to) = (index.0.id, index.1.id);
        &self.matrix[from][to]
    }
}

// Implement IndexMut trait for mut
impl IndexMut<(usize, usize)> for TransitionMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (from, to) = index;
        &mut self.matrix[from][to]
    }
}

impl IndexMut<(&State, &State)> for TransitionMatrix {
    fn index_mut(&mut self, index: (&State, &State)) -> &mut Self::Output {
        let (from, to) = (index.0.id, index.1.id);
        &mut self.matrix[from][to]
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
impl Index<usize> for StartMatrix {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {

        &self.matrix[index]
    }
}

impl Index<&State> for StartMatrix {
    type Output = f64;

    fn index(&self, state: &State) -> &Self::Output {
        
        &self.matrix[state.id]
    }
}

// Implement IndexMut trait for mut
impl IndexMut<usize> for StartMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        
        &mut self.matrix[index]
    }
}

impl IndexMut<&State> for StartMatrix {
    fn index_mut(&mut self, state: &State) -> &mut Self::Output {
        &mut self.matrix[state.id]
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
        let mut start_matrix = StartMatrix::new(vec![0.5, 0.4, 0.4]); // Sum is greater than 1.0

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
        let mut start_matrix = StartMatrix::new(vec![1.5, 0.5, -0.5]); // Invalid values: > 1.0 and < 0.0

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
}