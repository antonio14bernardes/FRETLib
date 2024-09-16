use super::state::*;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct TransitionMatrix {
    matrix: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    // Create a new transition matrix from full matrix input
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        Self { matrix }
    }

    // Create an empty transition matrix
    pub fn empty(size: usize) -> Self {
        Self {
            matrix: vec![vec![0.0; size]; size],
        }
    }

    // Get probability for a given transition
    pub fn get<T: IDTarget>(&self, from: T, to: T) -> f64 {
        self.matrix[from.get_id()][to.get_id()]
    }

    // Set the probability of transitioning from one state to another
    pub fn set<T: IDTarget>(&mut self, from_state: T, to_state: T, prob: f64) {
        self.matrix[from_state.get_id()][to_state.get_id()] = prob;
    }

    // Check if the matrix is valid
    pub fn validate(&self) -> Result<(), MatrixValidationError> {
        if self.matrix.len() != self.matrix[0].len() {
            return Err(MatrixValidationError::IncorrectShape)
        }
        let mut empty_rows = Vec::<usize>::new();
        let mut incorrect_rows_id = Vec::<usize>::new();
        let mut incorrect_rows_values = Vec::<Vec<f64>>::new();
    
        for (i, row) in self.matrix.iter().enumerate() {
            // Check if row is empty
            let sum: f64 = row.iter().sum();
    
            if sum == 0.0 {
                empty_rows.push(i);
            } else if (sum - 1.0).abs() > 1e-4 {
                incorrect_rows_id.push(i);
                incorrect_rows_values.push(row.clone());
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

// Define custom error for matrix value validation
pub enum MatrixValidationError {
    IncorrectShape,
    RowsIncorrectValues { rows: Vec<usize>, values: Vec<Vec<f64>> },   // Rows are incorrect, doesn't sum to 1
    MatrixEmpty,                                                 // Entire matrix sums to 0
    RowsEmpty { rows: Vec<usize> },                                     // Rows are empty 
}

// Implement Index trait for read-only access
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

// Implement IndexMut trait for mutable access
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

#[cfg(test)]
mod tests {
    use super::*;

    // Test for creating a new transition matrix from a full matrix
    #[test]
    fn test_transition_matrix_new() {
        let matrix = vec![
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ];
        let transition_matrix = TransitionMatrix::new(matrix.clone());
        assert_eq!(transition_matrix.matrix, matrix);
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
        let state1 = State::new(0, 0.5, 0.1);
        let state2 = State::new(1, 0.7, 0.2);

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
        transition_matrix.set(0, 0, 1.2); // Incorrect sum in row 0
        transition_matrix.set(1, 0, 0.5);
        transition_matrix.set(1, 1, 0.5);

        match transition_matrix.validate() {
            Err(MatrixValidationError::RowsIncorrectValues { rows, values }) => {
                assert_eq!(rows, vec![0]);
                assert_eq!(values, vec![vec![1.2, 0.0]]); // Expected incorrect row
            },
            _ => panic!("Expected RowsIncorrectValues error"),
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
        let state1 = State::new(0, 0.5, 0.1);
        let state2 = State::new(1, 0.7, 0.2);

        transition_matrix[(&state1, &state2)] = 0.6;
        assert_eq!(transition_matrix[(&state1, &state2)], 0.6);
    }
}