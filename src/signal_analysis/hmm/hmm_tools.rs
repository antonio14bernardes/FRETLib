// use std::ops::{Index, IndexMut};

// pub struct State {
//     pub id: usize,
//     pub value: f64,
//     pub noise_std: f64,
//     pub name: Option<String>,
// }

// impl State {
//     // Constructor to create a new State.
//     // Name is always set to None when isntantiating
//     // Use method set_name to give the state a cool name
//     pub fn new(id: usize, value: f64, noise_std: f64) -> Self {
//         State {
//             id,
//             value,
//             noise_std,
//             name: None,
//         }
//     }

//     // Method to baptize the state with a legit name
//     pub fn set_name(&mut self, name: String) {
//         self.name = Some(name);
//     }

//     // Method to calculate the  emission probability based on an observed FRET value using
//     // the standard Gaussian
//     pub fn standard_gaussian_emission_probability(&self, observed_fret: f64) -> f64 {
//         let variance = self.noise_std * self.noise_std;
//         let exponent = -(observed_fret - self.value).powi(2) / (2.0 * variance);
//         (1.0 / (self.noise_std * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
//     }

//     // Method to calculate the  emission probability based on an observed FRET value using
//     // the simplified Gaussian discussed in the paper (see hmm module root file)
//     pub fn paper_gaussian_emission_probability(&self, observed_fret: f64) -> f64 {
//         let scaled_diff = (observed_fret - self.value) / self.noise_std;
//         (-2.0 * scaled_diff.powi(2)).exp()
//     }
// }

// pub trait IDTarget {
//     fn get_id(&self) -> usize;
// }

// impl IDTarget for usize {
//     fn get_id(&self) -> usize {
//         *self  // Simply return the value of usize
//     }
// }

// impl IDTarget for State {
//     fn get_id(&self) -> usize {
//         self.id  // Return the ID field from the State struct
//     }
// }


// pub struct TransitionMatrix {
//     matrix: Vec<Vec<f64>>,
// }

// impl TransitionMatrix {
//     // Create a new transition matrix from full matrix input
//     pub fn new(matrix: Vec<Vec<f64>>) -> Self {
//         Self { matrix }
//     }

//     // Create an empty transition matrix
//     pub fn empty(size: usize) -> Self {
//         Self {
//             matrix: vec![vec![0.0; size]; size],
//         }
//     }

//     // Get probability for a given transition
//     pub fn get<T: IDTarget>(&self, from: T, to: T) -> f64 {
//         self.matrix[from.get_id()][to.get_id()]
//     }

//     // Set the probability of transitioning from one state to another
//     pub fn set<T: IDTarget>(&mut self, from_state: T, to_state: T, prob: f64) {
//         self.matrix[from_state.get_id()][to_state.get_id()] = prob;
//     }

//     // Check if the matrix is valid
//     pub fn validate(&self) -> Result<(), MatrixValidationError> {
//         let mut empty_rows = Vec::<usize>::new();
//         let mut incorrect_rows_id = Vec::<usize>::new();
//         let mut incorrect_rows_values = Vec::<Vec<f64>>::new();
    
//         for (i, row) in self.matrix.iter().enumerate() {
//             // Check if row is empty
//             let sum: f64 = row.iter().sum();
    
//             if sum == 0.0 {
//                 empty_rows.push(i);
//             } else if (sum - 1.0).abs() > 1e-4 {
//                 incorrect_rows_id.push(i);
//                 incorrect_rows_values.push(row.clone());
//             }
//         }
    
//         // Entire matrix is empty
//         if empty_rows.len() == self.matrix.len() {
//             return Err(MatrixValidationError::MatrixEmpty);
//         }
//         // Some rows are empty
//         else if !empty_rows.is_empty() {
//             return Err(MatrixValidationError::RowsEmpty { rows: empty_rows });
//         }
//         // Some rows have incorrect values
//         else if !incorrect_rows_id.is_empty() {
//             return Err(MatrixValidationError::RowsIncorrectValues {
//                 rows: incorrect_rows_id,
//                 values: incorrect_rows_values,
//             });
//         }
    
//         // Matrix is valid
//         Ok(())
//     }
// }

// // Define custom error for matrix value validation
// pub enum MatrixValidationError {
//     RowsIncorrectValues { rows: Vec<usize>, values: Vec<Vec<f64>> },   // Rows are incorrect, doesn't sum to 1
//     MatrixEmpty,                                                 // Entire matrix sums to 0
//     RowsEmpty { rows: Vec<usize> },                                     // Rows are empty 
// }

// // Implement Index trait for read-only access
// impl Index<(usize, usize)> for TransitionMatrix {
//     type Output = f64;

//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         let (from, to) = index;
//         &self.matrix[from][to]
//     }
// }

// impl Index<(&State, &State)> for TransitionMatrix {
//     type Output = f64;

//     fn index(&self, index: (&State, &State)) -> &Self::Output {
//         let (from, to) = (index.0.id, index.1.id);
//         &self.matrix[from][to]
//     }
// }

// // Implement IndexMut trait for mutable access
// impl IndexMut<(usize, usize)> for TransitionMatrix {
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         let (from, to) = index;
//         &mut self.matrix[from][to]
//     }
// }

// impl IndexMut<(&State, &State)> for TransitionMatrix {
//     fn index_mut(&mut self, index: (&State, &State)) -> &mut Self::Output {
//         let (from, to) = (index.0.id, index.1.id);
//         &mut self.matrix[from][to]
//     }
// }

