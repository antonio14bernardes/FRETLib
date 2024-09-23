use std::ops::{Index, IndexMut};
use super::state::*;



/*
Structs for State related Matrices.
If these end up not needing any specific intrinsic validation, they 
can be replaced by Vecs with implemented Index<(&State)> etc.
*/
#[derive(Debug, Clone)]
pub struct StateMatrix1D<T> {
    pub raw_matrix: Vec<T>,
}

impl<T> StateMatrix1D<T> {
    // Create new from a full matrix input
    pub fn new(raw_matrix: Vec<T>) -> Self {
        Self { raw_matrix }
    }

    // Create an empty matrix
    pub fn empty(size: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            raw_matrix: vec![T::default(); size],
        }
    }

    // Obtain secret knowledge from this magical matrix
    pub fn get(&self, index: usize) -> &T {
        &self.raw_matrix[index]
    }

    // Sneak a value into it
    pub fn set(&mut self, index: usize, value: T) {
        self.raw_matrix[index] = value;
    }

    // Get ref to full matrix
    pub fn get_matrix(&self) -> &Vec<T> {
        &self.raw_matrix
    }

    // Get mut ref to full matrix
    pub fn get_mut_matrix(&mut self) -> &mut Vec<T> {
        &mut self.raw_matrix
    }

    // Get len
    pub fn len(&self) -> usize {
        self.raw_matrix.len()
    }

    // Get as iterator
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.raw_matrix.iter()
    }
}

#[derive(Debug, Clone)]
pub struct StateMatrix2D<T> {
    pub raw_matrix: Vec<Vec<T>>,
}

impl<T> StateMatrix2D<T> {
    // Create new from a full matrix input
    pub fn new(raw_matrix: Vec<Vec<T>>) -> Self {
        Self { raw_matrix }
    }

    // Create an empty matrix
    pub fn empty(size: (usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        Self {
            raw_matrix: vec![vec![T::default(); size.1]; size.0],
        }
    }

    // Obtain secret knowledge from this magical matrix
    pub fn get(&self, index: (usize, usize)) -> &T {
        &self.raw_matrix[index.0][index.1]
    }

    // Sneak a value into it
    pub fn set(&mut self, index: (usize, usize), value: T) {
        self.raw_matrix[index.0][index.1] = value;
    }

    // Get ref to full matrix
    pub fn get_matrix(&self) -> &Vec<Vec<T>> {
        &self.raw_matrix
    }

    // Get mut ref to full matrix
    pub fn get_mut_matrix(&mut self) -> &mut Vec<Vec<T>> {
        &mut self.raw_matrix
    }

    // Get len
    pub fn len(&self) -> usize {
        self.raw_matrix.len()
    }

    // Get shape
    pub fn shape(&self) -> (usize, usize) {
        let rows = self.raw_matrix.len();
        let cols = if rows > 0 { self.raw_matrix[0].len() } else { 0 };
        (rows, cols)
    }

    // Get as iterator
    pub fn iter(&self) -> std::slice::Iter<'_, Vec<T>> {
        self.raw_matrix.iter()
    }
}

// Implement Index trait for not mut
impl<T> Index<usize> for StateMatrix1D<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.raw_matrix[index]
    }
}

impl<T> Index<&State> for StateMatrix1D<T> {
    type Output = T;

    fn index(&self, state: &State) -> &Self::Output {
        &self.raw_matrix[state.get_id()]
    }
}

impl<T> Index<(usize, usize)> for StateMatrix2D<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.raw_matrix[index.0][index.1]
    }
}

impl<T> Index<(&State, &State)> for StateMatrix2D<T> {
    type Output = T;

    fn index(&self, states: (&State, &State)) -> &Self::Output {
        &self.raw_matrix[states.0.get_id()][states.1.get_id()]
    }
}

impl<T> Index<(&State, usize)> for StateMatrix2D<T> {
    type Output = T;

    fn index(&self, index: (&State, usize)) -> &Self::Output {
        &self.raw_matrix[index.0.get_id()][index.1]
    }
}

impl<T> Index<usize> for StateMatrix2D<T> {
    type Output = Vec<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.raw_matrix[index]
    }
}

impl<T> Index<&State> for StateMatrix2D<T> {
    type Output = Vec<T>;

    fn index(&self, state: &State) -> &Self::Output {
        &self.raw_matrix[state.get_id()]
    }
}


// Implement IndexMut trait for mut
impl<T> IndexMut<usize> for StateMatrix1D<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.raw_matrix[index]
    }
}

impl<T> IndexMut<&State> for StateMatrix1D<T> {
    fn index_mut(&mut self, state: &State) -> &mut Self::Output {
        &mut self.raw_matrix[state.get_id()]
    }
}

impl<T> IndexMut<(usize, usize)> for StateMatrix2D<T> {

    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.raw_matrix[index.0][index.1]
    }
}

impl<T> IndexMut<(&State, &State)> for StateMatrix2D<T> {

    fn index_mut(&mut self, states: (&State, &State)) -> &mut Self::Output {
        &mut self.raw_matrix[states.0.get_id()][states.1.get_id()]
    }
}

impl<T> IndexMut<(&State, usize)> for StateMatrix2D<T> {

    fn index_mut(&mut self, index: (&State, usize)) -> &mut Self::Output {
        &mut self.raw_matrix[index.0.get_id()][index.1]
    }
}

impl<T> IndexMut<usize> for StateMatrix2D<T> {

    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.raw_matrix[index]
    }
}

impl<T> IndexMut<&State> for StateMatrix2D<T> {

    fn index_mut(&mut self, state: &State) -> &mut Self::Output {
        &mut self.raw_matrix[state.get_id()]
    }
}


#[derive(Debug, Clone)]
pub struct StateMatrix3D<T> {
    pub raw_matrix: Vec<Vec<Vec<T>>>,  // 3D matrix
}

impl<T> StateMatrix3D<T> {
    // Create new from a full matrix input
    pub fn new(raw_matrix: Vec<Vec<Vec<T>>>) -> Self {
        Self { raw_matrix }
    }

    // Create an empty matrix
    pub fn empty(size: (usize, usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        Self {
            raw_matrix: vec![vec![vec![T::default(); size.2]; size.1]; size.0],
        }
    }

    // Obtain secret knowledge from this magical matrix
    pub fn get(&self, index: (usize, usize, usize)) -> &T {
        &self.raw_matrix[index.0][index.1][index.2]
    }

    // Sneak a value into it
    pub fn set(&mut self, index: (usize, usize, usize), value: T) {
        self.raw_matrix[index.0][index.1][index.2] = value;
    }

    // Get ref to full matrix
    pub fn get_matrix(&self) -> &Vec<Vec<Vec<T>>> {
        &self.raw_matrix
    }

    // Get mut ref to full matrix
    pub fn get_mut_matrix(&mut self) -> &mut Vec<Vec<Vec<T>>> {
        &mut self.raw_matrix
    }

    // Get len
    pub fn len(&self) -> usize {
        self.raw_matrix.len()
    }

    // Get shape
    pub fn shape(&self) -> (usize, usize, usize) {
        let rows = self.raw_matrix.len();
        let cols = if rows > 0 { self.raw_matrix[0].len() } else { 0 };
        let depth = if cols > 0 { self.raw_matrix[0][0].len() } else { 0 };
        (rows, cols, depth)
    }

    // Get as iterator
    pub fn iter(&self) -> std::slice::Iter<'_, Vec<Vec<T>>> {
        self.raw_matrix.iter()
    }
}

// Implement Index trait for non-mutable access
impl<T> Index<(usize, usize, usize)> for StateMatrix3D<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.raw_matrix[index.0][index.1][index.2]
    }
}

impl<T> Index<(&State, &State, usize)> for StateMatrix3D<T> {
    type Output = T;

    fn index(&self, states: (&State, &State, usize)) -> &Self::Output {
        &self.raw_matrix[states.0.get_id()][states.1.get_id()][states.2]
    }
}

// Implement IndexMut trait for mutable access
impl<T> IndexMut<(usize, usize, usize)> for StateMatrix3D<T> {

    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.raw_matrix[index.0][index.1][index.2]
    }
}

impl<T> IndexMut<(&State, &State, usize)> for StateMatrix3D<T> {

    fn index_mut(&mut self, states: (&State, &State, usize)) -> &mut Self::Output {
        &mut self.raw_matrix[states.0.get_id()][states.1.get_id()][states.2]
    }
}


#[cfg(test)]
mod tests_state_matrix_1d {
    use super::*;

    #[test]
    fn test_state_matrix_1d_new() {
        let matrix = vec![1, 2, 3];
        let state_matrix = StateMatrix1D::new(matrix.clone());
        assert_eq!(state_matrix.raw_matrix, matrix);
    }

    #[test]
    fn test_state_matrix_1d_empty() {
        let state_matrix: StateMatrix1D<i32> = StateMatrix1D::empty(3);
        assert_eq!(state_matrix.raw_matrix.len(), 3);
        assert!(state_matrix.raw_matrix.iter().all(|&val| val == 0));
    }

    #[test]
    fn test_state_matrix_1d_get_set_usize() {
        let mut state_matrix = StateMatrix1D::empty(3);
        state_matrix.set(0, 42);
        assert_eq!(state_matrix.get(0), &42);
    }

    #[test]
    fn test_state_matrix_1d_get_set_state() {
        let mut state_matrix = StateMatrix1D::empty(3);
        let state = State::new(1, 0.5, 0.1);
        state_matrix.set(1, 99);
        assert_eq!(state_matrix.get(state.get_id()), &99);
    }

    #[test]
    fn test_state_matrix_1d_index_usize() {
        let mut state_matrix = StateMatrix1D::empty(3);
        state_matrix[0] = 100;
        assert_eq!(state_matrix[0], 100);
    }

    #[test]
    fn test_state_matrix_1d_index_state() {
        let mut state_matrix = StateMatrix1D::empty(3);
        let state = State::new(2, 0.5, 0.1);
        state_matrix[&state] = 200;
        assert_eq!(state_matrix[&state], 200);
    }

    #[test]
    fn test_state_matrix_1d_len() {
        let state_matrix: StateMatrix1D<i32> = StateMatrix1D::empty(5);
        assert_eq!(state_matrix.len(), 5);
    }

    #[test]
    fn test_state_matrix_1d_iter() {
        let state_matrix = StateMatrix1D::new(vec![10, 20, 30]);
        let mut iter = state_matrix.iter();
        assert_eq!(iter.next(), Some(&10));
        assert_eq!(iter.next(), Some(&20));
        assert_eq!(iter.next(), Some(&30));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_state_matrix_1d_get_matrix() {
        let matrix = vec![1, 2, 3];
        let state_matrix = StateMatrix1D::new(matrix.clone());
        assert_eq!(state_matrix.get_matrix(), &matrix);
    }

    #[test]
    fn test_state_matrix_1d_get_mut_matrix() {
        let matrix = vec![1, 2, 3];
        let mut state_matrix = StateMatrix1D::new(matrix.clone());
        state_matrix.get_mut_matrix()[0] = 99;
        assert_eq!(state_matrix.get_matrix()[0], 99);
    }
}

#[cfg(test)]
mod tests_state_matrix_2d {
    use super::*;

    #[test]
    fn test_state_matrix_2d_new() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let state_matrix = StateMatrix2D::new(matrix.clone());
        assert_eq!(state_matrix.raw_matrix, matrix);
    }

    #[test]
    fn test_state_matrix_2d_empty() {
        let state_matrix: StateMatrix2D<i32> = StateMatrix2D::empty((2, 2));
        assert_eq!(state_matrix.raw_matrix.len(), 2);
        assert_eq!(state_matrix.raw_matrix[0].len(), 2);
        assert!(state_matrix.raw_matrix.iter().all(|row| row.iter().all(|&val| val == 0)));
    }

    #[test]
    fn test_state_matrix_2d_get_set_usize() {
        let mut state_matrix = StateMatrix2D::empty((2, 2));
        state_matrix.set((0, 1), 42);
        assert_eq!(state_matrix.get((0, 1)), &42);
    }

    #[test]
    fn test_state_matrix_2d_get_set_states() {
        let mut state_matrix = StateMatrix2D::empty((3, 3));
        let state1 = State::new(1, 0.5, 0.1);
        let state2 = State::new(2, 0.7, 0.2);
        state_matrix.set((state1.id, state2.id), 99);
        assert_eq!(state_matrix.get((state1.id, state2.id)), &99);
    }

    #[test]
    fn test_state_matrix_2d_index_usize() {
        let mut state_matrix = StateMatrix2D::empty((2, 2));
        state_matrix[(0, 1)] = 100;
        assert_eq!(state_matrix[(0, 1)], 100);
    }

    #[test]
    fn test_state_matrix_2d_index_states() {
        let mut state_matrix = StateMatrix2D::empty((3, 3));
        let state1 = State::new(1, 0.5, 0.1);
        let state2 = State::new(2, 0.7, 0.2);
        state_matrix[(&state1, &state2)] = 200;
        assert_eq!(state_matrix[(&state1, &state2)], 200);
    }

    #[test]
    fn test_state_matrix_2d_len() {
        let state_matrix: StateMatrix2D<i32> = StateMatrix2D::empty((3, 3));
        assert_eq!(state_matrix.len(), 3);
    }

    #[test]
    fn test_state_matrix_2d_shape() {
        let state_matrix: StateMatrix2D<i32> = StateMatrix2D::empty((3, 4));
        assert_eq!(state_matrix.shape(), (3, 4));
    }

    #[test]
    fn test_state_matrix_2d_iter() {
        let state_matrix = StateMatrix2D::new(vec![vec![10, 20], vec![30, 40]]);
        let mut iter = state_matrix.iter();
        assert_eq!(iter.next(), Some(&vec![10, 20]));
        assert_eq!(iter.next(), Some(&vec![30, 40]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_state_matrix_2d_get_matrix() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let state_matrix = StateMatrix2D::new(matrix.clone());
        assert_eq!(state_matrix.get_matrix(), &matrix);
    }

    #[test]
    fn test_state_matrix_2d_get_mut_matrix() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let mut state_matrix = StateMatrix2D::new(matrix.clone());
        state_matrix.get_mut_matrix()[0][0] = 99;
        assert_eq!(state_matrix.get_matrix()[0][0], 99);
    }
}

mod tests_state_matrix_3d {
    use super::*;

    #[test]
    fn test_state_matrix_3d_new() {
        let matrix = vec![
            vec![vec![1, 2], vec![3, 4]],
            vec![vec![5, 6], vec![7, 8]],
        ];
        let state_matrix = StateMatrix3D::new(matrix.clone());
        assert_eq!(state_matrix.raw_matrix, matrix);
    }

    #[test]
    fn test_state_matrix_3d_empty() {
        let state_matrix: StateMatrix3D<i32> = StateMatrix3D::empty((2, 2, 2));
        assert_eq!(state_matrix.raw_matrix.len(), 2);
        assert_eq!(state_matrix.raw_matrix[0].len(), 2);
        assert_eq!(state_matrix.raw_matrix[0][0].len(), 2);
        assert!(state_matrix.raw_matrix.iter().all(|layer| {
            layer.iter().all(|row| row.iter().all(|&val| val == 0))
        }));
    }

    #[test]
    fn test_state_matrix_3d_get_set_usize() {
        let mut state_matrix = StateMatrix3D::empty((2, 2, 2));
        state_matrix.set((0, 1, 1), 42);
        assert_eq!(state_matrix.get((0, 1, 1)), &42);
    }

    #[test]
    fn test_state_matrix_3d_get_set_states() {
        let mut state_matrix = StateMatrix3D::empty((3, 3, 2));
        let state1 = State::new(1, 0.5, 0.1);
        let state2 = State::new(2, 0.7, 0.2);
        state_matrix.set((state1.id, state2.id, 1), 99);
        assert_eq!(state_matrix.get((state1.id, state2.id, 1)), &99);
    }

    #[test]
    fn test_state_matrix_3d_index_usize() {
        let mut state_matrix = StateMatrix3D::empty((2, 2, 2));
        state_matrix[(0, 1, 1)] = 100;
        assert_eq!(state_matrix[(0, 1, 1)], 100);
    }

    #[test]
    fn test_state_matrix_3d_index_states() {
        let mut state_matrix = StateMatrix3D::empty((3, 3, 2));
        let state1 = State::new(1, 0.5, 0.1);
        let state2 = State::new(2, 0.7, 0.2);
        state_matrix[(&state1, &state2, 1)] = 200;
        assert_eq!(state_matrix[(&state1, &state2, 1)], 200);
    }

    #[test]
    fn test_state_matrix_3d_len() {
        let state_matrix: StateMatrix3D<i32> = StateMatrix3D::empty((3, 3, 3));
        assert_eq!(state_matrix.len(), 3);
    }

    #[test]
    fn test_state_matrix_3d_shape() {
        let state_matrix: StateMatrix3D<i32> = StateMatrix3D::empty((3, 4, 2));
        assert_eq!(state_matrix.shape(), (3, 4, 2));
    }

    #[test]
    fn test_state_matrix_3d_iter() {
        let state_matrix = StateMatrix3D::new(vec![
            vec![vec![10, 20], vec![30, 40]],
            vec![vec![50, 60], vec![70, 80]],
        ]);
        let mut iter = state_matrix.iter();
        assert_eq!(iter.next(), Some(&vec![vec![10, 20], vec![30, 40]]));
        assert_eq!(iter.next(), Some(&vec![vec![50, 60], vec![70, 80]]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_state_matrix_3d_get_matrix() {
        let matrix = vec![
            vec![vec![1, 2], vec![3, 4]],
            vec![vec![5, 6], vec![7, 8]],
        ];
        let state_matrix = StateMatrix3D::new(matrix.clone());
        assert_eq!(state_matrix.get_matrix(), &matrix);
    }

    #[test]
    fn test_state_matrix_3d_get_mut_matrix() {
        let matrix = vec![
            vec![vec![1, 2], vec![3, 4]],
            vec![vec![5, 6], vec![7, 8]],
        ];
        let mut state_matrix = StateMatrix3D::new(matrix.clone());
        state_matrix.get_mut_matrix()[0][0][0] = 99;
        assert_eq!(state_matrix.get_matrix()[0][0][0], 99);
    }
}