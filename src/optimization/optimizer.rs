use super::variable_subsets::VariableSubsetError;

pub trait Optimizer<T>{
    type Error: OptimizationError;

    // Evaluate the fitness of a given solution
    fn evaluate(&self, individual: &Vec<T>) -> Result<f64, Self::Error>;

    // Initialize the population/solution for the optimizer
    fn initialize(&mut self) -> Result<(), Self::Error>;

    // Execute a single optimization step
    fn step(&mut self) -> Result<(), Self::Error>;

    // Run all
    fn run(&mut self, max_iterations: usize) -> Result<(), Self::Error>;

    // Retrieves the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&(Vec<f64>, f64)>;
}

pub trait OptimizationError {}