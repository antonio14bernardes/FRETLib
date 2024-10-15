use super::variable_subsets::VariableSubsetError;

pub trait Optimizor<T>{
    // Evaluate the fitness of a given solution
    fn evaluate(&self, individual: &Vec<T>) -> Result<f64, OptimizorError>;

    // Initialize the population/solution for the optimizer
    fn initialize(&mut self) -> Result<(), OptimizorError>;

    // Execute a single optimization step
    fn step(&mut self);

    // Run all
    fn run(&mut self);

    // Retrieves the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&Vec<f64>>;
}

pub enum OptimizorError {
    InitialValuesNotSet,
    FitnessFunctionNotSet,
    VariableSubsetsNotDefined,
    VariableSubsetError{err: VariableSubsetError},

    SubsetsIncompatibleWithProblemSize,
    PopulationIncompatibleWithProblemSize,

    OptimizorTraitMethodNotImplemented,

    SubsetError{err: VariableSubsetError},
    IncompatibleParameterSet,

}

