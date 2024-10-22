use super::variable_subsets::VariableSubsetError;

pub trait Optimizer<T>{
    // Evaluate the fitness of a given solution
    fn evaluate(&self, individual: &Vec<T>) -> Result<f64, OptimizerError>;

    // Initialize the population/solution for the optimizer
    fn initialize(&mut self) -> Result<(), OptimizerError>;

    // Execute a single optimization step
    fn step(&mut self) -> Result<(), OptimizerError>;

    // Run all
    fn run(&mut self, max_iterations: usize) -> Result<(), OptimizerError>;

    // Retrieves the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&(Vec<f64>, f64)>;
}

#[derive(Debug)]
pub enum OptimizerError {
    InitialValuesNotSet,
    FitnessFunctionNotSet,
    VariableSubsetsNotDefined,
    VariableSubsetError{err: VariableSubsetError},

    SubsetsIncompatibleWithProblemSize,
    PopulationIncompatibleWithProblemSize,
    PopulationIncompatibleWithParameters,

    OptimizorTraitMethodNotImplemented,

    SubsetError{err: VariableSubsetError},
    IncompatibleParameterSet,
    ParametersNoSet,

    InitializationNotPerformed,
    SelectionNotPerformed,


}


// Aux tool for performing selection. Returns the best n individuals and their fitnesses
pub fn select_top_n(
    individuals: &Vec<Vec<f64>>,
    fitnesses: &Vec<f64>,
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut fitness_with_individuals: Vec<(f64, Vec<f64>)> = fitnesses.clone()
        .into_iter()
        .zip(individuals.clone().into_iter())
        .collect();

    fitness_with_individuals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let selected: Vec<(f64, Vec<f64>)> = fitness_with_individuals.into_iter().take(n).collect();

    let (selected_fitnesses, selected_individuals): (Vec<f64>, Vec<Vec<f64>>) = selected
        .into_iter()
        .map(|(fitness, individual)| (fitness, individual))
        .unzip();

    (selected_individuals, selected_fitnesses)
}