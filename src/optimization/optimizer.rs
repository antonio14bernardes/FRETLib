pub trait Optimizer<T, Fitness> 
where
    Fitness: OptimizationFitness,
{
    type Error: OptimizationError;

    // Evaluate the fitness of a given solution, returning `FitnessOutput`
    fn evaluate(&self, individual: &Vec<T>) -> Result<Fitness, Self::Error>;

    // Initialize the population/solution for the optimizer
    fn initialize(&mut self) -> Result<(), Self::Error>;

    // Execute a single optimization step
    fn step(&mut self) -> Result<(), Self::Error>;

    // Run all optimization steps up to a maximum number of iterations
    fn run(&mut self) -> Result<(), Self::Error>;

    // Retrieves the best solution found by the optimizer
    fn get_best_solution(&self) -> Option<&(Vec<f64>, Fitness)>;
}

pub trait OptimizationError {}

pub trait OptimizationFitness: std::fmt::Debug + Clone {
    fn get_fitness(&self) -> f64;
}

pub trait FitnessFunction<T, Fitness> 
where Fitness: OptimizationFitness{
    fn evaluate(&self,individual: &[T]) -> Fitness;
}