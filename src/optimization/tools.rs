use super::optimizer::OptimizationFitness;

// Aux tool for performing selection. Returns the best n individuals and their fitnesses
pub fn select_top_n<Fitness>
(
    individuals: &Vec<Vec<f64>>,
    fitnesses: &Vec<Fitness>,
    n: usize,
) -> (Vec<Vec<f64>>, Vec<Fitness>) 
where Fitness: OptimizationFitness
{
    let mut fitness_with_individuals: Vec<(Fitness, Vec<f64>)> = fitnesses.clone()
        .into_iter()
        .zip(individuals.clone().into_iter())
        .collect();

    fitness_with_individuals.sort_by(|a, b| b.0.get_fitness().partial_cmp(&a.0.get_fitness()).unwrap());

    let selected: Vec<(Fitness, Vec<f64>)> = fitness_with_individuals.into_iter().take(n).collect();

    let (selected_fitnesses, selected_individuals): (Vec<Fitness>, Vec<Vec<f64>>) = selected
        .into_iter()
        .map(|(fitness, individual)| (fitness, individual))
        .unzip();

    (selected_individuals, selected_fitnesses)
}