use nalgebra::{DMatrix, DVector};
use std::fmt::{self, Debug};

use crate::optimization::optimizer::OptimizationFitness;
use crate::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};

use super::super::constraints::OptimizationConstraint;
use super::super::optimizer::{Optimizer, OptimizationError};
use super::super::set_of_var_subsets::*;
use super::super::variable_subsets::*;
use super::amalgam_parameters::*;


pub struct AmalgamIdea<'a, Fitness> 
where Fitness: OptimizationFitness
{
    pub(super) max_iterations: Option<usize>,
    pub(super) problem_size: usize,
    pub(super) iter_memory: bool,
    pub(super) parameters: Option<AmalgamIdeaParameters>,
    pub(super) fitness_function: Option<Box<dyn Fn(&[f64]) -> Fitness + 'a>>,
    pub(super) subsets: Option<SetVarSubsets<f64>>,
    pub(super) initial_population: Option<Vec<Vec<f64>>>,
    pub(super) current_population: Vec<Vec<f64>>,
    // pub(super) latest_selection: Option<Vec<Vec<f64>>>,
    pub(super) best_solution: Option<(Vec<f64>, Fitness)>,
    pub(super) current_fitnesses: Vec<Fitness>,
    pub(super) best_fitnesses: Vec<Fitness>,
    pub(super) stagnant_iterations: usize,

    pub(super) current_mean_shifts: Vec<DVector<f64>>,

    pub(super) c_mult: Option<f64>,

    pub(super) init_with_manual_distribution: bool, // True if the user initializes the distribution instead of the population
    pub(super) manual_pop_size: Option<usize>, // Give the option for the user to set the population size manually without changing any other parameters
}


// Basic non algo related methods
impl<'a, Fitness> AmalgamIdea<'a, Fitness>
where Fitness: OptimizationFitness
{
    pub fn new(problem_size: usize, iter_memory: bool) -> Self {
        Self {
            max_iterations: None,
            problem_size,
            iter_memory,

            parameters: None,

            fitness_function: None,

            subsets: None,

            initial_population: None,
            current_population: Vec::new(),
            // latest_selection: None,
            best_solution: None,

            current_fitnesses: Vec::new(),
            best_fitnesses: Vec::new(),

            current_mean_shifts: Vec::new(),

            stagnant_iterations: 0,

            c_mult: None,

            init_with_manual_distribution: false,
            manual_pop_size: None
        }
    } 

    pub fn set_max_iterations(&mut self, max_iters: usize) {
        self.max_iterations = Some(max_iters);
    }

    pub fn set_parameters(&mut self, parameters: AmalgamIdeaParameters) -> Result<(), AmalgamIdeaError> {
        // Check population size compatibility with parameters
        if let Some(population) = self.initial_population.as_ref() {
            if parameters.population_size != population.len() {
                return Err(AmalgamIdeaError::PopulationIncompatibleWithParameters);
            }
        }

        self.parameters = Some(parameters); // Set the single parameters object

        Ok(())
    }
    
    pub fn set_fitness_function<F>(&mut self, fitness_function: F)
    where F: Fn(&[f64]) -> Fitness + 'a,
    {
        self.fitness_function = Some(Box::new(fitness_function));
    }

    pub fn set_initial_population(&mut self, initial_population: Vec<Vec<f64>>) -> Result<(), AmalgamIdeaError> {

        if self.init_with_manual_distribution {return Err(AmalgamIdeaError::InitialDistributionAlreadySet)}
        if let Some(pop_size) = self.manual_pop_size {
            if pop_size != initial_population.len() {
                return Err(AmalgamIdeaError::PopulationIncompatibleWithParameters);
            }
        }

        if initial_population.iter().any(|ind| ind.len() != self.problem_size) {
            return Err(AmalgamIdeaError::PopulationIncompatibleWithProblemSize)
        }

        if let Some(params) = self.parameters.as_ref() {
            if params.population_size != initial_population.len() {
                return Err(AmalgamIdeaError::PopulationIncompatibleWithParameters);
            }
        }
        

        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_population(initial_population.clone())
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
        }
        self.initial_population = Some(initial_population);

        Ok(())
    }

    pub fn set_dependency_subsets(&mut self, dependency_subsets: Vec<Vec<usize>>) -> Result<(), AmalgamIdeaError> {

        if self.subsets.is_some() {
            return Err(AmalgamIdeaError::SubsetError { err: VariableSubsetError::SubsetsAlreadyDefined });
        }

        // Check if max index in the subsets is compatible with the defined problem size
        let max_idx = dependency_subsets
        .iter()
        .flat_map(|s| s.iter().cloned())
        .max()
        .ok_or(AmalgamIdeaError::SubsetsIncompatibleWithProblemSize)?;

        if max_idx != self.problem_size - 1 {
            return Err(AmalgamIdeaError::SubsetsIncompatibleWithProblemSize);
        }


        if let Some(ref init_pop) = self.initial_population {

            let set = SetVarSubsets::new(dependency_subsets, init_pop.clone(), None)
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        } else {

            let set = SetVarSubsets::new_empty(dependency_subsets)
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        }

        Ok(())
    }

    pub fn set_initial_distribution(&mut self, means: &Vec<DVector<f64>>, covs: &Vec<DMatrix<f64>>) -> Result<(), AmalgamIdeaError> {
        // Can't initialize distribution if initial population was manually set
        if self.initial_population.is_some() {return Err(AmalgamIdeaError::InitialPopulationAlreadySet)}

        // Can only set an initial distribution if the subsets are already setup, unless you provide only one set of means/stds
        // In that case, assume one single set of variables
        if self.subsets.is_none() {
            if means.len() != 1 {
                return Err(AmalgamIdeaError::VariableSubsetsNotDefined)
            } else {
                self.set_dependency_subsets(vec![(0..self.problem_size).collect()])?;
            }
        }

        // If subsets is Some and the len of the input is different than the number of subsets, shit is incorrect
        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_distribution_manual(means, covs)
            .map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
        }


        self.init_with_manual_distribution = true;

        Ok(())
    }

    pub fn set_constraints(&mut self, constraints: Vec<OptimizationConstraint<f64>>) -> Result<(), AmalgamIdeaError> {
        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_constraints(constraints).map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;
        
        } else {

            if constraints.len() == 1 { // Assume we are dealing with only one subset - all vars related with eachother
                let subset_indices = vec![(0..self.problem_size).collect::<Vec<usize>>()];
                
                self.set_dependency_subsets(subset_indices)?;

                let subsets = self.subsets.as_mut().unwrap();

                subsets.set_constraints(constraints).map_err(|err| AmalgamIdeaError::VariableSubsetError { err })?;

            } else {
                return Err(AmalgamIdeaError::VariableSubsetsNotDefined);
            }
        }

        Ok(())
    }

    pub fn set_population_size(&mut self, pop_size: usize) -> Result<(), AmalgamIdeaError>{

        if let Some(init_pop) = self.initial_population.as_ref() {
            if init_pop.len() != pop_size {
                return Err(AmalgamIdeaError::PopulationIncompatibleWithParameters);
            }
        }

        if let Some(params) = &mut self.parameters {
            params.population_size = pop_size;
        }

        self.manual_pop_size = Some(pop_size);

        Ok(())
    }

    pub fn get_pop_size(&self) -> Option<usize> {
        if let Some(params) = &self.parameters {
            return Some(params.population_size)
        }
        None
    }

    pub fn get_parameters(&self) -> Option<&AmalgamIdeaParameters> {
        self.parameters.as_ref()
    }

    pub fn get_initial_population(&self) -> Option<&Vec<Vec<f64>>> {
        self.initial_population.as_ref()
    }

    pub fn get_current_population(&self) -> &Vec<Vec<f64>> {
        &self.current_population
    }

    pub fn get_current_fitnesses(&self) -> &Vec<Fitness> {
        &self.current_fitnesses
    }

    pub fn get_current_fitness_values(&self) -> Vec<f64> {
        self.current_fitnesses.iter().map(|f| f.get_fitness()).collect()
    }

    pub fn get_subsets(&self) -> Option<&SetVarSubsets<f64>> {
        self.subsets.as_ref()
    }

    // pub fn get_latest_selection(&self) -> Option<&Vec<Vec<f64>>> {
    //     self.latest_selection.as_ref()
    // }

    pub fn get_cmult(&self) -> Option<&f64> {
        self.c_mult.as_ref()
    }

    pub fn get_stagnant_iterations(&self) -> usize {
        self.stagnant_iterations
    }

    pub fn get_fitnesses(&self) -> &Vec<Fitness> {
        &self.best_fitnesses
    }

    pub fn check_initialization(&self) -> Result<(), AmalgamIdeaError> {
        if self.parameters.is_none() || 
        self.initial_population.is_none() || 
        self.subsets.is_none() ||
        self.current_population.is_empty() ||
        self.best_solution.is_none() {
            return Err(AmalgamIdeaError::InitializationNotPerformed);
        }

        Ok(())
    }

    pub fn evaluate_population(&self) -> Result<Vec<Fitness>, AmalgamIdeaError> {
        let population = &self.current_population;
        let evals: Vec<Fitness> = population
            .iter()
            .map(|individual| self.evaluate(individual))
            .collect::<Result<Vec<Fitness>, AmalgamIdeaError>>()?;

        Ok(evals)
    }
}

impl<'a, Fitness> fmt::Debug for AmalgamIdea<'a, Fitness> 
where Fitness: OptimizationFitness + Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AmalgamIdea")
            .field("problem_size", &self.problem_size)
            .field("iter_memory", &self.iter_memory)
            .field("parameters", &self.parameters)
            .field("subsets", &self.subsets)
            .field("initial_population", &self.initial_population)
            .field("current_population", &self.current_population)
            .field("best_solution", &self.best_solution)
            .field("current_fitnesses", &self.current_fitnesses)
            .field("best_fitnesses", &self.best_fitnesses)
            .field("stagnant_iterations", &self.stagnant_iterations)
            .field("current_mean_shifts", &self.current_mean_shifts)
            .field("c_mult", &self.c_mult)
            .finish()
    }
}

#[derive(Debug)]
pub enum AmalgamIdeaError {
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

    IncompatibleInputSizes,

    InitialPopulationAlreadySet,
    InitialDistributionAlreadySet,


}

impl OptimizationError for AmalgamIdeaError {}

pub trait AmalgamIdeaFitness {
    fn get_fitness(&self) -> &f64;
}