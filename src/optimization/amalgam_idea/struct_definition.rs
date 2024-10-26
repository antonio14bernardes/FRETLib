use nalgebra::DVector;
use std::fmt;

use super::super::constraints::OptimizationConstraint;
use super::super::optimizer::{Optimizer, OptimizerError};
use super::super::set_of_var_subsets::*;
use super::super::variable_subsets::*;
use super::amalgam_parameters::*;


pub struct AmalgamIdea<'a> {
    pub(super) problem_size: usize,
    pub(super) iter_memory: bool,
    pub(super) parameters: Option<AmalgamIdeaParameters>,
    pub(super) fitness_function: Option<Box<dyn Fn(&[f64]) -> f64 + 'a>>,
    pub(super) subsets: Option<SetVarSubsets<f64>>,
    pub(super) initial_population: Option<Vec<Vec<f64>>>,
    pub(super) current_population: Vec<Vec<f64>>,
    // pub(super) latest_selection: Option<Vec<Vec<f64>>>,
    pub(super) best_solution: Option<(Vec<f64>, f64)>,
    pub(super) current_fitnesses: Vec<f64>,
    pub(super) best_fitnesses: Vec<f64>,
    pub(super) stagnant_iterations: usize,

    pub(super) current_mean_shifts: Vec<DVector<f64>>,

    pub(super) c_mult: Option<f64>,
}


// Basic non algo related methods
impl<'a> AmalgamIdea<'a> {
    pub fn new(problem_size: usize, iter_memory: bool) -> Self {
        Self {
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
        }
    } 

    pub fn set_parameters(&mut self, parameters: AmalgamIdeaParameters) -> Result<(), OptimizerError> {
        // Check population size compatibility with parameters
        if let Some(population) = self.initial_population.as_ref() {
            if parameters.population_size != population.len() {
                return Err(OptimizerError::PopulationIncompatibleWithParameters);
            }
        }

        self.parameters = Some(parameters); // Set the single parameters object

        Ok(())
    }
    
    pub fn set_fitness_function<F>(&mut self, fitness_function: F)
    where F: Fn(&[f64]) -> f64 + 'a,
    {
        self.fitness_function = Some(Box::new(fitness_function));
    }

    pub fn set_initial_population(&mut self, initial_population: Vec<Vec<f64>>) -> Result<(), OptimizerError> {

        if initial_population.iter().any(|ind| ind.len() != self.problem_size) {
            return Err(OptimizerError::PopulationIncompatibleWithProblemSize)
        }

        if let Some(params) = self.parameters.as_ref() {
            if params.population_size != initial_population.len() {
                return Err(OptimizerError::PopulationIncompatibleWithParameters);
            }
        }
        

        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_population(initial_population.clone())
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
        }
        self.initial_population = Some(initial_population);

        Ok(())
    }

    pub fn set_dependency_subsets(&mut self, dependency_subsets: Vec<Vec<usize>>) -> Result<(), OptimizerError> {

        if self.subsets.is_some() {
            return Err(OptimizerError::SubsetError { err: VariableSubsetError::SubsetsAlreadyDefined });
        }

        // Check if max index in the subsets is compatible with the defined problem size
        let max_idx = dependency_subsets
        .iter()
        .flat_map(|s| s.iter().cloned())
        .max()
        .ok_or(OptimizerError::SubsetsIncompatibleWithProblemSize)?;

        if max_idx != self.problem_size - 1 {
            return Err(OptimizerError::SubsetsIncompatibleWithProblemSize);
        }


        if let Some(ref init_pop) = self.initial_population {

            let set = SetVarSubsets::new(dependency_subsets, init_pop.clone(), None)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        } else {

            let set = SetVarSubsets::new_empty(dependency_subsets)
            .map_err(|err| OptimizerError::VariableSubsetError { err })?;
            self.subsets = Some(set);

        }

        Ok(())
    }

    pub fn set_constraints(&mut self, constraints: Vec<OptimizationConstraint<f64>>) -> Result<(), OptimizerError> {
        if let Some(subsets) = self.subsets.as_mut() {
            subsets.set_constraints(constraints).map_err(|err| OptimizerError::VariableSubsetError { err })?;
        
        } else {

            if constraints.len() == 1 { // Assume we are dealing with only one subset - all vars related with eachother
                let subset_indices = vec![(0..self.problem_size).collect::<Vec<usize>>()];
                
                self.set_dependency_subsets(subset_indices)?;

                let subsets = self.subsets.as_mut().unwrap();

                subsets.set_constraints(constraints).map_err(|err| OptimizerError::VariableSubsetError { err })?;

            } else {
                return Err(OptimizerError::VariableSubsetsNotDefined);
            }
        }

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

    pub fn get_current_fitnesses(&self) -> &Vec<f64> {
        &self.current_fitnesses
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

    pub fn get_fitnesses(&self) -> &Vec<f64> {
        &self.best_fitnesses
    }

    pub fn check_initialization(&self) -> Result<(), OptimizerError> {
        if self.parameters.is_none() || 
        self.initial_population.is_none() || 
        self.subsets.is_none() ||
        self.current_population.is_empty() ||
        self.best_solution.is_none() {
            return Err(OptimizerError::InitializationNotPerformed);
        }

        Ok(())
    }

    pub fn evaluate_population(&self) -> Result<Vec<f64>, OptimizerError> {
        let population = &self.current_population;
        let evals: Vec<f64> = population
            .iter()
            .map(|individual| self.evaluate(individual))
            .collect::<Result<Vec<f64>, OptimizerError>>()?;

        Ok(evals)
    }
}

impl<'a> fmt::Debug for AmalgamIdea<'a> {
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


pub enum AmalgamIdeaError {
    IncompatibleParameterSet
}

