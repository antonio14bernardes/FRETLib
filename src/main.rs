use std::collections::HashSet;

use fret_lib::optimization::amalgam_idea::AmalgamIdea;
use fret_lib::optimization::optimizer::Optimizer;
use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::optimization_tracker::TerminationCriterium;
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::{baum_welch, HMM};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use nalgebra::{DMatrix, DVector};
use rand::seq;
use plotters::prelude::*;
use rand_distr::num_traits::Pow;

fn fit_fn(individual: &[f64]) -> f64 {
    let center = [13.0, 1.0, 300.0, 145.0, 654.0, 1042.0, 333.333, 234.0,583.2, 1.0];
    let dist: f64 = individual.iter().zip(&center).map(|(v, c)| (c - v).pow(2)).sum();
    -dist
}

fn main() {
    let problem_size = 10;
    let iter_memory = true;

    let dependency_subsets = (0..problem_size).map(|i| vec![i as usize]).collect();

    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);
    amalgam.set_fitness_function(fit_fn);
    amalgam.set_dependency_subsets(dependency_subsets).unwrap();

    let res = amalgam.run(100000).unwrap();

    let best_solution = amalgam.get_best_solution().unwrap().clone().0;
    println!("result: {:?}", best_solution);
}

fn main_prev() {
    let problem_size = 3;
    let iter_memory = true;

    let mut amalgam = AmalgamIdea::new(problem_size, iter_memory);
    amalgam.set_fitness_function(fit_fn);

    amalgam.initialize().expect("Could not initialize");

    for i in 0..1000{
        println!("Iteration: {}\n", i);
        // Get the initial distributions
        let subsets = amalgam.get_subsets().unwrap();
        let subsets_vec = subsets.get_subsets();
        let mut old_means = Vec::<DVector<f64>>::new();
        let mut old_covs = Vec::<DMatrix<f64>>::new();
        for subset in subsets_vec {
            let (mean, cov) = subset.get_distribution().unwrap();
            old_means.push(mean.clone());
            old_covs.push(cov.clone());
        }

        let selection = amalgam.selection().expect("Could not initialize");

        let (new_means, new_covs) = amalgam.update_distribution(selection).expect("Did not update distribution");

        for (old_mean, new_mean) in old_means.iter().zip(&new_means) {
            println!("Old mean: {:?}", &old_mean);
            println!("New mean: {:?}", &new_mean);
            println!("\n\n")
        }

        for (old_cov, new_cov) in old_covs.iter().zip(new_covs) {
            println!("Old cov: {:?}", &old_cov);
            println!("New cov: {:?}", &new_cov);
            println!("\n\n")
        }

        let (best_current_ind, best_current_fit) = amalgam.get_best_solution().unwrap().clone();
        let improved_individuals = amalgam.update_population().expect("Could not update population");
        let improved_ind_fit: Vec<(Vec<f64>, f64)> = improved_individuals.iter().map(|ind| (ind.clone(), fit_fn(ind))).collect();
        
        println!("Previous best: {:?}, {}", &best_current_ind, best_current_fit);
        println!("Improved individuals: {:?}\n\n", &improved_ind_fit);


        amalgam.update_parameters(improved_individuals, new_means).expect("Could not update parameters");

        println!("New c_mult: {}", amalgam.get_cmult().unwrap());
    }

    

}