use std::io::{self, Write};
use terminal_size::{terminal_size, Width};


use crate::{optimization::optimizer::OptimizationFitness, signal_analysis::hmm::AmalgamFitness};

use super::{AmalgamIdeaParameters};

#[derive(Clone, Debug)]
pub enum YapLevel {
    ALot,
    ALittle,
    None,
}

#[derive(Clone, Debug)]
pub struct Yapper {
    pub max_iters: Option<usize>,
    pub problem_size: usize,
    pub iter_memory: bool,
    pub manual_population_size: Option<usize>,
    pub amalgam_parameters: AmalgamIdeaParameters,
    pub yap_level: YapLevel,
    last_msg_len: usize, // Tracks the length of the last message for clearing
}


impl Yapper {
    pub fn new
    (
        max_iters: Option<usize>,
        problem_size: usize,
        iter_memory: bool,
        manual_population_size: Option<usize>,
        amalgam_parameters: &AmalgamIdeaParameters,
        yap_level: YapLevel,

    ) -> Self {

        Self {
            max_iters,
            problem_size,
            iter_memory,
            manual_population_size,
            amalgam_parameters: amalgam_parameters.clone(),
            yap_level,
            last_msg_len: 0,
        }
    }

    pub fn startup_yap(&self) {
        match self.yap_level {
            YapLevel::None => return, // Skip printing if yap level is None
            YapLevel::ALittle => {
                println!("\n\nStarting a run of AmalgamIdea");
                println!("=============================================");
            }
            YapLevel::ALot => {
                let population_size = self.manual_population_size.unwrap_or(self.amalgam_parameters.population_size);

                println!("\n\nStarting a run of AmalgamIdea");
                println!("=============================================");
                println!("With parameters:\n");
                println!(
                         "Maximum Iterations               : {}",
                    self.max_iters
                        .map(|iters| iters.to_string())
                        .unwrap_or_else(|| "None (runs until convergence)".to_string())
                );
                println!("Problem Size                     : {}", self.problem_size);
                println!("Iteration Memory                 : {}", self.iter_memory);
                println!("Population Size                  : {}", population_size);
                println!("Selection Fraction (tau)         : {:.3}", self.amalgam_parameters.tau);
                println!("C-Multiplier Increase            : {:.3}", self.amalgam_parameters.c_mult_inc);
                println!("C-Multiplier Decrease            : {:.3}", self.amalgam_parameters.c_mult_dec);
                println!("C-Multiplier Min Threshold       : {:.3}", self.amalgam_parameters.c_mult_min);
                println!("Covariance Memory (eta_cov)      : {:.3}", self.amalgam_parameters.eta_cov);
                println!("Mean Shift Memory (eta_shift)    : {:.3}", self.amalgam_parameters.eta_shift);
                println!("Mean Shift Fraction (alpha_shift): {:.3}", self.amalgam_parameters.alpha_shift);
                println!("Mean Shift Factor (gamma_shift)  : {:.3}", self.amalgam_parameters.gamma_shift);
                println!("Stagnant Iterations Threshold    : {}", self.amalgam_parameters.stagnant_iterations_threshold);
                println!("=============================================\n");
            }
        }
    }

    pub fn iteration_yap<T>(&mut self, current_iteration: &usize, best_fitness: &T, best_individual: &Vec<f64>)
    where
        T: OptimizationFitness,
    {
        // Only print every 10 iterations
        if current_iteration % 10 != 0 {
            return;
        }

        // Prepare the message components
        let max_iters_display = self.max_iters.map_or_else(
            || format!("Current Iteration: {}", current_iteration),
            |max| format!("Current Iteration: {}/{}", current_iteration, max),
        );
        let fitness_info = format!(" | Best Fitness: {:.6}", best_fitness.get_fitness());

        // Format each element of best_individual to have 6 decimal places
        let individual_info = if let YapLevel::ALot = self.yap_level {
            let formatted_individual: Vec<String> = best_individual
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            format!(" | Best Individual: [{}]", formatted_individual.join(", "))
        } else {
            String::new()
        };

        let output: String;

        match self.yap_level {
            YapLevel::None => {output = String::new();}
            YapLevel::ALittle => {output = format!("{}", max_iters_display);}
            YapLevel::ALot => {output = format!("{}{}{}", max_iters_display, fitness_info, individual_info);}
        }

        // Determine terminal width
        let terminal_width = match terminal_size() {
            Some((Width(w), _)) => w as usize,
            None => 80,
        };

        // Calculate previous message lines
        let prev_lines = (self.last_msg_len + terminal_width - 1) / terminal_width;

        // Clear the previous message
        if self.last_msg_len > 0 {
            for _ in 0..prev_lines {
                print!("\x1B[1A"); // Move up one line
                print!("\x1B[2K"); // Clear the line
            }
            print!("\r"); // Move cursor to the beginning
        }

        // Print new message
        println!("{}", output);
        io::stdout().flush().unwrap();

        // Update last message length
        self.last_msg_len = output.len();
    }
}