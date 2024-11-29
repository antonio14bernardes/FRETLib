use eframe::egui;
use crate::signal_analysis::hmm::amalgam_integration::amalgam_modes::{AMALGAM_DEPENDENCY_DEFAULT, AMALGAM_FITNESS_DEFAULT, AMALGAM_ITER_MEMORY_DEFAULT, AMALGAM_MAX_ITERS_DEFAULT};
use crate::signal_analysis::hmm::hmm_struct::HMM;
use crate::signal_analysis::hmm::learning::learner_trait::HMMLearnerTrait;
use crate::signal_analysis::hmm::optimization_tracker::TerminationCriterium;
use crate::signal_analysis::hmm::{AmalgamDependencies, AmalgamFitness, LearnerSpecificSetup, LearnerType};
use std::collections::HashMap;

pub struct LearnSettingsWindow {
    pub is_open: bool,
    pub learner_type: LearnerType,
    pub learner_setup: LearnerSpecificSetup, // Aux to handle the set button
    
    input_buffers: HashMap<String, String>, // Temporary buffers for text inputs

    learner_type_temp: LearnerType,  // Aux to handle the set button
    learner_setup_temp: LearnerSpecificSetup,
    
}

impl LearnSettingsWindow {
    pub fn new() -> Self {
        let learner_type = LearnerType::default();
        Self {
            
            is_open: false,
            learner_type: learner_type.clone(),
            learner_setup_temp: LearnerSpecificSetup::default(&learner_type),
            input_buffers: HashMap::new(),

            learner_type_temp: learner_type.clone(),
            learner_setup: LearnerSpecificSetup::default(&learner_type),
        }
    }

    pub fn open(&mut self, hmm: &HMM) {
        self.is_open = true;
        // Sync the buffers with the values in learner_setup
        self.sync_buffers_with_setup();
    }

    pub fn show(&mut self, ctx: &egui::Context, ) {


        if self.is_open {
            egui::Window::new("Configure Learn Settings")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(true) // Allow resizing
                .default_size([500.0, 700.0]) // Adjust window size as needed
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
    
                        // Select Learner Type
                        ui.label("Select Learner Type:");
    
                        // Store the previous learner type to detect changes
                        let previous_learner_type = self.learner_type_temp.clone();
    
                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.learner_type_temp,
                                LearnerType::AmalgamIdea,
                                "AMaLGaM IDEA",
                            );
                            ui.radio_value(
                                &mut self.learner_type_temp,
                                LearnerType::BaumWelch,
                                "Baum-Welch",
                            );
                        });
    
                        // Check if learner type has changed
                        if self.learner_type_temp != previous_learner_type {
                            println!(
                                "Learner type changed from {:?} to {:?}",
                                previous_learner_type, self.learner_type_temp
                            );
    
                            // Update learner_setup_temp based on the new learner type
                            self.learner_setup_temp = LearnerSpecificSetup::default(&self.learner_type_temp);
    
                            // Optionally, clear input buffers
                            self.input_buffers.clear();
                        }
    
                        ui.add_space(20.0);
    
                        // Render settings based on the selected learner type
                        match self.learner_type_temp {
                            LearnerType::AmalgamIdea => {
                                self.render_amalgam_idea_settings(ui);
                            }
                            LearnerType::BaumWelch => {
                                self.render_baum_welch_settings(ui);
                            }
                        }
    
                        ui.add_space(20.0);
    
                        // Buttons at the bottom
                        ui.horizontal(|ui| {
                            if ui.button("Set").clicked() {
                                println!("Set button clicked (no functionality yet).");
                                self.learner_setup = self.learner_setup_temp.clone();
                                self.learner_type = self.learner_type_temp.clone();
                            }

                            if ui.button("Reset").clicked() {
                                println!("Reset button clicked.");
                                // Reset learner_setup_temp to the default for the current learner_type_temp
                                self.learner_setup_temp = LearnerSpecificSetup::default(&self.learner_type_temp);

                                // Clear input buffers to reflect defaults
                                self.input_buffers.clear();
                            }

                            if ui.button("Close").clicked() {

                                self.learner_setup_temp = self.learner_setup.clone();
                                self.learner_type_temp = self.learner_type.clone();
                                
                                
                                self.is_open = false;
                                println!("Learn Settings Window closed");
                            }
                        });
                    });
                });
        }
    
        
    }

    fn render_amalgam_idea_settings(&mut self, ui: &mut egui::Ui) {
        // Extract fields or break early if learner_setup_temp is not AmalgamIdea
        let (iter_memory, dependence_type, fitness_type, max_iterations) =
            if let LearnerSpecificSetup::AmalgamIdea {
                iter_memory,
                dependence_type,
                fitness_type,
                max_iterations,
            } = &mut self.learner_setup_temp
            {
                (
                    iter_memory.get_or_insert(AMALGAM_ITER_MEMORY_DEFAULT),
                    dependence_type.get_or_insert(AMALGAM_DEPENDENCY_DEFAULT),
                    fitness_type.get_or_insert(AMALGAM_FITNESS_DEFAULT),
                    max_iterations.get_or_insert(AMALGAM_MAX_ITERS_DEFAULT),
                )
            } else {
                ui.label("Invalid setup for AMaLGaM IDEA.");
                return;
            };
    
        // AMaLGaM IDEA Settings
        ui.label("AMaLGaM IDEA Settings:");
        ui.separator();
    
        // Iterative Memory checkbox
        ui.horizontal(|ui| {
            ui.label("Iterative Memory:");
            ui.checkbox(iter_memory, "");
        });
    
        ui.add_space(10.0);
    
        let space = 10.0;

        // Dependence Type ComboBox
        ui.horizontal(|ui| {
            ui.label("Dependence Type:");
            ui.add_space(20.0); // Add some spacing to push the combo box to the left
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {

                ui.add_space(space);
                egui::ComboBox::from_id_source("dependence_type")
                    .selected_text(dependence_type.to_str().to_string()) // Use to_str method
                    .width(200.0) // Increased size for Dependence Type combo box
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            dependence_type,
                            AmalgamDependencies::AllIndependent,
                            AmalgamDependencies::AllIndependent.to_str(),
                        );
                        ui.selectable_value(
                            dependence_type,
                            AmalgamDependencies::StateCompact,
                            AmalgamDependencies::StateCompact.to_str(),
                        );
                        ui.selectable_value(
                            dependence_type,
                            AmalgamDependencies::ValuesDependent,
                            AmalgamDependencies::ValuesDependent.to_str(),
                        );
                    });
            });
        });
    
        ui.add_space(10.0);
    
        // Fitness Type ComboBox
        ui.horizontal(|ui| {
            ui.label("Fitness Type:");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(space);
                egui::ComboBox::from_id_source("fitness_type")
                    .selected_text(fitness_type.to_str().to_string()) // Use to_str method
                    .width(200.0) // Consistent size for the Fitness Type combo box
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            fitness_type,
                            AmalgamFitness::Direct,
                            AmalgamFitness::Direct.to_str(),
                        );
                        ui.selectable_value(
                            fitness_type,
                            AmalgamFitness::BaumWelch,
                            AmalgamFitness::BaumWelch.to_str(),
                        );
                    });
            });
        });
    
        ui.add_space(10.0);
    
        // Max Iterations Input
        ui.horizontal(|ui| {
            ui.label("Max Iterations:");
            ui.add_space(20.0); // Add some spacing to push the text box to the left
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(space);
                ui.add_sized([100.0, 20.0], |ui: &mut egui::Ui| {
                    let buffer = self
                        .input_buffers
                        .entry("max_iterations".to_string())
                        .or_insert_with(|| max_iterations.to_string());
    
                    let response = ui.text_edit_singleline(buffer);
    
                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(parsed) = buffer.parse::<usize>() {
                            *max_iterations = parsed;
                        } else {
                            *buffer = max_iterations.to_string(); // Reset to previous valid value
                        }
                    }
    
                    response // Explicitly return the response
                });
            });
        });
    }

    fn render_baum_welch_settings(&mut self, ui: &mut egui::Ui) {
        // Extract fields or initialize with default if learner_setup_temp is not BaumWelch
        let termination_criterion = if let LearnerSpecificSetup::BaumWelch {
            termination_criterion,
        } = &mut self.learner_setup_temp
        {
            termination_criterion.get_or_insert_with(TerminationCriterium::default)
        } else {
            ui.label("Invalid setup for Baum-Welch.");
            return;
        };
    
        // Baum-Welch Settings
        ui.label("Baum-Welch Settings:");
        ui.separator();
    
        // ComboBox to select the termination criterion type
        ui.horizontal(|ui| {
            ui.label("Termination Criterion:");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                
                egui::ComboBox::from_id_source("termination_criterion")
                    .selected_text(termination_criterion.to_str().to_string())
                    .width(250.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(termination_criterion, TerminationCriterium::MaxIterations { max_iterations: 500 }, "Max Iterations")
                            .clicked()
                        {}
                        if ui
                            .selectable_value(
                                termination_criterion,
                                TerminationCriterium::OneStepConvergence { epsilon: 1e-5, max_iterations: Some(500) },
                                "One Step Convergence",
                            )
                            .clicked()
                        {}
                        if ui
                            .selectable_value(
                                termination_criterion,
                                TerminationCriterium::OneStepConvergenceAbsolute { epsilon: 1e-5, max_iterations: Some(500) },
                                "One Step Convergence (Absolute)",
                            )
                            .clicked()
                        {}
                        if ui
                            .selectable_value(
                                termination_criterion,
                                TerminationCriterium::PlateauConvergence { epsilon: 1e-5, plateau_len: 20, max_iterations: Some(500) },
                                "Plateau Convergence",
                            )
                            .clicked()
                        {}
                        if ui
                            .selectable_value(
                                termination_criterion,
                                TerminationCriterium::PlateauConvergenceAbsolute { epsilon: 1e-5, plateau_len: 20, max_iterations: Some(500) },
                                "Plateau Convergence (Absolute)",
                            )
                            .clicked()
                        {}
                    });
            });
        });
    
        // Add dynamic input fields based on the selected termination criterion
        match termination_criterion {
            TerminationCriterium::MaxIterations { max_iterations } => {
                render_numeric_input_with_layout(
                    ui,
                    "Max Iterations:",
                    "max_iterations",
                    max_iterations,
                    &mut self.input_buffers,
                );
            }
            TerminationCriterium::OneStepConvergence { epsilon, max_iterations }
            | TerminationCriterium::OneStepConvergenceAbsolute { epsilon, max_iterations } => {
                render_numeric_input_with_layout(
                    ui,
                    "Epsilon:",
                    "epsilon",
                    epsilon,
                    &mut self.input_buffers,
                );
                if let Some(max_iters) = max_iterations {
                    render_numeric_input_with_layout(
                        ui,
                        "Max Iterations:",
                        "max_iterations",
                        max_iters,
                        &mut self.input_buffers,
                    );
                }
            }
            TerminationCriterium::PlateauConvergence { epsilon, plateau_len, max_iterations }
            | TerminationCriterium::PlateauConvergenceAbsolute { epsilon, plateau_len, max_iterations } => {
                render_numeric_input_with_layout(
                    ui,
                    "Epsilon:",
                    "epsilon",
                    epsilon,
                    &mut self.input_buffers,
                );
                render_numeric_input_with_layout(
                    ui,
                    "Plateau Length:",
                    "plateau_len",
                    plateau_len,
                    &mut self.input_buffers,
                );
                if let Some(max_iters) = max_iterations {
                    render_numeric_input_with_layout(
                        ui,
                        "Max Iterations:",
                        "max_iterations",
                        max_iters,
                        &mut self.input_buffers,
                    );
                }
            }
        }
    
        // Add spacing for clarity
        ui.add_space(20.0);
    }

    /// Sync buffers with the current learner setup
    fn sync_buffers_with_setup(&mut self) {
        self.input_buffers.clear();

        match &self.learner_setup {
            LearnerSpecificSetup::AmalgamIdea {
                iter_memory,
                dependence_type,
                fitness_type,
                max_iterations,
            } => {
                self.input_buffers.insert(
                    "iter_memory".to_string(),
                    iter_memory.unwrap_or(false).to_string(),
                );
                self.input_buffers.insert(
                    "dependence_type".to_string(),
                    dependence_type
                        .as_ref()
                        .map_or("".to_string(), |d| d.to_str().to_string()),
                );
                self.input_buffers.insert(
                    "fitness_type".to_string(),
                    fitness_type
                        .as_ref()
                        .map_or("".to_string(), |f| f.to_str().to_string()),
                );
                self.input_buffers.insert(
                    "max_iterations".to_string(),
                    max_iterations.unwrap_or(0).to_string(),
                );
            }
            LearnerSpecificSetup::BaumWelch {
                termination_criterion,
            } => {
                if let Some(criterion) = termination_criterion {
                    self.input_buffers.insert(
                        "termination_criterion".to_string(),
                        criterion.to_str().to_string(),
                    );

                    match criterion {
                        TerminationCriterium::MaxIterations { max_iterations } => {
                            self.input_buffers.insert(
                                "max_iterations".to_string(),
                                max_iterations.to_string(),
                            );
                        }
                        TerminationCriterium::OneStepConvergence { epsilon, .. }
                        | TerminationCriterium::OneStepConvergenceAbsolute { epsilon, .. } => {
                            self.input_buffers.insert("epsilon".to_string(), epsilon.to_string());
                        }
                        TerminationCriterium::PlateauConvergence {
                            epsilon,
                            plateau_len,
                            ..
                        }
                        | TerminationCriterium::PlateauConvergenceAbsolute {
                            epsilon,
                            plateau_len,
                            ..
                        } => {
                            self.input_buffers.insert("epsilon".to_string(), epsilon.to_string());
                            self.input_buffers.insert(
                                "plateau_len".to_string(),
                                plateau_len.to_string(),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Helper function to render numeric input fields with layout
pub(super) fn render_numeric_input_with_layout<T>(
    ui: &mut egui::Ui,
    label: &str,
    buffer_key: &str,
    value: &mut T,
    input_buffers: &mut HashMap<String, String>,
) where
    T: ToString + std::str::FromStr + Copy,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add_space(10.0); // Spacing before the input box
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_sized([100.0, 20.0], |ui: &mut egui::Ui| {
                let buffer = input_buffers
                    .entry(buffer_key.to_string())
                    .or_insert_with(|| value.to_string());

                let response = ui.text_edit_singleline(buffer);

                // Update value on focus loss or invalid input
                if response.lost_focus() {
                    if let Ok(parsed) = buffer.parse::<T>() {
                        *value = parsed; // Update the actual value
                    } else {
                        *buffer = value.to_string(); // Reset buffer to previous valid value
                    }
                }

                response // Return the response for any additional chaining
            });
        });
    });
}

impl TerminationCriterium {
    /// Converts the `TerminationCriterium` variant to a user-friendly string.
    pub fn to_str(&self) -> &str {
        match self {
            TerminationCriterium::MaxIterations { .. } => "Max Iterations",
            TerminationCriterium::OneStepConvergence { .. } => "One Step Convergence",
            TerminationCriterium::OneStepConvergenceAbsolute { .. } => "One Step Convergence (Absolute)",
            TerminationCriterium::PlateauConvergence { .. } => "Plateau Convergence",
            TerminationCriterium::PlateauConvergenceAbsolute { .. } => "Plateau Convergence (Absolute)",
        }
    }
}

impl AmalgamDependencies {
    pub fn to_str(&self) -> &str {
        match self {
            AmalgamDependencies::AllIndependent => "All Independent",
            AmalgamDependencies::StateCompact => "State Compact",
            AmalgamDependencies::ValuesDependent => "Values Dependent",
        }
    }
}

impl AmalgamFitness {
    pub fn to_str(&self) -> &str {
        match self {
            AmalgamFitness::Direct => "Direct",
            AmalgamFitness::BaumWelch => "Baum-Welch",
        }
    }
}