use std::collections::HashMap;
use eframe::egui;

use crate::signal_analysis::hmm::{
    initialization::{eval_clusters::ClusterEvaluationMethod, kmeans::{KMEANS_EVAL_METHOD_DEFAULT, KMEANS_MAX_ITERS_DEFAULT, KMEANS_NUM_TRIES_DEFAULT, KMEANS_TOLERANCE_DEFAULT}}, number_states_finder::hmm_num_states_finder::{MAX_K_DEFAULT, MIN_K_DEFAULT}
};
use crate::signal_analysis::hmm::NumStatesFindStratWrapper;
use super::learn_settings_window::render_numeric_input_with_layout;

pub struct FindNumberOfStatesWindow {
    pub is_open: bool,
    pub strategy: NumStatesFindStratWrapper,
    pub min_k: usize,
    pub max_k: usize,
    pub input_buffers: HashMap<String, String>, // For numeric input fields
    pub min_max_valid: bool,
    pub set_button_pressed: bool,
}

impl FindNumberOfStatesWindow {
    pub fn new() -> Self {
        Self {
            is_open: false,
            strategy: NumStatesFindStratWrapper::KMeansClustering {
                num_tries: Some(KMEANS_NUM_TRIES_DEFAULT),
                max_iters: Some(KMEANS_MAX_ITERS_DEFAULT),
                tolerance: Some(KMEANS_TOLERANCE_DEFAULT),
                method: Some(KMEANS_EVAL_METHOD_DEFAULT),
            },
            min_k: MIN_K_DEFAULT,
            max_k: MAX_K_DEFAULT,
            input_buffers: HashMap::new(),
            min_max_valid: true,
            set_button_pressed: false,
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if self.is_open {
            egui::Window::new("Find Number of States Settings")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .fixed_size([350.0, 400.0])
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {

                            // For min k
                            ui.label("Min States (k):");
                            ui.add_space(10.0); // Optional spacing after the label
                            let buffer = self
                                .input_buffers
                                .entry("min_k".to_string())
                                .or_insert_with(|| self.min_k.to_string());
                        
                            let response = ui.add_sized([50.0, 20.0], |ui: &mut egui::Ui| ui.text_edit_singleline(buffer));
                        
                            if response.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                if let Ok(parsed) = buffer.parse::<usize>() {
                                    self.min_k = parsed;
                                } else {
                                    *buffer = self.min_k.to_string(); // Reset to previous valid value
                                }
                            }
                            
                            ui.add_space(10.0);
                            // For max k
                            ui.label("Max States (k):");
                            ui.add_space(10.0); // Optional spacing after the label
                            let buffer = self
                                .input_buffers
                                .entry("max_k".to_string())
                                .or_insert_with(|| self.max_k.to_string());
                        
                                let response = ui.add_sized([50.0, 20.0], |ui: &mut egui::Ui| ui.text_edit_singleline(buffer));
                        
                            if response.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                if let Ok(parsed) = buffer.parse::<usize>() {
                                    self.max_k = parsed;
                                } else {
                                    *buffer = self.max_k.to_string(); // Reset to previous valid value
                                }
                            }
                        });

                        // Validate min max inputs
                        self.validate_min_max();
    
                        ui.add_space(10.0);
    
                        // Dropdown for selecting the strategy
                        let current_strategy = self.strategy.to_str().to_string();
                        ui.horizontal(|ui| {
                            ui.label("Strategy:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                egui::ComboBox::from_id_source("num_states_strategy")
                                    .selected_text(current_strategy.clone())
                                    .width(200.0)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_value(
                                                &mut self.strategy,
                                                NumStatesFindStratWrapper::KMeansClustering {
                                                    num_tries: Some(KMEANS_NUM_TRIES_DEFAULT),
                                                    max_iters: Some(KMEANS_MAX_ITERS_DEFAULT),
                                                    tolerance: Some(KMEANS_TOLERANCE_DEFAULT),
                                                    method: Some(KMEANS_EVAL_METHOD_DEFAULT),
                                                },
                                                "KMeans Clustering",
                                            )
                                            .clicked()
                                        {}
    
                                        if ui
                                            .selectable_value(
                                                &mut self.strategy,
                                                NumStatesFindStratWrapper::BaumWelch,
                                                "Baum-Welch",
                                            )
                                            .clicked()
                                        {}
    
                                        if ui
                                            .selectable_value(
                                                &mut self.strategy,
                                                NumStatesFindStratWrapper::CurrentSetup,
                                                "Current Setup",
                                            )
                                            .clicked()
                                        {}
                                    });
                            });
                        });
    
                        ui.add_space(10.0);
    
                        // Render strategy-specific settings
                        match &mut self.strategy {
                            NumStatesFindStratWrapper::KMeansClustering {
                                num_tries,
                                max_iters,
                                tolerance,
                                method,
                            } => {
                                render_numeric_input_with_layout(
                                    ui,
                                    "Number of Tries:",
                                    "num_tries",
                                    num_tries.get_or_insert(KMEANS_NUM_TRIES_DEFAULT),
                                    &mut self.input_buffers,
                                );
                                render_numeric_input_with_layout(
                                    ui,
                                    "Max Iterations:",
                                    "max_iters",
                                    max_iters.get_or_insert(KMEANS_MAX_ITERS_DEFAULT),
                                    &mut self.input_buffers,
                                );
                                ui.horizontal(|ui| {
                                    ui.label("Tolerance:");
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        // Retrieve or initialize the buffer
                                        let buffer = self
                                            .input_buffers
                                            .entry("tolerance".to_string())
                                            .or_insert_with(|| tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT).to_string());
                                
                                        // Render the text edit and capture response
                                        let response = ui.add_sized([100.0, 20.0], |ui: &mut egui::Ui| ui.text_edit_singleline(buffer));
                                
                                        // Validate and update on losing focus
                                        if response.lost_focus() {
                                            if let Ok(parsed) = buffer.parse::<f64>() {
                                                if parsed < 0.0 {
                                                    // Reset to the previous valid value if negative
                                                    *buffer = tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT).to_string();
                                                } else {
                                                    // Update tolerance with valid parsed value
                                                    *tolerance = Some(parsed);
                                                }
                                            } else {
                                                // Reset to the previous valid value if parsing fails
                                                *buffer = tolerance.unwrap_or(KMEANS_TOLERANCE_DEFAULT).to_string();
                                            }
                                        }
                                    });
                                });
                                

                                ui.horizontal(|ui| {
                                    ui.label("Evaluation Method:");
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        egui::ComboBox::from_id_source("kmeans_eval_method")
                                            .selected_text(
                                                method
                                                    .get_or_insert(KMEANS_EVAL_METHOD_DEFAULT)
                                                    .to_string(),
                                            )
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(
                                                    method,
                                                    Some(ClusterEvaluationMethod::Silhouette),
                                                    "Silhouette",
                                                );
                                                ui.selectable_value(
                                                    method,
                                                    Some(ClusterEvaluationMethod::SimplifiedSilhouette),
                                                    "Simplified Silhouette",
                                                );
                                            });
                                    });
                                });
                            }
                            NumStatesFindStratWrapper::BaumWelch => {
                                ui.label("Strategy: Baum-Welch");
                                ui.label("No additional settings.");
                            }
                            NumStatesFindStratWrapper::CurrentSetup => {
                                ui.label("Strategy: Current Setup");
                                ui.label("No additional settings.");
                            }
                        }
    
                        ui.add_space(20.0);
    
                        // Bottom buttons
                        ui.horizontal(|ui| {
                            if ui.button("Set").clicked() {
                                self.set_function();
                            }
                            if ui.button("Reset").clicked() {
                                self.reset();
                            }
                            if ui.button("Close").clicked() {
                                self.is_open = false;
                            }
                        });
                    });
                    if !self.min_max_valid && self.set_button_pressed {
                        ui.colored_label(
                            egui::Color32::RED,
                            "Error: Maximum states (k) cannot be less than minimum states (k).",
                        );
                    }
                });
        }
        
    }
    fn set_function(&mut self) {
        self.set_button_pressed = true;
    }

    fn validate_min_max(&mut self) {
        if self.max_k < self.min_k {
            self.min_max_valid = false;
        } else {
            self.min_max_valid = true;
        }
    }

    fn reset(&mut self) {
        self.strategy = NumStatesFindStratWrapper::default();
        self.min_k = MIN_K_DEFAULT;
        self.max_k = MAX_K_DEFAULT;
        self.input_buffers.clear();
    }
}

impl NumStatesFindStratWrapper {
    pub fn to_str(&self) -> &str {
        match self {
            NumStatesFindStratWrapper::KMeansClustering { .. } => "KMeans Clustering",
            NumStatesFindStratWrapper::BaumWelch => "Baum-Welch",
            NumStatesFindStratWrapper::CurrentSetup => "Current Setup"
        }
    }
}

