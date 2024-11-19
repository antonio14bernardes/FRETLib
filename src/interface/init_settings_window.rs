use std::collections::HashMap;

use eframe::egui;

use crate::signal_analysis::hmm::{initialization::{eval_clusters::ClusterEvaluationMethod, hmm_initializer::STATE_NOISE_MULT, kmeans::{KMEANS_EVAL_METHOD_DEFAULT, KMEANS_MAX_ITERS_DEFAULT, KMEANS_NUM_TRIES_DEFAULT, KMEANS_TOLERANCE_DEFAULT}}, StartMatrixInitMethod, StateNoiseInitMethod, StateValueInitMethod, TransitionMatrixInitMethod};

use super::learn_settings_window::render_numeric_input_with_layout;

#[derive(Debug, Clone, PartialEq)]
enum InitTab {
    StateValue,
    StateNoise,
    StartMatrix,
    TransitionMatrix,
}
impl Default for InitTab {
    fn default() -> Self {
        InitTab::StateValue
    }
}
pub struct InitializationSettingsWindow {
    pub is_open: bool,
    pub selected_tab: InitTab,
    pub state_value_init: StateValueInitMethod,
    pub state_noise_init: StateNoiseInitMethod,
    pub start_matrix_init: StartMatrixInitMethod,
    pub transition_matrix_init: TransitionMatrixInitMethod,
    pub input_buffers: HashMap<String, String>, // For numeric input fields

    state_hint_ok: bool,
    set_button_has_been_pressed: bool,
}

impl InitializationSettingsWindow {
    pub fn new() -> Self {
        Self {
            is_open: false,
            selected_tab: InitTab::default(),
            state_value_init: StateValueInitMethod::default(),
            state_noise_init: StateNoiseInitMethod::default(),
            start_matrix_init: StartMatrixInitMethod::default(),
            transition_matrix_init: TransitionMatrixInitMethod::default(),
            input_buffers: HashMap::new(),
            state_hint_ok: true,
            set_button_has_been_pressed: false,
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
        self.set_button_has_been_pressed = false;
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if self.is_open {
            egui::Window::new("Initialization Settings")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .resizable(true)
                .default_size([600.0, 400.0])
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
                        // Tab Bar for switching between settings
                        egui::TopBottomPanel::top("tab_bar_initialize").show_inside(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .selectable_label(self.selected_tab == InitTab::StateValue, "State Values")
                                    .clicked()
                                {
                                    self.selected_tab = InitTab::StateValue;
                                }
                                if ui
                                    .selectable_label(self.selected_tab == InitTab::StateNoise, "State Noise")
                                    .clicked()
                                {
                                    self.selected_tab = InitTab::StateNoise;
                                }
                                if ui
                                    .selectable_label(self.selected_tab == InitTab::StartMatrix, "Start Matrix")
                                    .clicked()
                                {
                                    self.selected_tab = InitTab::StartMatrix;
                                }
                                if ui
                                    .selectable_label(self.selected_tab == InitTab::TransitionMatrix, "Transition Matrix")
                                    .clicked()
                                {
                                    self.selected_tab = InitTab::TransitionMatrix;
                                }
                            });
                        });

                        ui.add_space(10.0);

                        // Render content based on the selected tab
                        match self.selected_tab {
                            InitTab::StateValue => self.render_state_value_settings(ui),
                            InitTab::StateNoise => self.render_state_noise_settings(ui),
                            InitTab::StartMatrix => self.render_start_matrix_settings(ui),
                            InitTab::TransitionMatrix => self.render_transition_matrix_settings(ui),
                            _ => {}
                        }

                        ui.add_space(10.0);

                        // Buttons at the bottom
                        ui.horizontal(|ui| {
                            if ui.button("Set").clicked() {
                                self.set();
                            }
                            if ui.button("Reset").clicked() {
                                self.reset();
                            }
                            if ui.button("Close").clicked() {
                                self.is_open = false;
                            }
                        });
                    });
                });
        }
    }

    fn render_state_value_settings(&mut self, ui: &mut egui::Ui) {
        // Dropdown to select the initialization method
        let current_method = self.state_value_init.to_str().to_string();
        ui.horizontal(|ui| {
            ui.label("Initialization Method:");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_source("state_value_init_method")
                    .selected_text(current_method.clone())
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut self.state_value_init,
                                StateValueInitMethod::KMeansClustering {
                                    max_iters: None,
                                    tolerance: None,
                                    num_tries: None,
                                    eval_method: None,
                                },
                                "KMeans Clustering",
                            )
                            .clicked()
                        {}

                        if ui
                            .selectable_value(
                                &mut self.state_value_init,
                                StateValueInitMethod::Random,
                                "Random",
                            )
                            .clicked()
                        {}

                        if ui
                            .selectable_value(
                                &mut self.state_value_init,
                                StateValueInitMethod::Sparse,
                                "Sparse",
                            )
                            .clicked()
                        {}

                        if ui
                            .selectable_value(
                                &mut self.state_value_init,
                                StateValueInitMethod::StateHints {
                                    state_values: vec![0.0],
                                },
                                "State Hints",
                            )
                            .clicked()
                        {}
                    });
            });
        });

        ui.add_space(10.0);

        // Render settings for the selected initialization method
        match &mut self.state_value_init {
            StateValueInitMethod::KMeansClustering {
                max_iters,
                tolerance,
                num_tries,
                eval_method,
            } => {
                render_numeric_input_with_layout(
                    ui,
                    "Max Iterations:",
                    "state_value_max_iters",
                    max_iters.get_or_insert(KMEANS_MAX_ITERS_DEFAULT),
                    &mut self.input_buffers,
                );
                render_numeric_input_with_layout(
                    ui,
                    "Tolerance:",
                    "state_value_tolerance",
                    tolerance.get_or_insert(KMEANS_TOLERANCE_DEFAULT),
                    &mut self.input_buffers,
                );
                render_numeric_input_with_layout(
                    ui,
                    "Number of Tries:",
                    "state_value_num_tries",
                    num_tries.get_or_insert(KMEANS_NUM_TRIES_DEFAULT),
                    &mut self.input_buffers,
                );
                ui.horizontal(|ui| {
                    ui.label("Evaluation Method:");
                    ui.add_space(10.0); // Add spacing before the dropdown
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        egui::ComboBox::from_id_source("state_value_eval_method")
                            .selected_text(
                                eval_method
                                    .get_or_insert(KMEANS_EVAL_METHOD_DEFAULT)
                                    .to_string(),
                            )
                            .width(200.0) // Consistent width for the dropdown
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    eval_method,
                                    Some(ClusterEvaluationMethod::Silhouette),
                                    "Silhouette",
                                );
                                ui.selectable_value(
                                    eval_method,
                                    Some(ClusterEvaluationMethod::SimplifiedSilhouette),
                                    "Simplified Silhouette",
                                );
                            });
                    });
                });
            }
            StateValueInitMethod::Random => {
                ui.label("Method: Random. No additional settings.");
            }
            StateValueInitMethod::Sparse => {
                ui.label("Method: Sparse. No additional settings.");
            }
            StateValueInitMethod::StateHints { state_values } => {
                ui.label("State Hints:");
            
                // Ensure at least one state value exists
                if state_values.is_empty() {
                    state_values.push(0.0);
                }
            
                // Render each state value with a red "x" to remove it
                let mut i = 0;
                while i < state_values.len() {
                    ui.horizontal(|ui| {
                        if ui
                            .button(egui::RichText::new("x").color(egui::Color32::RED))
                            .clicked()
                        {
                            state_values.remove(i);
                            return; // Skip rendering the rest of this row if removed
                        }
                        ui.label(format!("State Value {}:", i + 1));
                        render_numeric_input_with_layout(
                            ui,
                            "",
                            &format!("state_hint_{}", i),
                            &mut state_values[i],
                            &mut self.input_buffers,
                        );
                    });

                    // Only increment the index if no removal happened
                    i += 1;
                }
            
                // Add a green "+" button to append a new state value
                ui.horizontal(|ui| {
                    if ui
                        .button(egui::RichText::new("+").color(egui::Color32::GREEN))
                        .clicked()
                    {
                        state_values.push(0.0); // Add a new state value initialized to 0.0
                    }
                    ui.label("Add a new state value");
                });

                self.ensure_correct_state_hints();

                if !self.state_hint_ok && self.set_button_has_been_pressed {
                    ui.colored_label(
                        egui::Color32::RED,
                        "Error: State values cannot contain duplicates.",
                    );
                }
            }
        }

        // println!("State values: {:?}", self.state_value_init);
    }


    fn render_state_noise_settings(&mut self, ui: &mut egui::Ui) {
        // Dropdown to select the state noise initialization method
        let current_method = self.state_noise_init.to_str().to_string();
        ui.horizontal(|ui| {
            ui.label("State Noise Method:");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_source("state_noise_init_method")
                    .selected_text(current_method.clone())
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut self.state_noise_init,
                                StateNoiseInitMethod::Sparse { mult: None },
                                "Sparse",
                            )
                            .clicked()
                        {}
                    });
            });
        });
    
        ui.add_space(10.0);
    
        // Render settings for the selected initialization method
        match &mut self.state_noise_init {
            StateNoiseInitMethod::Sparse { mult } => {
                // Obtain a mutable reference to the multiplier value
                let mult_value = mult.get_or_insert(STATE_NOISE_MULT);
    
                // Render the multiplier field in a horizontal layout
                ui.horizontal(|ui| {
                    ui.label("Multiplier:"); // Ensure the label is visible
                    
                    // Retrieve the buffer or initialize it with the current value of `mult`
                    let buffer = self
                        .input_buffers
                        .entry("state_noise_mult".to_string())
                        .or_insert_with(|| mult_value.to_string());
    
                    // Render the text input for the multiplier
                    let response = ui.add_sized([100.0, 20.0], |ui: &mut egui::Ui| {
                        ui.text_edit_singleline(buffer)
                    });
    
                    // Check if the value has been updated or focus is lost
                    if response.lost_focus() {
                        if let Ok(parsed) = buffer.parse::<f64>() {
                            if parsed >= 0.0 {
                                *mult_value = parsed;
                            } else {
                                // Reset the buffer to the previous valid value
                                *buffer = mult_value.to_string();
                            }
                        } else {
                            // Reset the buffer if parsing fails
                            *buffer = mult_value.to_string();
                        }
                    }
                });
            }
        }
    }
    fn render_start_matrix_settings(&mut self, ui: &mut egui::Ui) {
        
    }

    fn render_transition_matrix_settings(&mut self, ui: &mut egui::Ui) {
        
    }

    fn ensure_correct_state_hints(&mut self) {
        let mut has_duplicates = false;
        if let StateValueInitMethod::StateHints { state_values } = &self.state_value_init {
            // Use a HashSet of strings to check for duplicate values
            let mut seen_values = std::collections::HashSet::new();
    
            for &value in state_values {
                // Convert f64 to string for comparison
                let value_str = format!("{:.15}", value); // Ensure precision in comparison
                if !seen_values.insert(value_str) {
                    has_duplicates = true;
                    break;
                }
            }
        }
        self.state_hint_ok = !has_duplicates; // Update state only once, outside the loop
    }

    fn reset(&mut self) {
        self.state_value_init = StateValueInitMethod::default();
        self.state_noise_init = StateNoiseInitMethod::default();
        self.start_matrix_init = StartMatrixInitMethod::default();
        self.transition_matrix_init = TransitionMatrixInitMethod::default();
        self.input_buffers.clear();
        self.set_button_has_been_pressed = false;
    }

    fn set(&mut self) {
        self.set_button_has_been_pressed = true;
    }
}


impl StateValueInitMethod {
    pub fn to_str(&self) -> &str {
        match self {
            StateValueInitMethod::KMeansClustering { .. } => "KMeans Clustering",
            StateValueInitMethod::Random => "Random",
            StateValueInitMethod::Sparse => "Sparse",
            StateValueInitMethod::StateHints { .. } => "State Hints",
        }
    }
}

impl StateNoiseInitMethod {
    /// Converts the `StateNoiseInitMethod` variant to a user-friendly string.
    pub fn to_str(&self) -> &str {
        match self {
            StateNoiseInitMethod::Sparse { .. } => "Sparse",
        }
    }
}