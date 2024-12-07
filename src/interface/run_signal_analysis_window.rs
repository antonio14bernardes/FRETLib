use std::collections::HashMap;

use eframe::egui;

use crate::signal_analysis::hmm::{hmm_instance::HMMInstance, hmm_matrices::{StartMatrix, TransitionMatrix}, hmm_struct::{HMMComponent, HMMInput}, state::State, LearnerSpecificInitialValues, LearnerType};

use super::{learn_settings_window::render_numeric_input_with_layout, main_tab::HMMInputHint};

pub struct RunSignalProcessingWindow {
    pub is_open: bool,
    pub hmm_input_hint: Option<HMMInputHint>, // Hint to determine input type and learner specifics
    pub sequence_set: Option<Vec<Vec<f64>>>,
    pub hmm_input: Option<HMMInput>, // The final constructed HMMInput

    // Fields for user input
    pub num_states: Option<usize>,
    pub state_values: Vec<f64>,
    pub state_noises: Vec<f64>,
    pub start_matrix: Vec<f64>,         // 1D representation of start matrix
    pub transition_matrix: Vec<Vec<f64>>, // 2D matrix for transitions

    pub input_buffers: HashMap<String, String>, // Temporary buffers for input fields

    // Input validity checks
    run_has_been_pressed: bool,
    state_values_valid: bool,
    states_empty: bool,
    state_noises_valid: bool,
    start_matrix_valid: bool,
    transition_matrix_valid: bool,
}

impl RunSignalProcessingWindow {
    pub fn new() -> Self {
        Self {
            is_open: false,
            hmm_input_hint: None,
            sequence_set: None,
            hmm_input: None,

            num_states: None,
            state_values: vec![0.2, 0.8],
            state_noises: vec![0.1, 0.1],
            start_matrix: Vec::new(),
            transition_matrix: Vec::new(),

            input_buffers: HashMap::new(),

            run_has_been_pressed: false,
            state_values_valid: false,
            state_noises_valid: false,
            states_empty: false,
            start_matrix_valid: false,
            transition_matrix_valid: false,
        }
    }

    pub fn open(&mut self, sequence_set: Vec<Vec<f64>>, hint: HMMInputHint) {
        self.is_open = true;
        self.hmm_input_hint = Some(hint.clone());
        self.sequence_set = Some(sequence_set);
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<HMMInput>{
        if !self.is_open {return None;}

        egui::Window::new("Run Signal Processing")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .fixed_size([500.0, 600.0])
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    if let Some(hint) = &self.hmm_input_hint {
                        match hint {
                            HMMInputHint::NumStatesFinder => self.render_num_states_finder(ui),
                            HMMInputHint::Initializer => self.render_initializer(ui),
                            HMMInputHint::Learner { learner_type } => {
                                // Clone or copy learner_type to avoid the borrow conflict
                                let learner_type = learner_type.clone();
                                self.render_learner(ui, &learner_type)
                            }
                            HMMInputHint::Analyzer => self.render_analyzer(ui),
                        }
                    }

                    ui.add_space(20.0);

                    // Bottom Buttons
                    ui.horizontal(|ui| {
                        if ui.button("Run").clicked() {
                            self.run_has_been_pressed = true;

                            // if !(self.state_values_valid && 
                            //     self.state_noises_valid && 
                            //     self.start_matrix_valid && 
                            //     self.transition_matrix_valid) ||
                            //     self.states_empty {
                                

                            //     // Get outta here and don't construct input nor HMM process
                            //     return;
                            // }

                            if !self.check_overall_validity() {
                                // Get outta here and don't construct input nor HMM process
                                return;
                            }

                            // If checks have been passed, construct the input and HMM process
                            self.construct_hmm_input();
                            if self.hmm_input.is_some() {
                                // println!("HMM Input constructed: {:?}", self.hmm_input);
                                self.is_open = false;
                            }
                        }

                        if ui.button("Close").clicked() {
                            self.is_open = false;
                            self.run_has_been_pressed = false;

                            // Clean the state values, noises, and hmm matrices vecs
                            self.state_values = vec![0.2, 0.8];
                            self.state_noises = vec![0.1, 0.1];
                            self.start_matrix = Vec::new();
                            self.transition_matrix = Vec::new();
                        }
                    });
                });
            });
        
        // println!("Num States {:?}", self.num_states); 
        // println!("States empty {}", self.states_empty);  
        // println!("State values: {:?}", self.state_values);
        // println!("State values valid: {}", self.state_values_valid);
        // println!("State noises: {:?}", self.state_noises);
        // println!("State noises valid: {:?}", self.state_noises_valid);
        // println!("Start matrix: {:?}", self.start_matrix);
        // println!("Start matrix valid: {}", self.start_matrix_valid);
        // println!("Transition matrix: {:?}", self.transition_matrix);
        // println!("Transition matrix valid: {}", self.transition_matrix_valid);


        // Return Option of HMMInput. If Some(input), then the HMM is ready to go, otherwise not yet.
        self.hmm_input.take()
        
    }

    fn render_num_states_finder(&mut self, ui: &mut egui::Ui) {
        ui.label("Num States Finder does not require additional inputs.");
    }

    fn render_initializer(&mut self, ui: &mut egui::Ui) {
        ui.label("Initializer Configuration:");
        render_numeric_input_with_layout(
            ui,
            "Number of States:",
            "num_states",
            self.num_states.get_or_insert(1),
            &mut self.input_buffers,
        );

        // Validate and correct the number of states. They cant be 0
        if let Some(num_states) = &mut self.num_states {
            if *num_states == 0 {
                *num_states = 1; // Correct the value if 0. Put it back to 1
                self.input_buffers
                    .insert("num_states".to_string(), "1".to_string()); // Update the buffer
            }
        }
    }

    
    fn render_learner(&mut self, ui: &mut egui::Ui, learner_type: &LearnerType) {
        ui.label(format!("Learner Configuration: {:?}", learner_type));
    
        if let LearnerType::BaumWelch = learner_type {
            self.render_states(ui); // Delegate the state rendering to `render_states`
            self.show_states_error_messages(ui);
        } else {
            render_numeric_input_with_layout(
                ui,
                "Number of States:",
                "num_states",
                self.num_states.get_or_insert(1),
                &mut self.input_buffers,
            );
    
            // Validate and correct the number of states. They cant be 0
            if let Some(num_states) = &mut self.num_states {
                if *num_states == 0 {
                    *num_states = 1; // Correct the value if 0. Put it back to 1
                    self.input_buffers
                        .insert("num_states".to_string(), "1".to_string()); // Update the buffer
                }
            }
        }
    }

    fn render_analyzer(&mut self, ui: &mut egui::Ui) {
        ui.label("Analyzer Configuration");
        ui.separator();
    
        // Wrap the entire analyzer configuration in a scrollable area
        egui::ScrollArea::vertical()
            .max_height(400.0) // Adjust max height for the scroll area
            .show(ui, |ui| {
                // States
                self.render_states(ui);
                self.show_states_error_messages(ui);
    
                // Start matrix
                self.render_start_matrix(ui);
                self.show_start_matrix_error_messages(ui);
    
                // Transition matrix
                self.render_transition_matrix(ui);
                self.show_transition_matrix_error_messages(ui);
            });
    }

    fn render_states(&mut self, ui: &mut egui::Ui) {
        ui.label("States Configuration:");
        ui.separator();
    
        // Header Row for Value and Noise labels
        ui.horizontal(|ui| {
            ui.label(""); // Empty space for alignment with remove button
            ui.label(""); // Empty space for alignment with State label
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(20.0);
                ui.label("Noise"); // Noise label
                ui.add_space(40.0);
                ui.label("Value"); // Value label
            });
        });
    
        ui.separator();
    
        // Iterate through each state to create rows
        let mut i = 0;
        while i < self.state_values.len() {
            ui.horizontal(|ui| {
                // Show remove button only if this is not the first state
                if i > 0 {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("x").color(egui::Color32::RED),
                            )
                            .small(),
                        )
                        .clicked()
                    {
                        // Remove the state and its buffers
                        self.state_values.remove(i);
                        self.state_noises.remove(i);
                        self.input_buffers.remove(&format!("value_{}", i));
                        self.input_buffers.remove(&format!("noise_{}", i));

                        // Shift remaining buffers to match new indices
                        for j in i..self.state_values.len() {
                            if let Some(buffer) = self.input_buffers.remove(&format!("value_{}", j + 1)) {
                                self.input_buffers.insert(format!("value_{}", j), buffer);
                            }
                            if let Some(buffer) = self.input_buffers.remove(&format!("noise_{}", j + 1)) {
                                self.input_buffers.insert(format!("noise_{}", j), buffer);
                            }
                        }
                        return; // Exit early to avoid rendering stale data
                    }
                } else {
                    ui.add_space(37.0); // Add empty space where the button would be
                }
    
                // State Label
                ui.label(format!("State {}:", i + 1));
    
                // Value Input
                let value_key = format!("value_{}", i);
                let noise_key = format!("noise_{}", i);
    
                // Ensure buffers exist for this state
    
                // Right-to-Left Layout for Value and Noise Inputs
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Noise Input
                    let noise_buffer = {
                        self.input_buffers
                            .entry(noise_key.clone())
                            .or_insert_with(|| self.state_noises[i].to_string())
                    };
                    if ui
                        .add_sized(
                            [80.0, 20.0],
                            egui::TextEdit::singleline(noise_buffer),
                        )
                        .lost_focus() || ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        if let Ok(parsed) = noise_buffer.parse::<f64>() {
                            self.state_noises[i] = parsed; // Update noise
                        } else {
                            *noise_buffer = self.state_noises[i].to_string(); // Reset to valid value
                        }
                    }

                    ui.add_space(10.0); // Add spacing between inputs

                    // Value Input
                    let value_buffer = {
                        self.input_buffers
                            .entry(value_key.clone())
                            .or_insert_with(|| self.state_values[i].to_string())
                    };
                    if ui
                        .add_sized(
                            [80.0, 20.0],
                            egui::TextEdit::singleline(value_buffer),
                        )
                        .lost_focus() || ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        if let Ok(parsed) = value_buffer.parse::<f64>() {
                            self.state_values[i] = parsed; // Update value
                        } else {
                            *value_buffer = self.state_values[i].to_string(); // Reset to valid value
                        }
                    }
                });
            });
    
            i += 1;
        }
    
        // Add New State Row
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("+").color(egui::Color32::GREEN),
                    )
                    .small(),
                )
                .clicked()
            {
                let new_index = self.state_values.len();
                self.state_values.push(0.0); // Default value
                self.state_noises.push(1.0); // Default noise
                self.input_buffers
                    .insert(format!("value_{}", new_index), "0.0".to_string());
                self.input_buffers
                    .insert(format!("noise_{}", new_index), "1.0".to_string());
            }
            ui.label("Add a new state");
        });
    
        ui.separator();
    
        // Validate the state vectors after editing
        self.check_state_vecs_validity();
    }
    
    
    
    fn render_start_matrix(&mut self, ui: &mut egui::Ui) {
        ui.label("Start Matrix Configuration:");
    
        // Initialize start_matrix and buffer if not already done
        if self.start_matrix.is_empty() {
            let n = if self.state_values.is_empty() {1} else {self.state_values.len()};
            self.start_matrix = vec![1.0 / n as f64; n]; // Initialize with 1/n values
    
            // Convert to Python-style list string representation
            self.input_buffers
                .entry("start_matrix".to_string())
                .or_insert(format_start_matrix(&self.start_matrix));
        }
    
        // Retrieve or initialize the buffer
        let buffer = self
            .input_buffers
            .entry("start_matrix".to_string())
            .or_insert_with(|| format_start_matrix(&self.start_matrix));
    
        // Render the multiline input box
        if ui
            .add(
                egui::TextEdit::multiline(buffer)
                    .desired_width(f32::INFINITY)
                    .desired_rows(4),
            )
            .lost_focus()
        {
            // Parse the input buffer as a Python-style list
            if let Some(parsed) = parse_python_list(buffer) {
                // If valid, update start_matrix
                if parsed.len() == self.state_values.len() {
                    self.start_matrix = parsed;
                
                    // Reset buffer to the current valid start_matrix if sizes don't match
                    *buffer = format_start_matrix(&self.start_matrix);
                } else {
                    // Reset buffer to the current valid start_matrix if parsing fails
                    *buffer = format_start_matrix(&self.start_matrix);
                }
            } else {
                // Reset buffer to the current valid start_matrix if parsing fails
                *buffer = format_start_matrix(&self.start_matrix);
            }
        }


        // Render clickable links for correcting and resetting the matrix
        ui.horizontal(|ui| {
            if ui.link(egui::RichText::new("Correct matrix").size(13.0)).clicked() {
                self.start_matrix = correct_probs(&self.start_matrix);

                // Update the input buffer to reflect the corrected matrix
                *buffer = format_start_matrix(&self.start_matrix);
            }

            ui.add_space(10.0);

            if ui.link(egui::RichText::new("Reset").size(13.0).color(egui::Color32::BLUE)).clicked() {
                self.start_matrix = reset_start_matrix(self.state_values.len());

                // Update the input buffer to reflect the reset matrix
                *buffer = format_start_matrix(&self.start_matrix);
            }
        });

        self.check_start_matrix_validity();
    }

    fn render_transition_matrix(&mut self, ui: &mut egui::Ui) {
        ui.label("Transition Matrix Configuration:");
    
        // Initialize transition_matrix and buffer if not already done
        if self.transition_matrix.is_empty() {
            let n = if self.state_values.is_empty() { 1 } else { self.state_values.len() };
            self.transition_matrix = vec![vec![1.0 / n as f64; n]; n]; // Initialize with 1/n values for each row
    
            // Format the entire matrix dynamically into a string with Python-style brackets
            let formatted_matrix = format_transition_matrix(&self.transition_matrix);
    
            self.input_buffers
                .entry("transition_matrix".to_string())
                .or_insert(formatted_matrix);
        }
    
        // Retrieve or initialize the buffer
        let buffer = self
            .input_buffers
            .entry("transition_matrix".to_string())
            .or_insert_with(|| format_transition_matrix(&self.transition_matrix));
    
        // Render the multiline input box
        if ui
            .add(
                egui::TextEdit::multiline(buffer)
                    .desired_width(f32::INFINITY)
                    .desired_rows(self.transition_matrix.len() + 2),
            )
            .lost_focus()
        {
            // Parse the input buffer as a Python-style list-of-lists
            if let Some(parsed) = parse_python_matrix(buffer) {
                // If valid, update transition_matrix
                if parsed.len() == self.state_values.len() {
                    self.transition_matrix = parsed;
                } else {
                    // Reset buffer to the current valid transition_matrix if sizes don't match
                    *buffer = format_transition_matrix(&self.transition_matrix);
                }
            } else {
                // Reset buffer to the current valid transition_matrix if parsing fails
                *buffer = format_transition_matrix(&self.transition_matrix);
            }
        }
    
        ui.horizontal(|ui| {
            if ui.link(egui::RichText::new("Correct matrix").size(13.0)).clicked() {
                for row in &mut self.transition_matrix {
                    *row = correct_probs(row);
                }
        
                // Update the input buffer to reflect the corrected matrix
                *buffer = format_transition_matrix(&self.transition_matrix);
            }

            ui.add_space(10.0);
        
            if ui.link(egui::RichText::new("Reset").size(13.0).color(egui::Color32::BLUE)).clicked() {
                self.transition_matrix = reset_transition_matrix(self.state_values.len());
        
                // Update the input buffer to reflect the reset matrix
                *buffer = format_transition_matrix(&self.transition_matrix);
            }
        });
    
        self.check_transition_matrix_validity();
    }

    fn check_state_vecs_validity(&mut self) {

        // Check if states are empty
        self.states_empty = self.state_values.is_empty();

        // Check if there is no repeated values in the values vec:

        // Use a HashSet of strings to check for duplicate values
        let mut seen_values = std::collections::HashSet::new();
        let mut has_duplicates = false;
    
        for value in &self.state_values {
            // Convert f64 to string for comparison
            let value_str = format!("{:.15}", value); // Ensure precision in comparison
            if !seen_values.insert(value_str) {
                has_duplicates = true;
                break;
            }
        }

        self.state_values_valid = !has_duplicates;


        // Check if state noises are valid (larger than 0)
        self.state_noises_valid = true;
        self.state_noises.iter().for_each(|noise| self.state_noises_valid &= *noise > 0.0);



    }

    fn check_start_matrix_validity(&mut self) {
        self.start_matrix_valid = true;
        self.start_matrix_valid &= self.start_matrix.len() == self.state_values.len();
        self.start_matrix.iter().for_each(|value| self.start_matrix_valid &= *value >= 0.0 && *value <= 1.0);
        self.start_matrix_valid &= self.start_matrix.iter().sum::<f64>() == 1.0;
    }

    fn check_transition_matrix_validity(&mut self) {
        self.transition_matrix_valid = true;

        // Check for correct number of rows and columns
        self.transition_matrix_valid &= self.transition_matrix.len() == self.state_values.len();
        // if !self.transition_matrix_valid {
        //     println!("Failed on row count");
        // }
        let prev = self.transition_matrix_valid;
        self.transition_matrix.iter().for_each(|row| self.transition_matrix_valid &= row.len() == self.state_values.len());
        // if !self.transition_matrix_valid && prev {
        //     println!("Failed on column count");
        // }

        let prev = self.transition_matrix_valid;
        // Check for correct values
        for row in &self.transition_matrix {
            for value in row {
                self.transition_matrix_valid &= *value >= 0.0 && *value <= 1.0;
            }
        }
        // if !self.transition_matrix_valid && prev {
        //     println!("Failed on limits");
        // }

        let prev = self.transition_matrix_valid;
        for row in &self.transition_matrix {
            self.transition_matrix_valid &= row.iter().sum::<f64>() == 1.0;
        }
        // if !self.transition_matrix_valid && prev {
        //     println!("Failed on sum");
        // }
    }


    fn show_states_error_messages(&mut self, ui: &mut egui::Ui) {
        if !self.run_has_been_pressed {return;}

        // Check and display errors if no states
        if self.states_empty {
            ui.colored_label(egui::Color32::RED, "Error: Must have at least one state.");
        }

        // Check and display errors for state values
        if !self.state_values_valid {
            ui.colored_label(egui::Color32::RED, "Error: State values cannot contain duplicates.");
        }

        ui.add_space(10.0);

        // Check and display errors for state noises
        if !self.state_noises_valid {
            ui.colored_label(egui::Color32::RED, "Error: State noise must be > 0.");
        }
    }

    fn show_start_matrix_error_messages(&mut self, ui: &mut egui::Ui) {
        if !self.run_has_been_pressed {return;}

        if !self.start_matrix_valid {
            ui.group(|ui| {
                ui.label(egui::RichText::new("Start Matrix Error:").color(egui::Color32::RED).size(16.0));
                ui.add_space(5.0);
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("• The start matrix must have {} elements.", self.state_values.len()),
                );
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "• Each value must be a probability in the range [0.0, 1.0].",
                );
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "• The values must sum to exactly 1.0.",
                );
            });
        }
    }

    fn show_transition_matrix_error_messages(&mut self, ui: &mut egui::Ui) {
        if !self.run_has_been_pressed {
            return;
        }
    
        if !self.transition_matrix_valid {
            ui.group(|ui| {
                ui.label(egui::RichText::new("Transition Matrix Error:").color(egui::Color32::RED).size(16.0));
                ui.add_space(5.0);
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("• The transition matrix must have shape {} x {}.", self.state_values.len(), self.state_values.len()),
                );
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "• Each value must be a probability in the range [0.0, 1.0].",
                );
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "• Each row must sum to exactly 1.0.",
                );
            });
        }
    }

    fn check_overall_validity(&self) -> bool {
        match &self.hmm_input_hint {
            Some(HMMInputHint::NumStatesFinder) => true,
            Some(HMMInputHint::Initializer) => {
                let check = self.num_states.map(|value| value > 0);
                // println!("Check: {:?}", check);
                check.unwrap_or(false)
            }
            Some(HMMInputHint::Learner { learner_type }) => {
                match learner_type {
                    LearnerType::AmalgamIdea => true,
                    LearnerType::BaumWelch => self.state_values_valid && self.state_noises_valid && !self.states_empty,
                }
            }
            Some(HMMInputHint::Analyzer) => {
                !self.states_empty &&
                self.state_values_valid &&
                self.state_noises_valid &&
                self.start_matrix_valid &&
                self.transition_matrix_valid
            },

            None => false,

        }
    }

    fn construct_hmm_input(&mut self) {
        self.hmm_input = match &self.hmm_input_hint {
            Some(HMMInputHint::NumStatesFinder) => Some(HMMInput::NumStatesFinder {
                sequence_set: self.sequence_set.as_ref().unwrap().clone(),
            }),
            Some(HMMInputHint::Initializer) => Some(HMMInput::Initializer {
                num_states: self.num_states.expect("How the hell did we get here without setting the number of states?"),
                sequence_set: self.sequence_set.as_ref().unwrap().clone(),
            }),
            Some(HMMInputHint::Learner { learner_type }) => {
                let sequence_set = self.sequence_set.as_ref().unwrap().clone();
                let num_states: usize;
                let learner_init: LearnerSpecificInitialValues;

                match learner_type {
                    LearnerType::AmalgamIdea => {
                        num_states = self.num_states.expect("How the hell did we get here without setting the number of states?");
                        learner_init = LearnerSpecificInitialValues::AmalgamIdea {
                            initial_distributions: None,
                        };
                    }
                    LearnerType::BaumWelch => {
                        num_states = self.state_values.len();
                        learner_init = LearnerSpecificInitialValues::BaumWelch {
                            states: HMMInstance::generate_state_set(&self.state_values, &self.state_noises).unwrap(),
                            start_matrix: Some(StartMatrix::new(self.start_matrix.clone())),
                            transition_matrix: Some(TransitionMatrix::new(self.transition_matrix.clone())),
                        };
                    }
                }

                Some(HMMInput::Learner {
                    num_states,
                    sequence_set,
                    learner_init,
                })

            }
            
            Some(HMMInputHint::Analyzer) => Some(HMMInput::Analyzer {
                sequence_set: self.sequence_set.as_ref().unwrap().clone(),
                states: HMMInstance::generate_state_set(&self.state_values, &self.state_noises).unwrap(),
                start_matrix: Some(StartMatrix::new(self.start_matrix.clone())),
                transition_matrix: Some(TransitionMatrix::new(self.transition_matrix.clone())),
            }),
            None => None,
        };
    }
}

fn parse_python_list(input: &str) -> Option<Vec<f64>> {
    // Remove surrounding brackets and split by commas
    let trimmed = input.trim().trim_start_matches('[').trim_end_matches(']');
    let values: Result<Vec<f64>, _> = trimmed
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect();

    values.ok()
}

fn parse_python_matrix(input: &str) -> Option<Vec<Vec<f64>>> {
    // Remove the outer brackets and trim
    let trimmed = input.trim().trim_start_matches('[').trim_end_matches(']');
    
    // Split the content into rows
    let rows: Vec<&str> = trimmed
        .split("],")
        .map(|row| row.trim().trim_start_matches('[').trim_end_matches(']'))
        .collect();

    // Parse each row as a Python list
    let parsed_rows: Option<Vec<Vec<f64>>> = rows
        .into_iter()
        .map(|row| parse_python_list(row)) // Keep as Option
        .collect(); // Collect into Option<Vec<Vec<f64>>>

    parsed_rows
}

fn correct_probs(row: &Vec<f64>) -> Vec<f64> {
    let no_negatives: Vec<f64> = row.clone().iter().map(|value| if *value < 0.0 {0.0} else {*value}).collect();

    let sum: f64 = no_negatives.iter().copied().sum();
    if sum == 0.0 {
        vec![1.0 / (no_negatives.len() as f64); no_negatives.len()] // If the sum is 0, return a zero vector
    } else {
        no_negatives.iter().map(|&v| v / sum).collect()
    }
}

fn reset_start_matrix(num_states: usize) -> Vec<f64> {
    let n = if num_states > 0 {num_states} else {0};
    vec![1.0 / n as f64; n]
}

fn reset_transition_matrix(num_states: usize) -> Vec<Vec<f64>> {
    let n = if num_states > 0 {num_states} else {0};
    vec![vec![1.0 / n as f64; n]; n]
}

fn format_start_matrix(start_matrix: &Vec<f64>) -> String {
    format!(
        "[{}]",
        start_matrix
            .iter()
            .map(|v| format!("{:.6}", v))
            .collect::<Vec<String>>()
            .join(", ")
    )
}

fn format_transition_matrix(transition_matrix: &Vec<Vec<f64>>) -> String {
    format!(
        "[{}]",
        transition_matrix
            .iter()
            .map(|row| {
                format!(
                    "[{}]",
                    row.iter()
                        .map(|v| format!("{:.6}", v))
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            })
            .collect::<Vec<String>>()
            .join(",\n  ")
    )
}