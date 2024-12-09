use eframe::egui;
use crate::signal_analysis::hmm::hmm_struct::HMM;
use crate::signal_analysis::hmm::{InitializationMethods, LearnerType, StateValueInitMethod};
use crate::trace_selection::point_traces::PointTracesError;
use crate::trace_selection::set_of_points::{SetOfPoints, SetOfPointsError};

use super::app::{Tab, TabOutput};
use super::filter_settings_window::FilterSettingsWindow;
use super::init_settings_window::InitializationSettingsWindow;
use super::learn_settings_window::LearnSettingsWindow;
use super::load_traces_window::LoadTracesWindow;
use super::nsf_settings_window::FindNumberOfStatesWindow;
use super::run_signal_analysis_window::RunSignalProcessingWindow;

pub struct MainTab {
    filter_enabled: bool,
    filter_performed: bool,
    received_sequence_set: Option<Vec<Vec<f64>>>,

    learn_enabled: bool,
    initialize_enabled: bool,
    num_states_find_enabled: bool,

    // Settings windows
    filter_settings_window: FilterSettingsWindow,
    learn_settings_window: LearnSettingsWindow,
    initialize_settings_window: InitializationSettingsWindow,
    nsf_seettings_window: FindNumberOfStatesWindow,

    // Trace loading window
    load_traces_window: LoadTracesWindow,

    // Signal Analysis run window
    run_signal_analysis_window: RunSignalProcessingWindow,

}

impl Default for MainTab {
    fn default() -> Self {
        Self {
            filter_enabled: true,
            filter_performed: false,
            received_sequence_set: None,

            learn_enabled: true,
            initialize_enabled: true,
            num_states_find_enabled: false,
            filter_settings_window: FilterSettingsWindow::new(),
            learn_settings_window: LearnSettingsWindow::new(),
            initialize_settings_window: InitializationSettingsWindow::new(),
            nsf_seettings_window: FindNumberOfStatesWindow::new(),
            load_traces_window: LoadTracesWindow::new(),
            run_signal_analysis_window: RunSignalProcessingWindow::new(),

            
        }
    }
}



impl Tab for MainTab {
    fn render(&mut self, ctx: &egui::Context, hmm: &mut HMM, preprocessing: &mut SetOfPoints, logs: &mut Vec<String>) -> Option<TabOutput>{
        // The global styles are now applied in MyApp, so we don't need to call apply_global_styles here.

        // 1. Top Panel: Load Traces Button
        self.top_panel(ctx);

        // 2. Central Panel: Contains Side Panels
        self.central_panel(ctx, preprocessing, hmm, logs);

        // 3. Bottom Panel: Console Output (Resizable)
        self.bottom_panel(ctx, logs);

        // 4. Render the Filter Settings Window
        self.filter_settings_window.show(ctx, preprocessing);

        // 5. Render the Learn Settings Window
        self.learn_settings_window.show(ctx);

        // 6. Render the Initialize Settings Window
        self.initialize_settings_window.show(ctx);

        // 7. Render the Find Number of States Settings Window
        self.nsf_seettings_window.show(ctx);

        // 8. Render the Trace Loading window
        self.load_traces_window.show(ctx, preprocessing);

        // 9. Render the Run Signal Analysis window
        let hmm_input_ready = self.run_signal_analysis_window.show(ctx);

        if let Some(hmm_input) = hmm_input_ready {
            println!("Received input: {:?}", hmm_input);

            self.setup_hmm(hmm);

            println!("Setup hmm: {:?}", hmm);

            return Some(TabOutput::Main { hmm_input })
        }

        return None;



        // println!("Showing current HMM setup\n");
        // println!("Learner:");
        // println!("Active?: {}", self.learn_enabled);
        // if self.learn_enabled {
        //     println!("Learner type: {:?}", self.learn_settings_window.learner_type);
        //     println!("Learner setup: {:?}", self.learn_settings_window.learner_setup);
        // }
        // println!("\n");
        // println!("Initializer:");
        // println!("Active?: {}", self.initialize_enabled);
        // if self.initialize_enabled {
        //     println!("Initializer setup:");
        //     println!("-    Values init:       {:?}",  self.initialize_settings_window.state_value_init);
        //     println!("-    Noise init:        {:?}",  self.initialize_settings_window.state_noise_init);
        //     println!("-    Start matrix init: {:?}",  self.initialize_settings_window.start_matrix_init);
        //     println!("-    Trans matrix init: {:?}",  self.initialize_settings_window.transition_matrix_init);
        // }
        // println!("\n");
        // println!("Num States Finder:");
        // println!("Active?: {}", self.num_states_find_enabled);
        // if self.num_states_find_enabled {
        //     println!("Strategy: {:?}", self.nsf_seettings_window.strategy);
        // }
        
        
        // println!("filter: {:?}", preprocessing.get_filter_setup());

    }
}

impl MainTab {
    /// Draw the top panel with the "Load Traces" button.
    fn top_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_panel")
            .show(ctx, |ui| {
                ui.add_space(15.0); // Add padding at the top
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add padding on the left
                    if ui.button("+ Load Traces").clicked() {
                        self.load_traces_window.open();
                    }
                });
                ui.add_space(10.0);
            });
    }

    /// Draw the central panel containing side panels.
    fn central_panel(&mut self, ctx: &egui::Context, preprocessing: &mut SetOfPoints, hmm: &mut HMM, logs: &mut Vec<String>) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Use ui.columns to divide the space into two columns
            ui.columns(2, |columns| {
                // Left Panel Content
                columns[0].vertical(|ui| {
                    self.preprocessing_panel_content(ui, preprocessing, logs);
                });

                // Right Panel Content
                columns[1].vertical(|ui| {
                    self.signal_analysis_panel_content(ui, hmm, logs);
                });
            });
        });
    }

    fn bottom_panel(&self, ctx: &egui::Context, logs: &mut Vec<String>) {
        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(true)
            .default_height(200.0)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.add_space(10.0);
                    ui.heading("Console output");
                    ui.separator();
    
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            // Save the original spacing (global/default settings)
                            let original_spacing = ui.spacing().clone(); // Clone ensures a copy is saved
    
                            // Temporarily reduce spacing for Monospace text
                            ui.spacing_mut().item_spacing.y = 2.0; // Smaller gap between lines for console-like text
    
                            // Render console-like output
                            for log in logs {
                                ui.monospace(log);
                            }
    
                            // Restore the original spacing after Monospace text
                            *ui.spacing_mut() = original_spacing;
                        });
                });
            });
    }

    /// Content of the preprocessing panel.
    fn preprocessing_panel_content(&mut self, ui: &mut egui::Ui, preprocessing: &mut SetOfPoints, logs: &mut Vec<String>) {
        ui.add_space(10.0); // Add padding at the top
        ui.vertical(|ui| {
            ui.add_space(10.0); // Add left padding
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Left padding for the button
                if ui.button("Run Preprocessing").clicked() {
                    // Check if files have been loaded
                    if !self.has_files_loaded(preprocessing) {
                        logs.push("Preprocessing failed: No files have been loaded.".to_string());
                        return; // Exit early if no files have been loaded
                    }
                
                    // Update flag for checking if filtering has been performed
                    if self.filter_enabled {
                        self.filter_performed = true;
                    }
                
                    let sequence_set_result = run_preprocessing(self.filter_enabled, preprocessing);
                
                    if let Ok(sequence_set) = sequence_set_result {
                        // Check if sequences are empty
                        let mut is_full = true;
                        is_full &= !sequence_set.is_empty();
                        is_full &= !sequence_set.iter().any(|sequence| sequence.is_empty());
                
                        if is_full {
                            self.received_sequence_set = Some(sequence_set);
                            // Update logs
                            logs.push("Preprocessing completed successfully.".to_string());
                        }

                        print_rejected_points(&preprocessing, logs);
                
                        
                    } else {
                        // Update logs with the error
                        logs.push("Preprocessing failed: Could not get any valid FRET sequence.".to_string());
                    }
                }
            });

            ui.add_space(10.0); // Spacing between elements
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Horizontal padding inside the layout
                ui.checkbox(&mut self.filter_enabled, "Filter");
                ui.add_space(10.0);
                // Conditionally render the "See options" link only if the checkbox is checked
                if self.filter_enabled {
                    ui.add_space(10.0); // Spacing between checkbox and link
                    if ui
                        .link(egui::RichText::new("See options").size(12.0))
                        .clicked()
                    {
                        self.filter_settings_window.open(&preprocessing);
                    }
                }
            });


            // Add conditional label if filtering has been performed
            if self.filter_performed {
                ui.add_space(5.0); // Add spacing before the label
                ui.label(egui::RichText::new("Filtering has been performed").size(12.0));
            }
        });
    }

    /// Content of the signal analysis panel.
    fn signal_analysis_panel_content(&mut self, ui: &mut egui::Ui, hmm: &mut HMM, logs: &mut Vec<String>) {
        ui.add_space(10.0); // Add padding at the top
        ui.vertical(|ui| {
            ui.add_space(10.0); // Add left padding
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Left padding for the button
                if ui.button("Run Signal Analysis").clicked() {
                    if self.received_sequence_set.is_none() {
                        logs.push("Could not un signal analysis, since no valid fret sequences are available.".to_string());
                        return;
                    }

                    let sequence_set = self.received_sequence_set.as_ref().unwrap().clone();
                    let input_hint: HMMInputHint;

                    if self.num_states_find_enabled {
                        input_hint = HMMInputHint::NumStatesFinder;
                    } else if self.initialize_enabled {
                        let state_hints_num: Option<usize>;
                        if let StateValueInitMethod::StateHints { state_values } = 
                        &self.initialize_settings_window.state_value_init {
                            state_hints_num = Some(state_values.len());
                        } else {
                            state_hints_num = None;
                        }
                        input_hint = HMMInputHint::Initializer {state_hints_num};
                    } else if self.learn_enabled {
                        let learner_type = self.learn_settings_window.learner_type.clone();
                        input_hint = HMMInputHint::Learner { learner_type };
                    } else {
                        input_hint = HMMInputHint::Analyzer;
                    }

                    self.run_signal_analysis_window.open(sequence_set, input_hint);
                }
            });

            ui.add_space(10.0); // Spacing between elements
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Left padding for the label
                ui.label("Setup Components:");
            });
            ui.add_space(10.0); // Spacing between elements

            // Learn Component
            if self.learn_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("x").color(egui::Color32::RED))
                        .clicked()
                    {
                        self.learn_enabled = false;
                        self.initialize_enabled = false;
                        self.num_states_find_enabled = false;
                    }
                    ui.label("Learn");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                        ui.add_space(20.0);

                        if ui
                            .link(egui::RichText::new("See options").size(12.0))
                            .clicked()
                        {
                            println!("Got before calling the opening of the window");
                            self.learn_settings_window.open(&hmm);
                        }
                    });
                });
            } else {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("+").color(egui::Color32::GREEN))
                        .clicked()
                    {
                        self.learn_enabled = true;
                    }
                    ui.label("Learn");
                });
            }

            // Initialize Component
            if self.initialize_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("x").color(egui::Color32::RED))
                        .clicked()
                    {
                        self.initialize_enabled = false;
                        self.num_states_find_enabled = false;
                    }
                    ui.label("Initialize");
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                        ui.add_space(20.0);
                        if ui
                            .link(egui::RichText::new("See options").size(12.0))
                            .clicked()
                        {
                            self.initialize_settings_window.open();
                        }
                    });

                    
                    

                });
            } else if self.learn_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("+").color(egui::Color32::GREEN))
                        .clicked()
                    {
                        self.initialize_enabled = true;
                    }
                    ui.label("Initialize");
                });
            }

            // Find Number of States Component
            if self.num_states_find_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("x").color(egui::Color32::RED))
                        .clicked()
                    {
                        self.num_states_find_enabled = false;
                    }
                    ui.label("Find Number of States");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                        ui.add_space(20.0);
                        if ui
                            .link(egui::RichText::new("See options").size(12.0))
                            .clicked()
                        {
                            self.nsf_seettings_window.open();
                        }
                    });
                });
            } else if self.initialize_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(10.0); // Add horizontal padding inside the layout
                    if ui
                        .button(egui::RichText::new("+").color(egui::Color32::GREEN))
                        .clicked()
                    {
                        self.num_states_find_enabled = true;
                    }
                    ui.label("Find Number of States");
                });
            }
        });
    }

    /// Check if files have been loaded for preprocessing
    fn has_files_loaded(&self, preprocessing: &SetOfPoints) -> bool {
        !preprocessing.get_points().is_empty()
    }

    /// Build the HMM
    fn setup_hmm(&self, hmm: &mut HMM) {
        
        // Check if learner is enabled and get its setup
        if self.learn_enabled {
            // Retrieve stuff from the learner settings window
            let learner_type = self.learn_settings_window.learner_type.clone();
            let learner_setup = self.learn_settings_window.learner_setup.clone();

            // Set the retrieved stuff in the hmm object
            hmm.add_learner();
            hmm.set_learner_type(learner_type).unwrap();
            hmm.setup_learner(learner_setup).unwrap();
        }


        // Check if initializer is enabled and get its setup
        if self.initialize_enabled {
            // Retrieve relevant stuff
            let state_values_init = self.initialize_settings_window.state_value_init.clone();
            let state_noise_init = self.initialize_settings_window.state_noise_init.clone();
            let start_matrix_init = self.initialize_settings_window.start_matrix_init.clone();
            let transition_matrix_init = self.initialize_settings_window.transition_matrix_init.clone();

            let init_methods = InitializationMethods {
                state_values_method: Some(state_values_init),
                state_noises_method: Some(state_noise_init),
                start_matrix_method: Some(start_matrix_init),
                transition_matrix_method: Some(transition_matrix_init),
            };

            hmm.add_initializer().unwrap();
            hmm.set_initialization_method(init_methods).unwrap();
        }


        // Check if num states finder is enabled and get its setup
        if self.num_states_find_enabled {
            // Retrieve relevant stuff
            let strategy = self.nsf_seettings_window.strategy.clone();

            hmm.add_number_of_states_finder().unwrap();
            hmm.set_state_number_finder_strategy(strategy).unwrap();
        }
    }
}



fn run_preprocessing(filter: bool, preprocess: &mut SetOfPoints) -> Result<Vec<Vec<f64>>, SetOfPointsError> {
    if filter {
        preprocess.filter(); // Already includes photobleaching detection
    }
    preprocess.get_valid_fret()
}

fn print_rejected_points(preprocess: &SetOfPoints, logs: &mut Vec<String>) {
    // Retrieve the rejected points
    let rejected_points = preprocess.get_rejected_points();

    if rejected_points.is_none() {
        println!("No points were rejected.");
        logs.push("No points were rejected.".to_string());
        return;
    }

    let rejected_points = rejected_points.unwrap();

    if rejected_points.is_empty() {
        println!("No rejected points found.");
        logs.push("No rejected points found.".to_string());
        return;
    }

    // Print the rejected points and their errors
    println!("Rejected Points and Failed Tests:");
    logs.push("Rejected Points and Failed Tests:".to_string());

    for (file_path, (_point_trace, error)) in rejected_points {
        println!("File: {}", file_path);
        logs.push(format!("File: {}", file_path));

        match error {
            PointTracesError::FailedFilterTests { tests_failed } => {
                println!("  Failed Tests:");
                logs.push("  Failed Tests:".to_string());

                for test in tests_failed {
                    println!("    - {:?}", test);
                    logs.push(format!("    - {:?}", test));
                }
            }
            _ => {
                println!("  Error: {:?}", error);
                logs.push(format!("  Error: {:?}", error));
            }
        }
    }
}

#[derive(Clone)]
pub enum HMMInputHint {
    Analyzer,
    Learner{learner_type: LearnerType},
    Initializer{state_hints_num: Option<usize>},
    NumStatesFinder,
}