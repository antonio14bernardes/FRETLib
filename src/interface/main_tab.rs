use eframe::egui;
use crate::signal_analysis::hmm::hmm_struct::HMM;
use crate::trace_selection::set_of_points::{SetOfPoints, SetOfPointsError};

use super::app::Tab;
use super::filter_settings_window::FilterSettingsWindow;
use super::init_settings_window::InitializationSettingsWindow;
use super::learn_settings_window::LearnSettingsWindow;
use super::load_traces_window::LoadTracesWindow;
use super::nsf_settings_window::FindNumberOfStatesWindow;

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

    // Log messages
    logs: Vec<String>,
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

            logs: Vec::new(),
        }
    }
}



impl Tab for MainTab {
    fn render(&mut self, ctx: &egui::Context, hmm: &mut HMM, preprocessing: &mut SetOfPoints) {
        // The global styles are now applied in MyApp, so we don't need to call apply_global_styles here.

        // 1. Top Panel: Load Traces Button
        self.top_panel(ctx);

        // 2. Central Panel: Contains Side Panels
        self.central_panel(ctx, preprocessing, hmm);

        // 3. Bottom Panel: Console Output (Resizable)
        self.bottom_panel(ctx);

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


        println!("filter: {:?}", preprocessing.get_filter_setup());

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
    fn central_panel(&mut self, ctx: &egui::Context, preprocessing: &mut SetOfPoints, hmm: &mut HMM ) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Use ui.columns to divide the space into two columns
            ui.columns(2, |columns| {
                // Left Panel Content
                columns[0].vertical(|ui| {
                    self.preprocessing_panel_content(ui, preprocessing);
                });

                // Right Panel Content
                columns[1].vertical(|ui| {
                    self.signal_analysis_panel_content(ui, hmm);
                });
            });
        });
    }

    fn bottom_panel(&self, ctx: &egui::Context) {
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
                            for log in &self.logs {
                                ui.monospace(log);
                            }
    
                            // Restore the original spacing after Monospace text
                            *ui.spacing_mut() = original_spacing;
                        });
                });
            });
    }

    /// Content of the preprocessing panel.
    fn preprocessing_panel_content(&mut self, ui: &mut egui::Ui, preprocessing: &mut SetOfPoints) {
        ui.add_space(10.0); // Add padding at the top
        ui.vertical(|ui| {
            ui.add_space(10.0); // Add left padding
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Left padding for the button
                if ui.button("Run Preprocessing").clicked() {
                    // Check if files have been loaded
                    if !self.has_files_loaded(preprocessing) {
                        self.logs.push("Preprocessing failed: No files have been loaded.".to_string());
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
                        }
                
                        // Update logs
                        self.logs.push("Preprocessing completed successfully.".to_string());
                    } else {
                        // Update logs with the error
                        self.logs.push("Preprocessing failed: Could not get FRET values.".to_string());
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
    fn signal_analysis_panel_content(&mut self, ui: &mut egui::Ui, hmm: &mut HMM) {
        ui.add_space(10.0); // Add padding at the top
        ui.vertical(|ui| {
            ui.add_space(10.0); // Add left padding
            ui.horizontal(|ui| {
                ui.add_space(10.0); // Left padding for the button
                if ui.button("Run Signal Analysis").clicked() {
                    // Add signal analysis logic here
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
}



fn run_preprocessing(filter: bool, preprocess: &mut SetOfPoints) -> Result<Vec<Vec<f64>>, SetOfPointsError> {
    if filter {
        preprocess.filter(); // Already includes photobleaching detection
    }
    preprocess.get_valid_fret()

    
  
}