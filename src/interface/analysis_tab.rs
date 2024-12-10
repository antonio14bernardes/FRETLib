use eframe::egui::{self, Color32, RichText, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};
use egui_heatmap::{BitmapWidget, Data, MultiBitmapWidgetSettings};

use std::f64::consts::PI;
use crate::signal_analysis::hmm;
use crate::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use crate::signal_analysis::hmm::{hmm_struct::HMM, state::State};
use crate::trace_selection::set_of_points::SetOfPoints;
use super::app::{Tab, TabOutput};
use super::reporter::Reporter;

const COLORS: [Color32; 10] = [
    Color32::RED,
    Color32::GREEN,
    Color32::BLUE,
    Color32::YELLOW,
    Color32::LIGHT_BLUE,
    Color32::LIGHT_GREEN,
    Color32::GOLD,
    Color32::LIGHT_RED,
    Color32::LIGHT_GRAY,
    Color32::WHITE,
];


pub struct AnalysisTab{
    selected_trace_key: Option<String>,
}

impl Default for AnalysisTab {
    fn default() -> Self {
        Self {
            selected_trace_key: None,
        }
    }
}

impl Tab for AnalysisTab {
    fn render(
        &mut self,
        ctx: &egui::Context,
        hmm: &mut HMM,
        preprocessing: &mut SetOfPoints,
        logs: &mut Vec<String>,
    ) -> Option<TabOutput> {

        let hmm_analyzer: &hmm::analysis::hmm_analyzer::HMMAnalyzer = hmm.get_analyzer();
        let states: Option<&Vec<State>> = hmm_analyzer.get_states();
        let state_occupancy: Option<&Vec<f64>> = hmm_analyzer.get_state_occupancy();
        let start_matrix: Option<&StartMatrix> = hmm_analyzer.get_start_matrix();
        let transition_matrix: Option<&TransitionMatrix> = hmm_analyzer.get_transition_matrix();

        let state_id_sequences: Option<&Vec<Vec<usize>>> = hmm_analyzer.get_state_sequences();
        let sequences: Option<&Vec<Vec<f64>>> = hmm_analyzer.get_sequences();
        
        // let traces: &std::collections::HashMap<String, crate::trace_selection::point_traces::PointTraces> = preprocessing.get_points();

        egui::CentralPanel::default().show(ctx, |ui| {

            if ui.button("Generate Report").clicked() {
                if let Some(reporter) = Reporter::new(logs) {
                    reporter.report(hmm_analyzer, &preprocessing, logs).unwrap();
                    println!("Reporter created at: {:?}", reporter.path);
                } else {
                    println!("Reporter creation was canceled.");
                }
            }

            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.visualize_states(ui, states);

                    self.visualize_state_occupancy(ui, state_occupancy);

                    ui.separator();

                    self.visualize_start_matrix(ui, start_matrix);
                    ui.separator();

                    self.visualize_transition_matrix(ui, transition_matrix);
                    ui.separator();

                    self.render_trace_with_idealized(ui, preprocessing, state_id_sequences, sequences, states);
                    
                });

            
        });

        None
    }
}

impl AnalysisTab {
    fn visualize_states(&mut self, ui: &mut egui::Ui, states: Option<&Vec<State>>) {
        if let Some(states) = states {
            if states.is_empty() {
                ui.label("No states available.");
                return;
            }

            // Show a table of states
            ui.heading("State Information");
            egui::Grid::new("state_info_grid")
                .striped(true)
                .spacing([40.0, 4.0])
                .show(ui, |ui| {
                    ui.heading("ID");
                    ui.heading("Mean");
                    ui.heading("Std");
                    // ui.heading("Name");
                    ui.end_row();

                    for state in states {
                        ui.label(format!("{}", state.id));
                        ui.label(format!("{:.3}", state.value));
                        ui.label(format!("{:.3}", state.noise_std));
                        // ui.label(state.name.clone().unwrap_or_else(|| "N/A".to_string()));
                        ui.end_row();
                    }
                });

            ui.separator();

            ui.heading("State Distributions");

            // Plotting parameters
            let num_points = 200; // kinda like resolution
            let num_stds = 3.0; // ±3 stds by default
            

            // Create a vector of lines from states
            let lines: Vec<Line> = states
                .iter()
                .enumerate()
                .map(|(i, state)| {
                    let color = COLORS[i % COLORS.len()];
                    self.gaussian_line_for_state(state, num_points, num_stds, color)
                })
                .collect();

            // Show the lines in a single plot
            Plot::new("states_gaussians")
                .legend(Legend::default())
                .allow_drag(false)
                .allow_scroll(false)
                .show(ui, |plot_ui| {
                    for line in lines {
                        plot_ui.line(line);
                    }
                });

        } else {
            ui.label("No states available.");
        }
    }

    fn visualize_state_occupancy(&mut self, ui: &mut egui::Ui, state_occupancy: Option<&Vec<f64>>) {
        if let Some(occupancies) = state_occupancy {
            if occupancies.is_empty() {
                ui.label("No state occupancy data available.");
                return;
            }
    
            ui.heading("State Occupancy");
    
            // Create bars for each state
            let bars: Vec<Bar> = occupancies
            .iter()
            .enumerate()
            .map(|(i, &occupancy)| {
                let color = COLORS[i % COLORS.len()]; // Same color palette
                Bar::new(i as f64, occupancy)
                    .name(format!("State {}", i))
                    .fill(color) // Set bar color
            })
            .collect();
    
            // Set a fixed bar width but this is kinda sus
            let chart = BarChart::new(bars).width(0.5);
    
            Plot::new("state_occupancy_plot")
                .legend(Legend::default())
                .height(300.0)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                // Use x_axis_formatter to convert x-values (0,1,2,...) to "State 0", "State 1", etc.
                .x_axis_formatter(|mark, _range| {
                    let x = mark.value; // Get the f64 value from the GridMark
                    let idx = x.round() as usize;
                    format!("State {}", idx)
                })
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(chart);
                });
    
        } else {
            ui.label("No state occupancy data available.");
        }
    }

    fn visualize_start_matrix(&mut self, ui: &mut egui::Ui, start_matrix: Option<&StartMatrix>) {
        if let Some(start_matrix) = start_matrix {
            let values = &start_matrix.matrix; 
            let size = values.len(); 
    
            if size == 0 {
                ui.label("Empty start matrix.");
                return;
            }
    
            ui.heading("Start Matrix");
    
            egui::Grid::new("start_matrix_grid")
                .striped(true) 
                .spacing([20.0, 8.0]) // Column and row spacing
                .show(ui, |ui| {
                    // Header Row: State IDs
                    ui.label("State ID"); 
                    for col in 0..size {
                        ui.label(format!("State {}", col));
                    }
                    ui.end_row();
    
                    // Row with Start Probabilities
                    ui.label("Start Probabilities");
                    for &value in values {
                        ui.label(format!("{:.3}", value));
                    }
                    ui.end_row();
                });
        } else {
            ui.label("No start matrix data available.");
        }
    }
   
    fn visualize_transition_matrix(&mut self, ui: &mut egui::Ui, transition_matrix: Option<&TransitionMatrix>) {
        if let Some(matrix) = transition_matrix {
            let raw_matrix = &matrix.matrix.raw_matrix;
            let size = raw_matrix.len();
    
            if size == 0 {
                ui.label("Empty transition matrix.");
                return;
            }
    
            ui.heading("Transition Matrix");
    
            egui::Grid::new("transition_matrix_grid")
                .striped(true) // Alternate row coloring
                .spacing([20.0, 8.0]) // Column and row spacing
                .show(ui, |ui| {
                    ui.label("");
                    for col in 0..size {
                        ui.label(format!("State {}", col));
                    }
                    ui.end_row();
    
                    // Matrix Rows
                    for (row_idx, row) in raw_matrix.iter().enumerate() {
                        ui.label(format!("State {}", row_idx)); 
                        for &value in row {
                            ui.label(format!("{:.3}", value));
                        }
                        ui.end_row();
                    }
                });
        } else {
            ui.label("No transition matrix data available.");
        }
    }

    fn gaussian_line_for_state(
        &self,
        state: &State,
        num_points: usize,
        num_stds: f64,
        color: Color32,
    ) -> Line {
        let mean = state.value;
        let std = state.noise_std;
        let x_min = mean - num_stds * std;
        let x_max = mean + num_stds * std;

        let step = (x_max - x_min) / num_points as f64;
        let coeff = 1.0 / (std * (2.0 * PI).sqrt());
        let mut points = Vec::with_capacity(num_points);

        for j in 0..num_points {
            let x = x_min + j as f64 * step;
            let exponent = -((x - mean) * (x - mean)) / (2.0 * std * std);
            let y = coeff * exponent.exp();
            points.push([x, y]);
        }

        Line::new(PlotPoints::from(points))
            .color(color)
            .name(format!("State {}", state.id))
    }

    fn render_trace_with_idealized(
        &mut self,
        ui: &mut egui::Ui,
        preprocessing: &mut SetOfPoints,
        state_id_sequences: Option<&Vec<Vec<usize>>>,
        sequences: Option<&Vec<Vec<f64>>>,
        states: Option<&Vec<State>>,
    ) {
        let traces = preprocessing.get_points();
    
        // Ensure we have all the required inputs
        if let (Some(state_sequences), Some(sequences), Some(states)) = (state_id_sequences, sequences, states) {
            // Initialize selected key if not set
            if self.selected_trace_key.is_none() {
                self.selected_trace_key = traces.keys().next().cloned();
            }
    
            
            ui.horizontal(|ui| {
                ui.label("Select a trace:");

                egui::ComboBox::from_label("")
                    .selected_text(
                        RichText::new(
                            self.selected_trace_key.clone().unwrap_or_default()
                        )
                        .size(12.0), // Smaller text for the selected value
                    )
                    .width(120.0) 
                    .show_ui(ui, |ui| {
                        for key in traces.keys() {
                            let display_text = RichText::new(key).size(12.0); // Smaller text for selectable items
                            if ui
                                .selectable_label(self.selected_trace_key.as_ref() == Some(key), display_text)
                                .clicked()
                            {
                                self.selected_trace_key = Some(key.clone());
                            }
                        }
                    });
            });
    
    
            // Check if a trace key is selected
            if let Some(selected_key) = &self.selected_trace_key {
                if let Some(point_trace) = traces.get(selected_key) {
                    if let Ok(fret_values) = point_trace.get_valid_fret() {
                        // Find matching sequence
                        let matching_idx = sequences.iter().position(|seq| seq == &fret_values);
    
                        if let Some(idx) = matching_idx {    
                            // Plot raw FRET values
                            let fret_plot_points: PlotPoints = PlotPoints::from_iter(
                                fret_values.iter().enumerate().map(|(i, &y)| [i as f64, y]),
                            );
    
                            // Generate idealized state values based on the matching state sequence
                            let idealized_points = PlotPoints::from_iter(
                                state_sequences[idx]
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &state_id)| {
                                        let state_value = states
                                            .get(state_id)
                                            .map(|s| s.value)
                                            .unwrap_or(0.0);
                                        [i as f64, state_value]
                                    }),
                            );
    
                            // Show the plot with both raw FRET values and idealized states
                            Plot::new("trace_with_idealized")
                                .height(300.0)
                                .allow_drag(false)
                                .allow_scroll(false)
                                .legend(Legend::default())
                                .show(ui, |plot_ui| {
                                    plot_ui.line(Line::new(fret_plot_points)
                                        .color(Color32::BLUE)
                                        .name("Raw FRET Trace"));
                                    plot_ui.line(Line::new(idealized_points)
                                        .color(Color32::RED)
                                        .name("Idealized States"));
                                });
                        } else {
                            ui.label("No matching sequence found for this trace.");
                        }
                    } else {
                        ui.label("Invalid FRET data for the selected trace.");
                    }
                } else {
                    ui.label("Selected trace not found.");
                }
            } else {
                ui.label("No trace selected.");
            }
        } else {
            ui.label("Required data (state sequences, sequences, or states) is unavailable.");
        }
    }



}