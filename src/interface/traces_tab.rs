use eframe::egui::{self, RichText};
use egui_plot::{Line, Plot, PlotPoints};use crate::{signal_analysis::hmm::hmm_struct::HMM, trace_selection::{individual_trace::TraceType, set_of_points::SetOfPoints}};
use super::app::{Tab, TabOutput};

pub struct TracesTab {
    selected_trace_key: Option<String>,
}

impl Default for TracesTab {
    fn default() -> Self {
        Self {
            selected_trace_key: None,
        }
    }
}

impl Tab for TracesTab {
    fn render(
        &mut self,
        ctx: &egui::Context,
        _hmm: &mut HMM,
        preprocessing: &mut SetOfPoints,
        _logs: &mut Vec<String>,
    ) -> Option<TabOutput> {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Check if there are any traces available
            let points = preprocessing.get_points();
            if points.is_empty() {
                ui.label("No traces available to display.");
                return;
            }

            // Initialize selected trace
            if self.selected_trace_key.is_none() {
                self.selected_trace_key = points.keys().next().cloned();
            }

            ui.horizontal(|ui| {
                ui.label("Select a trace:");

                // Custom ComboBox with smaller font and text wrapping
                egui::ComboBox::from_label("")
                    .selected_text(
                        RichText::new(self.selected_trace_key.clone().unwrap_or_default()).size(12.0),
                    )
                    .show_ui(ui, |ui| {
                        for key in points.keys() {
                            let display_text = RichText::new(key).size(12.0);
                            if ui
                                .selectable_label(self.selected_trace_key.as_ref() == Some(key), display_text)
                                .clicked()
                            {
                                self.selected_trace_key = Some(key.clone());
                            }
                        }
                    });
            });

            ui.add_space(20.0);

            // Left-to-right layout with explicit left padding
            ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                ui.add_space(30.0); // Add fixed left margin to center the plot visually

                ui.vertical(|ui| {
                    if let Some(key) = &self.selected_trace_key {
                        if let Some(point_trace) = points.get(key) {
                            if let Ok(fret_values) = point_trace.get_valid_fret() {

                                ui.heading("FRET values");
                                

                                let plot_points: PlotPoints = PlotPoints::from_iter(
                                    fret_values.iter().enumerate().map(|(i, &y)| [i as f64, y]),
                                );


                                Plot::new("trace_plot")
                                    
                                    .height(340.0)
                                    // .width(720.0) // Explicit plot width
                                    // .allow_zoom(true)
                                    // .allow_drag(false)
                                    
                                    // .grid_spacing(20.0..=30.0)
                                    // .x_grid_spacer(Box::new(|_| Vec::new()))
                                    // .show_x(false) // Hide the default axis
                                    .show(ui, |plot_ui| {
                                        plot_ui.line(Line::new(plot_points));
                                    });

                                ui.add_space(10.0);
                                ui.label(format!("Number of data points: {}", fret_values.len()));
                            } else {
                                ui.label("FRET trace is missing for the selected trace.");
                            }
                        } else {
                            ui.label("Selected trace not found.");
                        }
                    }
                });
            });
        });

        None
    }
}