use eframe::egui::{self, RichText};
use egui_plot::{Line, Plot, PlotPoints, BarChart, Bar};
use crate::{
    signal_analysis::hmm::hmm_struct::HMM,
    trace_selection::{individual_trace::TraceType, set_of_points::SetOfPoints}
};
use super::app::{Tab, TabOutput};

pub struct RawAnalysis {
    selected_trace_key: Option<String>,
    bin_count: usize,
}

impl Default for RawAnalysis {
    fn default() -> Self {
        Self {
            selected_trace_key: None,
            bin_count: 50, // Default number of bins
        }
    }
}

impl Tab for RawAnalysis {
    fn render(
        &mut self,
        ctx: &egui::Context,
        _hmm: &mut HMM,
        preprocessing: &mut SetOfPoints,
        _logs: &mut Vec<String>,
    ) -> Option<TabOutput> {
        egui::CentralPanel::default().show(ctx, |ui| {
            let points = preprocessing.get_points();
            if points.is_empty() {
                ui.label("No traces available to display.");
                return;
            }

            // Initialize selected trace if not already set
            if self.selected_trace_key.is_none() {
                self.selected_trace_key = points.keys().next().cloned();
            }

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label("Select a trace:");

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

            // Scrollable area for the content
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                        ui.add_space(30.0); // Add fixed left margin
                        ui.vertical(|ui| {
                            self.render_trace_plot(ui, preprocessing);
                        });
                    });
                });
        });

        None
    }
}

impl RawAnalysis {
    /// Render the trace plot and related information in a separate function.
    fn render_trace_plot(&mut self, ui: &mut egui::Ui, preprocessing: &mut SetOfPoints) {
        let points = preprocessing.get_points();

        if let Some(key) = &self.selected_trace_key {
            if let Some(point_trace) = points.get(key) {
                if let Ok(fret_values) = point_trace.get_valid_fret() {
                    ui.heading("FRET values");

                    let plot_points: PlotPoints = PlotPoints::from_iter(
                        fret_values.iter().enumerate().map(|(i, &y)| [i as f64, y]),
                    );

                    Plot::new("trace_plot")
                        .height(400.0)
                        .allow_drag(false)
                        .allow_scroll(false)
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(plot_points));
                        });

                    ui.add_space(10.0);
                    ui.label(format!("Number of data points (selected trace): {}", fret_values.len()));

                    ui.add_space(20.0);
                    ui.heading("Histogram of All FRET values (Combined)");

                    // Slider for adjusting bin count
                    ui.horizontal(|ui| {
                        ui.label("Number of bins:");
                        ui.add(egui::Slider::new(&mut self.bin_count, 1..=200).text("bins"));
                    });

                    // Collect all FRET values from every trace
                    let all_fret_values = self.collect_all_fret_values(preprocessing);

                    // Render the histogram from the combined FRET values
                    self.render_histogram(ui, &all_fret_values);
                } else {
                    ui.label("FRET trace is missing for the selected trace.");
                }
            } else {
                ui.label("Selected trace not found.");
            }
        }
    }

    /// Collect all FRET values from all traces in preprocessing
    fn collect_all_fret_values(&self, preprocessing: &SetOfPoints) -> Vec<f64> {
        let points = preprocessing.get_points();
        let mut all_values = Vec::new();
        for (_, point_trace) in points {
            if let Ok(fret_values) = point_trace.get_valid_fret() {
                all_values.extend_from_slice(&fret_values);
            }
        }
        all_values
    }

    /// Render a histogram from the given values.
    fn render_histogram(&self, ui: &mut egui::Ui, values: &[f64]) {
        if values.is_empty() {
            ui.label("No data to create a histogram.");
            return;
        }

        let bins = self.bin_count;
        let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_value - min_value).abs() < f64::EPSILON {
            ui.label("All values are identical, histogram not meaningful.");
            return;
        }

        let bin_width = (max_value - min_value) / bins as f64;
        let mut counts = vec![0; bins];

        for &v in values {
            let mut bin_idx = ((v - min_value) / bin_width) as usize;
            if bin_idx >= bins {
                bin_idx = bins - 1; // handle edge cases
            }
            counts[bin_idx] += 1;
        }

        let total_count = values.len();

        let mut bars = Vec::new();
        for i in 0..bins {
            let x = min_value + (i as f64 + 0.5)*bin_width; // center of the bin
            let height = counts[i] as f64 / total_count as f64 * 100.0;
            bars.push(Bar::new(x, height));
        }

        let chart = BarChart::new(bars).width(bin_width * 0.8);

        Plot::new("histogram_plot")
            .height(200.0)
            .allow_drag(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(chart);
            });
    }
}