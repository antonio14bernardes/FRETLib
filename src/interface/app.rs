use eframe::egui;
use crate::signal_analysis::hmm::hmm_struct::HMMInput;
use crate::signal_analysis::hmm::hmm_struct::HMM;
use crate::trace_selection::set_of_points::SetOfPoints;

use super::analysis_tab::AnalysisTab;
use super::main_tab::*;
use super::raw_analysis_tab::RawAnalysis;

#[derive(PartialEq)]
enum TabType {
    Main,
    RawAnalysis,
    Analysis,
}

pub enum TabOutput {
    Main {hmm_input: HMMInput}
}

pub trait Tab {
    fn render(&mut self, ctx: &egui::Context, hmm: &mut HMM, preprocessing: &mut SetOfPoints, logs: &mut Vec<String>) -> Option<TabOutput>;
}

pub struct MyApp {
    // System stuff
    hmm: HMM,
    preprocessing: SetOfPoints,

    // GUI stuff
    current_tab: TabType,
    main_tab: MainTab,
    raw_analysis_tab: RawAnalysis,
    analysis_tab: AnalysisTab,

    logs: Vec<String>,

    hmm_analysis_ready: bool,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            // System stuff
            hmm: HMM::new(),
            preprocessing: SetOfPoints::new(),

            // GUI stuff
            current_tab: TabType::Main,
            main_tab: MainTab::default(),
            raw_analysis_tab: RawAnalysis::default(),
            analysis_tab: AnalysisTab::default(),

            logs: Vec::new(),

            hmm_analysis_ready: false,
        }
    }
}

impl MyApp {
    /// Apply global styles to ensure consistent text sizes and padding.
    fn apply_global_styles(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();

        // Set consistent global text sizes
        style.text_styles = [
            (egui::TextStyle::Heading, egui::FontId::proportional(24.0)), // Large headings
            (egui::TextStyle::Body, egui::FontId::proportional(18.0)),    // Body text
            (egui::TextStyle::Button, egui::FontId::proportional(20.0)),  // Button text
            // Monospace: Intended only for console-like output
            (egui::TextStyle::Monospace, egui::FontId::monospace(12.0)),  // Smaller text for console-like appearance
        ]
        .into();

        // Apply padding and spacing for buttons and labels
        style.spacing.button_padding = egui::vec2(10.0, 6.0); // Larger button padding
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);    // Spacing between elements

        // Apply changes
        ctx.set_style(style);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply global styles!
        self.apply_global_styles(ctx);

        // Check if points are loaded to conditionally show the "Raw Analysis" tab
        let raw_analysis_available = !self.preprocessing.get_points().is_empty();

        // Draw the tab bar at the top
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.current_tab, TabType::Main, "Main");
                if raw_analysis_available {
                    ui.selectable_value(&mut self.current_tab, TabType::RawAnalysis, "Raw Analysis");
                }
                if self.hmm_analysis_ready {
                    ui.selectable_value(&mut self.current_tab, TabType::Analysis, "Analysis");
                }
            });
        });

        // Show the content of the current tab
        match self.current_tab {
            TabType::Main => {
                let tab_output_option = self.main_tab.render(ctx, &mut self.hmm, &mut self.preprocessing, &mut self.logs);

                if let Some(TabOutput::Main { hmm_input }) = tab_output_option {
                    match self.hmm.run(hmm_input) {
                        Ok(_) => {
                            self.logs.push("HMM run was successful.".to_string());
                            self.hmm_analysis_ready = true;
                        }
                        Err(e) => {
                            self.logs.push(format!("Error running HMM: {:?}", e));
                        }
                    }
                }
            }
            TabType::RawAnalysis => {
                let _ = self.raw_analysis_tab.render(ctx, &mut self.hmm, &mut self.preprocessing, &mut self.logs);
            }
            TabType::Analysis => {
                let _ = self.analysis_tab.render(ctx, &mut self.hmm, &mut self.preprocessing, &mut self.logs);
            }
        }
    }
}