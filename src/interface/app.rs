use eframe::egui;
use crate::signal_analysis::hmm::hmm_struct::HMMInput;
use crate::signal_analysis::hmm::hmm_struct::HMM;
use crate::trace_selection::set_of_points::SetOfPoints;

use super::main_tab::*;
use super::other_tab::*;

#[derive(PartialEq)]
enum TabType {
    Main,
    Other,
}

pub enum TabOutput {
    Main {hmm_input: HMMInput}
}

pub trait Tab {
    fn render(&mut self, ctx: &egui::Context, hmm: &mut HMM, preprocessing: &mut SetOfPoints) -> Option<TabOutput>;
}

pub struct MyApp {
    // System stuff
    hmm: HMM,
    preprocessing: SetOfPoints,

    // GUI stuff
    current_tab: TabType,
    main_tab: MainTab,
    other_tab: OtherTab,
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
            other_tab: OtherTab::default(),
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
        // Apply global styles here
        self.apply_global_styles(ctx);

        // Draw the tab bar at the top
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.current_tab, TabType::Main, "Main");
                ui.selectable_value(&mut self.current_tab, TabType::Other, "Other");
            });
        });

        // Show the content of the current tab
        match self.current_tab {
            TabType::Main => {
                let tab_output_option = self.main_tab.render(ctx, &mut self.hmm, &mut self.preprocessing);

                if let Some(TabOutput::Main { hmm_input }) = tab_output_option {
                    self.hmm.run(hmm_input).unwrap();
                }
            }
            TabType::Other => {
                let _ = self.other_tab.render(ctx, &mut self.hmm, &mut self.preprocessing);
            }
        }
    }
}