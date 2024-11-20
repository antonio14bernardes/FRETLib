use std::collections::HashMap;
use rfd::FileDialog;
use eframe::egui;

use crate::trace_selection::{set_of_points::{SetOfPoints, SetOfPointsError}, trace_loader::TraceLoaderError};

pub struct LoadTracesWindow {
    pub is_open: bool,
    pub loaded_file_paths: Vec<String>, // To store loaded file paths
    pub loaded_dir_paths: Vec<String>, // To store loaded directory paths. Does not necessarily mean all files in the dir were successful!
    failed_files: Vec<(String, TraceLoaderError)>, // Store failed files and their errors
}

impl LoadTracesWindow {
    pub fn new() -> Self {
        Self {
            is_open: false,
            loaded_file_paths: Vec::new(),
            loaded_dir_paths: Vec::new(),
            failed_files: Vec::new(),
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
    }

    pub fn show(&mut self, ctx: &egui::Context, preprocessing: &mut SetOfPoints) {
        if self.is_open {
            egui::Window::new("Load Traces")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                // .resizable(false)
                .fixed_size([500.0, 400.0])
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
                        ui.label("Load traces by selecting files or directories:");

                        ui.add_space(10.0);

                        // Button to load files
                        if ui.button("Load Files").clicked() {
                            if let Some(files) = FileDialog::new()
                                .add_filter("Trace Files", &["txt"])
                                .pick_files()
                            {
                                let mut new_files = Vec::new();
                                new_files.extend(
                                    files.into_iter().map(|p| p.to_string_lossy().to_string()),
                                );

                                for file in new_files {
                                    let load_attempt = preprocessing.add_point_from_file(&file);

                                    match load_attempt {
                                        Err(SetOfPointsError::TraceLoaderError { error }) => {
                                            self.failed_files.push((file, error));
                                        }
                                        Ok(_) => self.loaded_file_paths.push(file),
                                        _ => {}
                                    }
                                }
                            }
                        }

                        ui.add_space(10.0);

                        // Button to load directories
                        if ui.button("Load Directory").clicked() {
                            if let Some(directory) = FileDialog::new().pick_folder() {
                                let new_dir = directory.to_string_lossy().to_string();
                                let load_attempt = preprocessing.add_points_from_dir(&new_dir);

                                match load_attempt {
                                    Err(SetOfPointsError::FailedToLoadFiles { fails }) => {
                                        self.failed_files.extend(fails);
                                    }
                                    Ok(_) => self.loaded_dir_paths.push(new_dir),
                                    _ => {}
                                }
                            }
                        }

                        ui.add_space(20.0);

                        // Scrollable console-like output
                        ui.label("Status:");
                        ui.separator();
                        ui.add_space(5.0);

                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2]) // Prevent the scroll area from shrinking unnecessarily
                            .show(ui, |ui| {
                                // Save the original spacing (global/default settings)
                                let original_spacing = ui.spacing().clone();

                                // Temporarily reduce spacing for Monospace text
                                ui.spacing_mut().item_spacing.y = 2.0; // Smaller gap between lines for console-like text

                                // Render the output
                                if self.failed_files.is_empty() {
                                    if !self.loaded_file_paths.is_empty() || !self.loaded_dir_paths.is_empty() {
                                        ui.colored_label(egui::Color32::GREEN, "All good!");
                                    } else {
                                        ui.monospace("No files or directories loaded.");
                                    }
                                } else {
                                    ui.colored_label(egui::Color32::RED, "Loading errors:");
                                    for (file, error) in &self.failed_files {
                                        ui.monospace(format!("{} -> {:?}", file, error));
                                    }
                                }

                                // Restore the original spacing after Monospace text
                                *ui.spacing_mut() = original_spacing;
                            });

                        ui.add_space(20.0);

                        // Close button
                        if ui.button("Close").clicked() {
                            self.is_open = false;
                        }
                    });
                });
        }

        
    }
}