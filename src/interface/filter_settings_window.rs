use eframe::egui;
use crate::trace_selection::filter::{self, Comparison, FilterSetup};
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;
use std::collections::HashMap;


pub struct FilterSettingsWindow {
    pub is_open: bool,
    pub filter_setup: FilterSetup,
    pub input_buffers: HashMap<String, String>, // Temporary buffers for text inputs
}

impl FilterSettingsWindow {
    pub fn new() -> Self {
        Self {
            is_open: false,
            filter_setup: FilterSetup::default(),
            input_buffers: HashMap::new(),
        }
    }

    pub fn open(&mut self) {
        self.is_open = true;
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if self.is_open {
            egui::Window::new("Configure Filter Settings")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                // .resizable(true) // Allow resizing
                .fixed_size([500.0, 450.0]) // Enforce strict size
                .show(ctx, |ui| {
                    ui.vertical(|ui| {
    
                        // Wrap all fields in a scrollable area
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false]) // Prevent auto-shrinking
                            .show(ui, |ui| {
                                let space_between_fields = 10.0_f32;

                                let fields: Vec<(&str, Option<&mut Option<Comparison<usize>>>, Option<&mut Option<Comparison<f64>>>)> = vec![
                                    ("Photobleaching Steps", Some(&mut self.filter_setup.photobleaching_steps), None),
                                    ("Donor Lifetime", Some(&mut self.filter_setup.donor_lifetime), None),
                                    ("FRET Lifetimes", Some(&mut self.filter_setup.fret_lifetimes), None),
                                    ("SNR Background", None, Some(&mut self.filter_setup.snr_background)),
                                    ("Correlation Coefficient", None, Some(&mut self.filter_setup.correlation_coefficient)),
                                    ("Background Noise", None, Some(&mut self.filter_setup.background_noise)),
                                    ("Mean Total Intensity", Some(&mut self.filter_setup.mean_total_intensity), None),
                                    ("SNR Signal", None, Some(&mut self.filter_setup.snr_signal)),
                                    ("FRET First Time Step", None, Some(&mut self.filter_setup.fret_first_time_step)),
                                    ("Highest FRET", None, Some(&mut self.filter_setup.highest_fret)),
                                    ("Average FRET", None, Some(&mut self.filter_setup.average_fret)),
                                ];

                                // Process each field in order
                                for (label, usize_field, f64_field) in fields {
                                    if let Some(field) = usize_field {
                                        FilterSettingsWindow::render_numeric_filter_field(ui, ctx, label, field, &mut self.input_buffers);
                                    } else if let Some(field) = f64_field {
                                        FilterSettingsWindow::render_numeric_filter_field(ui, ctx, label, field, &mut self.input_buffers);
                                    }
                                    ui.add_space(space_between_fields);
                                }
                            });
    
                        ui.add_space(10.0);
    
                        // Reset and Close buttons
                        ui.horizontal(|ui| {
                            if ui.button("Set").clicked() {
                                println!("Set button clicked (no functionality yet).");
                                // Add functionality here: Store filter setup in the actual SetOfPointTraces
                            }
                            if ui.button("Reset").clicked() {
                                self.reset_to_defaults();
                                println!("Filter Settings Reset to Defaults");
                            }
    
                            if ui.button("Close").clicked() {
                                self.is_open = false;
                                println!("Filter Settings Window closed");
                            }
                        });
                    });
                });
        }

        // println!("Current filter: {:?}", self.filter_setup);
    }

    
    fn reset_to_defaults(&mut self) {
            let default_filter = FilterSetup::default();
    
            self.filter_setup = default_filter.clone();
    
            // Update input buffers to reflect default values
            self.input_buffers.clear();
    
            // Populate input buffers with default values
            if let Some(default) = default_filter.photobleaching_steps {
                self.input_buffers.insert(
                    "Photobleaching Steps".to_string(),
                    default.value_as_string(),
                );
            }
    
            if let Some(default) = default_filter.donor_lifetime {
                self.input_buffers.insert("Donor Lifetime".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.fret_lifetimes {
                self.input_buffers
                    .insert("FRET Lifetimes".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.snr_background {
                self.input_buffers
                    .insert("SNR Background".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.correlation_coefficient {
                self.input_buffers.insert(
                    "Correlation Coefficient".to_string(),
                    default.value_as_string(),
                );
            }
    
            if let Some(default) = default_filter.background_noise {
                self.input_buffers
                    .insert("Background Noise".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.mean_total_intensity {
                self.input_buffers.insert(
                    "Mean Total Intensity".to_string(),
                    default.value_as_string(),
                );
            }
    
            if let Some(default) = default_filter.snr_signal {
                self.input_buffers
                    .insert("SNR Signal".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.fret_first_time_step {
                self.input_buffers
                    .insert("FRET First Time Step".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.highest_fret {
                self.input_buffers
                    .insert("Highest FRET".to_string(), default.value_as_string());
            }
    
            if let Some(default) = default_filter.average_fret {
                self.input_buffers
                    .insert("Average FRET".to_string(), default.value_as_string());
            }
        }
    
   /// Static method for rendering numeric filter fields
   fn render_numeric_filter_field<T>(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    label: &str,
    filter_option: &mut Option<Comparison<T>>,
    input_buffers: &mut HashMap<String, String>,
    ) where
        T: Default + Clone + Display + FromStr,
        <T as FromStr>::Err: fmt::Debug,
    {
        let buffer_key = label.to_string();
        let input_buffer = input_buffers.entry(buffer_key.clone()).or_insert_with(|| {
            filter_option
                .as_ref()
                .map_or_else(|| T::default().to_string(), |comparison| comparison.value_as_string())
        });

        let is_active = filter_option.is_some();

        ui.horizontal(|ui| {
            let mut active = is_active;

            // Checkbox and label
            if ui.checkbox(&mut active, label).changed() {
                if active {
                    // Set to default value
                    *filter_option = Some(Comparison::Equal {
                        value: T::default(),
                    });
                    *input_buffer = T::default().to_string();
                } else {
                    *filter_option = None;
                    *input_buffer = String::new();
                }
            }

            if active {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if let Some(ref mut comparison) = filter_option {
                        // Render the text input to the right of the combo box
                        // Render the text input and detect focus loss
                        let response = ui.add_sized([100.0, 20.0], |ui: &mut egui::Ui| {
                            ui.text_edit_singleline(input_buffer)
                        });

                        // ComboBox for selecting comparison type, with increased width
                        let comparison_types = get_comparison_types::<T>();
                        let current_comparison = comparison.to_string();
                        let mut selected = current_comparison.clone();

                        egui::ComboBox::from_id_source(format!("combo_{}", label))
                        .selected_text(current_comparison)
                        .width(150.0) // Set the width of the combo box
                        .show_ui(ui, |ui| {
                            for comp in &comparison_types {
                                if ui.selectable_value(&mut selected, comp.clone(), comp.clone()).clicked() {
                                    // Update comparison type when ComboBox selection changes
                                    if let Ok(new_comparison) = Comparison::from_str(&selected) {
                                        *comparison = new_comparison.with_value(
                                            input_buffer
                                                .parse::<T>()
                                                .unwrap_or_else(|_| T::default()),
                                        );
                                    }
                                }
                            }
                        });

                        // Update comparison value if input is valid on Enter key or focus loss
                        if !input_buffer.is_empty()
                            && (response.lost_focus() || ctx.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            match input_buffer.parse::<T>() {
                                Ok(parsed_value) => {
                                    println!("Selected: {}", &selected);
                                    *comparison = Comparison::from_str(&selected).unwrap().with_value(parsed_value);
                                }
                                Err(_) => {
                                    println!(
                                        "Invalid input for {}: {}. Keeping old value.",
                                        label, input_buffer
                                    );
                                    // Revert to the previous value if parsing fails
                                    *input_buffer = comparison.value_as_string();
                                }
                            }
                        }

                        
                    }
                });
            }
        });

        ui.add_space(5.0);
    }
}

/// Helper function to get com

/// Helper function to get comparison types based on the generic type T.
fn get_comparison_types<T>() -> Vec<String>
where
    T: Display,
{
    match std::any::type_name::<T>() {
        "usize" | "f64" => vec![
            "Larger".to_string(),
            "LargerEq".to_string(),
            "Smaller".to_string(),
            "SmallerEq".to_string(),
            "Equal".to_string(),
        ],
        _ => vec![],
    }
}



impl<T> Comparison<T>
where
    T: Default + Clone + Display + FromStr,
    <T as FromStr>::Err: fmt::Debug,
{
    /// Returns the value as a string for display in the UI.
    fn value_as_string(&self) -> String {
        match self {
            Comparison::Larger { value }
            | Comparison::LargerEq { value }
            | Comparison::Smaller { value }
            | Comparison::SmallerEq { value }
            | Comparison::Equal { value } => value.to_string(),
            Comparison::WithinNStd { n } => n.to_string(),
        }
    }

    /// Creates a new Comparison with the given value, preserving the comparison type.
    fn with_value(&self, new_value: T) -> Self {
        match self {
            Comparison::Larger { .. } => Comparison::Larger { value: new_value },
            Comparison::LargerEq { .. } => Comparison::LargerEq { value: new_value },
            Comparison::Smaller { .. } => Comparison::Smaller { value: new_value },
            Comparison::SmallerEq { .. } => Comparison::SmallerEq { value: new_value },
            Comparison::Equal { .. } => Comparison::Equal { value: new_value },
            Comparison::WithinNStd { n } => Comparison::WithinNStd { n: *n },
        }
    }

    ///Attempts to parse a Comparison from a string.
    fn from_str(s: &str) -> Result<Self, ()>
    where
        T: Default + Clone,
    {
        match s {
            "Larger" => Ok(Comparison::Larger {
                value: T::default(),
            }),
            "LargerEq" => Ok(Comparison::LargerEq {
                value: T::default(),
            }),
            "Smaller" => Ok(Comparison::Smaller {
                value: T::default(),
            }),
            "SmallerEq" => Ok(Comparison::SmallerEq {
                value: T::default(),
            }),
            "Equal" => Ok(Comparison::Equal {
                value: T::default(),
            }),
            "WithinNStd" => Ok(Comparison::WithinNStd {
                n: 0, // Avoid type problems
            }),
            _ => Err(()),
        }
    }
}

impl<T> Display for Comparison<T>
where
    T: Display + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Comparison::Larger { .. } => write!(f, "Larger"),
            Comparison::LargerEq { .. } => write!(f, "LargerEq"),
            Comparison::Smaller { .. } => write!(f, "Smaller"),
            Comparison::SmallerEq { .. } => write!(f, "SmallerEq"),
            Comparison::Equal { .. } => write!(f, "Equal"),
            Comparison::WithinNStd { .. } => write!(f, "WithinNStd"),
        }
    }
}

impl<T> FromStr for Comparison<T>
where
    T: Default + Clone,
{
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Larger" => Ok(Comparison::Larger {
                value: T::default(),
            }),
            "LargerEq" => Ok(Comparison::LargerEq {
                value: T::default(),
            }),
            "Smaller" => Ok(Comparison::Smaller {
                value: T::default(),
            }),
            "SmallerEq" => Ok(Comparison::SmallerEq {
                value: T::default(),
            }),
            "Equal" => Ok(Comparison::Equal {
                value: T::default(),
            }),
            _ => Err(()),
        }
    }
}

