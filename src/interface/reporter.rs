use rfd::FileDialog;

use crate::signal_analysis::hmm::analysis::hmm_analyzer::HMMAnalyzer;
use crate::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use crate::signal_analysis::hmm::state::State;
use crate::trace_selection::set_of_points::SetOfPoints;

use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::{Path, PathBuf};

/// Reporter struct for managing report generation
pub struct Reporter {
    pub path: PathBuf, // Path to store the output files
}

impl Reporter {
    /// Open a dialog to select a folder and create a new `Reporter`
    pub fn new(logs: &mut Vec<String>) -> Option<Self> {
        // Open save dialog to select folder and input file name
        if let Some(file_path) = FileDialog::new()
            .set_file_name("hmm_setup_report.txt") // Default file name
            .set_title("Save HMM Report")         // Dialog title
            .save_file()                         // Save file dialog
        {
            logs.push(format!("Reporter file initialized in: {:?}", file_path));
            Some(Self { path: file_path })
        } else {
            logs.push("No file was selected.".to_string());
            None
        }
    }

    /// Generate a report file for HMM setup
    pub fn report(
        &self,
        hmm_analyzer: &HMMAnalyzer,
        preprocessing: &SetOfPoints,
        logs: &mut Vec<String>,
    ) -> std::io::Result<()> {
        // Ensure the directory exists
        create_dir_all(&self.path)?;
    
        // Open the main report file
        let file_path = Path::new(&self.path).join("hmm_setup_and_sequences_report.txt");
        let file = File::create(&file_path)?;
        let mut writer = BufWriter::new(file);
    
        // Write HMM Setup Header
        writeln!(writer, "HMM Setup Report")?;
        writeln!(writer, "=================")?;
        writeln!(writer)?;
    
        // Report states
        if let Some(states) = hmm_analyzer.get_states() {
            writeln!(writer, "[States]")?;
            writeln!(writer, "ID,Mean,Std")?;
            for state in states {
                writeln!(writer, "{},{:.3},{:.3}", state.id, state.value, state.noise_std)?;
            }
            writeln!(writer)?;
        }
    
        // Report start matrix
        if let Some(start_matrix) = hmm_analyzer.get_start_matrix() {
            writeln!(writer, "[Start Matrix]")?;
            writeln!(writer, "State ID,Start Probability")?;
            for (i, &value) in start_matrix.matrix.iter().enumerate() {
                writeln!(writer, "{}, {:.3}", i, value)?;
            }
            writeln!(writer)?;
        }
    
        // Report transition matrix
        if let Some(transition_matrix) = hmm_analyzer.get_transition_matrix() {
            writeln!(writer, "[Transition Matrix]")?;
            writeln!(
                writer,
                "From State \\ To State,{}",
                (0..transition_matrix.matrix.raw_matrix.len())
                    .map(|i| format!("State {}", i))
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
    
            for (from, row) in transition_matrix.matrix.raw_matrix.iter().enumerate() {
                let row_string = row
                    .iter()
                    .map(|&value| format!("{:.3}", value))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(writer, "State {},{}", from, row_string)?;
            }
            writeln!(writer)?;
        }
    
        // Retrieve sequences and state ID sequences from HMMAnalyzer
        let sequences = hmm_analyzer.get_sequences();
        let state_id_sequences = hmm_analyzer.get_state_sequences();
        let states = hmm_analyzer.get_states();
    
        // Report per-sequence data
        if let (Some(sequences), Some(state_id_sequences), Some(states)) =
            (sequences, state_id_sequences, states)
        {
            writeln!(writer, "\n[Per-Sequence Data]")?;
            writeln!(writer, "====================")?;
            writeln!(writer)?;
    
            for (key, sequence) in preprocessing.get_points().keys().zip(sequences) {
                // Matching state IDs
                let idx = sequences.iter().position(|seq| seq == sequence).unwrap_or(0);
                let state_ids = &state_id_sequences[idx];
    
                writeln!(writer, "Trace: {}", key)?;
                writeln!(writer, "Real,Idealized,State ID")?;
    
                for (i, (&real_value, &state_id)) in sequence.iter().zip(state_ids).enumerate() {
                    let idealized_value = states
                        .get(state_id)
                        .map(|s| s.value)
                        .unwrap_or(0.0);
                    writeln!(writer, "{:.3},{:.3},{}", real_value, idealized_value, state_id)?;
                }
    
                writeln!(writer)?; // Add spacing between traces
            }
        } else {
            writeln!(writer, "\n[Per-Sequence Data]\nNot all data available to generate sequence reports.")?;
        }
    
        writeln!(writer, "\nReport generation completed successfully.")?;

        logs.push(format!("HMM setup and sequences report stored at {:?}", file_path));
    
        Ok(())
    }

}