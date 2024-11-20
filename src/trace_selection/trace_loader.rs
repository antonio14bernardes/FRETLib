use super::individual_trace::*;
use super::point_traces::*;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::str::FromStr;

// Function to map column header to TraceType enum
fn parse_trace_type(header: &str) -> Result<TraceType, TraceLoaderError> {
    match header {
        "Dem-Dexc" => Ok(TraceType::DemDexc),
        "Aem-Dexc" => Ok(TraceType::AemDexc),
        "Dem-Aexc" => Ok(TraceType::DemAexc),
        "Aem-Aexc" => Ok(TraceType::AemAexc),
        "S" => Ok(TraceType::Stoichiometry),
        "E" => Ok(TraceType::FRET),
        "D-Dexc-bg." => Ok(TraceType::BackgroundDemDexc),
        "A-Dexc-bg." => Ok(TraceType::BackgroundAemDexc),
        "A-Aexc-bg." => Ok(TraceType::BackgroundAemAexc),
        "D-Dexc-rw." => Ok(TraceType::RawDemDexc),
        "A-Dexc-rw." => Ok(TraceType::RawAemDexc),
        "A-Aexc-rw." => Ok(TraceType::RawAemAexc),
        "S-uncorr." => Ok(TraceType::UncorrectedS),
        "E-uncorr." => Ok(TraceType::UncorrectedE),
        _ => Err(TraceLoaderError::InvalidHeader { header: header.to_string() }),
    }
}


// Function to parse the file and return a vector of IndividualTrace
pub fn parse_file(file_path: &str) -> Result<PointTraces, TraceLoaderError> {
    let path = Path::new(file_path);
    let file = File::open(&path).map_err(|_| TraceLoaderError::FailedToLoadSingleFile { file: file_path.to_string() })?;
    let reader = io::BufReader::new(file);

    let lines = reader.lines().enumerate();

    // Store the file path
    let file_path_string = file_path.to_string();

    // Read the metadata lines
    let mut file_date = String::new();
    let mut movie_filename = String::new();
    let mut fret_pair = 0;    

    // Parse the traces
    let mut traces: Vec<Vec<f64>> = Vec::new();
    let mut headers: Vec<TraceType> = Vec::new();
    let mut line_count = 0; // Initialize line counter

    for (index, line) in lines {
        let line = line.map_err(|_| TraceLoaderError::FailedToAnalyzeFile)?;
        line_count += 1; // Increment line count
        let tokens: Vec<&str> = line.trim().split_whitespace().collect();

        if index == 0 {
            // Skip the first line (header line or unused)
            continue;
        } else if index == 1 {
            // Check and parse the file date
            if !line.starts_with("Date: ") {
                return Err(TraceLoaderError::InvalidLine {
                    reason: "Expected line to start with 'Date: '.".to_string(),
                });
            }
            file_date = line.trim().replace("Date: ", "");
        } else if index == 2 {
            // Check and parse the movie filename
            if !line.starts_with("Movie filename: ") {
                return Err(TraceLoaderError::InvalidLine {
                    reason: "Expected line to start with 'Movie filename: '.".to_string(),
                });
            }
            movie_filename = line.trim().replace("Movie filename: ", "");
        } else if index == 3 {
            // Check and parse the fret pair number
            if !line.starts_with("FRET pair #") {
                return Err(TraceLoaderError::InvalidLine {
                    reason: "Expected line to start with 'FRET pair #'.".to_string(),
                });
            }
            let pair_info = line.trim().replace("FRET pair #", "");
            fret_pair = pair_info.parse::<usize>().map_err(|_| TraceLoaderError::InvalidLine {
                reason: "Invalid FRET pair number.".to_string(),
            })?;
        } else if index == 4 {
            // Check and parse headers
            if tokens.is_empty() {
                return Err(TraceLoaderError::InvalidLine {
                    reason: "Header line is empty or invalid.".to_string(),
                });
            }
            headers = tokens.iter()
                .map(|&header| parse_trace_type(header))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| TraceLoaderError::InvalidLine {
                    reason: "Header contains invalid trace type.".to_string(),
                })?;
            traces.resize(headers.len(), Vec::new());
        } else {
            // Parse data rows
            let values: Vec<f64> = tokens.iter().filter_map(|&token| token.parse::<f64>().ok()).collect();
            if values.len() == headers.len() {
                for (i, &value) in values.iter().enumerate() {
                    traces[i].push(value);
                }
            } else {
                return Err(TraceLoaderError::LineHasLessValues {
                    line_num: index,
                    num_line_values: values.len(),
                    num_header_values: headers.len(),
                });
            }
        }
    }

    // Check if the file ends prematurely
    if line_count < 6 {
        return Err(TraceLoaderError::InvalidLine {
            reason: "File is too short bruv.".to_string(),
        });
    }

    // Get metadata into a struct
    let metadata = PointFileMetadata::new(file_path_string, file_date, movie_filename, fret_pair);

    // Create IndividualTrace objects
    let mut point_traces = PointTraces::new_empty();

    for (trace, header) in traces.into_iter().zip(headers.into_iter()) {
        point_traces.insert_new_trace(trace, header)
        .map_err(|e: PointTracesError| TraceLoaderError::PointTracesError { error: e })?;
    }

    point_traces.insert_metadata(metadata);

    Ok(point_traces)
}

pub fn load_traces_from_directory(dir: &str) -> Result<Vec<PointTraces>, TraceLoaderError> {
    let mut successful_traces = Vec::new();
    let mut failed_files: Vec<(String, TraceLoaderError)> = Vec::new();

    // Read the directory entries
    let entries = fs::read_dir(dir).map_err(|_| TraceLoaderError::InvalidDirectoryName)?;

    for entry in entries {
        let entry = entry.map_err(|_| TraceLoaderError::InvalidFileName)?; // Should not have this error ever since the input is a directory right?
        let path = entry.path();

        // Only process files (not directories)
        if path.is_file() {
            let file_path = path.to_str().unwrap_or_default().to_string();
            println!("Loading file: {}", &file_path);

            match parse_file(&file_path) {
                Ok(traces) => successful_traces.push(traces),
                Err(err) => failed_files.push((file_path, err)),
            }
        }
    }

    if !failed_files.is_empty() {
        Err(TraceLoaderError::FailedToLoadFiles {successful_traces, failed_files })
    } else {
        Ok(successful_traces)
    }
}

#[derive(Debug, Clone)]
pub enum TraceLoaderError {
    LineHasLessValues {line_num: usize, num_line_values: usize, num_header_values: usize},
    InvalidHeader {header: String},
    PointTracesError {error: PointTracesError},
    FailedToLoadSingleFile {file: String},
    FailedToLoadFiles { successful_traces: Vec<PointTraces>, failed_files: Vec<(String, TraceLoaderError)> },
    FailedToAnalyzeFile,
    InvalidFileName,
    InvalidDirectoryName,
    InvalidLine {reason: String},
}