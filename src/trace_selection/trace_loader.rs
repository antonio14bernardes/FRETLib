use super::individual_trace::*;
use super::point_traces::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

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
    let file = File::open(&path).map_err(|error| TraceLoaderError::IOError { error })?;
    let reader = io::BufReader::new(file);

    let mut lines = reader.lines().enumerate();

    // Read the metadata lines
    let mut file_date = String::new();
    let mut movie_filename = String::new();
    let mut fret_pair = 0;    

    // Parse the traces
    let mut traces: Vec<Vec<f64>> = Vec::new();
    let mut headers: Vec<TraceType> = Vec::new();

    for (index, line) in lines {
        let line = line.map_err(|error| TraceLoaderError::IOError { error })?;
        let tokens: Vec<&str> = line.trim().split_whitespace().collect();
        if index == 0 {
            continue;
        } else if index == 1{
            // Parse the file date
            file_date = line.trim().replace("Date: ", "");
        } else if index == 2{
            movie_filename = line.trim().replace("Movie filename: ", "");
        } else if index == 3 {
            // Parse the fret pair number
            let pair_info = line.trim().replace("FRET pair #", "");
            fret_pair = match pair_info.parse::<usize>() {
                Ok(value) => value,
                Err(_) => 0,
            }
        } else if index == 4 {
            
            // Parse headers
            headers = tokens.iter()
                .map(|&header| parse_trace_type(header))
                .collect::<Result<Vec<_>, _>>()?;
            traces.resize(headers.len(), Vec::new());
        } else {
            // Parse data rows
            let values: Vec<f64> = tokens.iter().filter_map(|&token| token.parse::<f64>().ok()).collect();
            if values.len() == headers.len() {
                for (i, &value) in values.iter().enumerate() {
                    traces[i].push(value);
                }
            } else {
                return Err(
                    TraceLoaderError::LineHasLessValues {
                        line_num: index,
                        num_line_values: values.len(),
                        num_header_values: headers.len(),
                    }
                );
            }
        }
    }

    // Get metadata into a struct
    let metadata = PointFileMetadata::new(file_date, movie_filename, fret_pair);

    // Create IndividualTrace objects
    let mut point_traces = PointTraces::new_empty();
    

    for (trace, header) in traces.into_iter().zip(headers.into_iter()) {
        point_traces.insert_new_trace(trace, header)
        .map_err(|e| TraceLoaderError::PointTracesError { error: e })?;
    }

    point_traces.insert_metadata(metadata);

    Ok(point_traces)
}

#[derive(Debug)]
pub enum TraceLoaderError {
    IOError {error: io::Error},
    LineHasLessValues {line_num: usize, num_line_values: usize, num_header_values: usize},
    InvalidHeader {header: String},
    PointTracesError {error: PointTracesError},
}