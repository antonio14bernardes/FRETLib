use std::collections::HashMap;
use super::point_traces::*;
use super::filter::*;
use super::individual_trace::*;
use super::trace_loader::*;

#[derive(Debug, Clone)]
pub struct SetOfPoints {
    points: HashMap<String, PointTraces>,
    rejected_points: Option<HashMap<String, (PointTraces, PointTracesError)>>,

    filter: FilterSetup,
    photobleaching_params: PhotobleachingFilterValues,
    fret_lifetimes_params: FretLifetimesFilterValues,
}

impl SetOfPoints {
    pub fn new() -> Self {
        Self {
            points: HashMap::new(),
            rejected_points: None,

            filter: FilterSetup::default(),
            photobleaching_params: PhotobleachingFilterValues::default(),
            fret_lifetimes_params: FretLifetimesFilterValues::default(),
        }
    }

    pub fn add_point_from_file(&mut self, file_path: &str) -> Result<(), SetOfPointsError> {
        // Get traces from file
        let point_traces = parse_file(file_path).map_err(|error| SetOfPointsError::TraceLoaderError { error })?;
        let metadata = point_traces.get_metadata().ok_or(SetOfPointsError::PointTraceMetadataNotFound)?;
        let file_path = metadata.file_path.clone();

        // Insert into the HashMap with file_path as key
        self.points.insert(file_path, point_traces);

        Ok(())
    }

    pub fn add_points_from_dir(&mut self, dir: &str) -> Result<(), SetOfPointsError> {
        let points = load_traces_from_directory(dir).map_err(|error| SetOfPointsError::TraceLoaderError { error })?;

        for point_traces in points {
            // Get traces from file
            let metadata = point_traces.get_metadata().ok_or(SetOfPointsError::PointTraceMetadataNotFound)?;
            let file_path = metadata.file_path.clone();
            self.points.insert(file_path, point_traces);
        }

        Ok(())
    }

    pub fn set_filter_setup(&mut self, filter: FilterSetup) {
        self.filter = filter;
    }

    pub fn clear_filter(&mut self) {
        self.filter = FilterSetup::empty();
    }

    pub fn get_points(&self) -> &HashMap<String, PointTraces> {
        &self.points
    }

    pub fn filter(&mut self) {
        // Initialize `rejected_points` to store rejected items
        self.rejected_points = Some(HashMap::new());
        let rejected_points = self.rejected_points.as_mut().unwrap();

        // Collect keys of points to be removed
        let mut keys_to_remove = Vec::new();

        // Go through each point in `points`
        for (file_path, point) in self.points.iter_mut() {
            let output = point.prepare_filter_values(&self.photobleaching_params, &self.fret_lifetimes_params);
            if let Ok(values_to_filter) = output {
                let (valid, tests_failed) = self.filter.check_valid(&values_to_filter);

                if !valid {
                    let failed_tests_err = PointTracesError::FailedFilterTests { tests_failed };
                    
                    // Move to `rejected_points` and mark for removal
                    rejected_points.insert(file_path.clone(), (point.clone(), failed_tests_err));
                    keys_to_remove.push(file_path.clone());
                }
            } else {
                // Add to rejected points on error and mark for removal
                rejected_points.insert(file_path.clone(), (point.clone(), output.err().unwrap()));
                keys_to_remove.push(file_path.clone());
            }
        }

        // Remove rejected points from `points`
        for key in keys_to_remove {
            self.points.remove(&key);
        }
    }
}

#[derive(Debug, Clone)]
pub enum SetOfPointsError {
    PointTracesError { error: PointTracesError },
    TraceLoaderError { error: TraceLoaderError },
    PointTraceMetadataNotFound,
}