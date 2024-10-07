use std::hash::{Hash, Hasher};
use super::filters::*;

#[derive(Debug, Clone)]
pub struct IndividualTrace {
    values: Vec<f64>,
    trace_type: TraceType,
}
impl IndividualTrace {
    pub fn new(values: Vec<f64>, trace_type: TraceType) -> Result<Self, IndividualTraceError> {
        check_trace_validity(&values)?;
        
        Ok(Self { values, trace_type })
            
    }

    pub fn trace_type_wrapper(trace_type: &TraceType) -> Self {
        Self { values: Vec::new(), trace_type: trace_type.clone()}
    }

    pub fn check_validity(&self) -> Result<(), IndividualTraceError> {
        check_trace_validity(&self.values)

    }

    pub fn get_values(&self) -> &[f64] {
        &self.values
    }

    pub fn get_type(&self) -> &TraceType {
        &self.trace_type
    }

    pub fn get_len(&self) -> usize {
        self.values.len()
    }

    pub fn detect_photobleaching_event(&self, threshold: f64, window_size: usize) -> Option<usize> {
        
        // Apply median filter
        let filtered_values = median_filter(&self.values, window_size);
        // Compute gradient
        let gradient = compute_gradient(&filtered_values);
        // Find noise threshold
        let [_, std] = compute_mean_and_std(&gradient);
        let noise_threshold = threshold * std;
        // Find photobleaching event (significant negative drop)
        for (i, &grad) in gradient.iter().enumerate().rev() {
            if grad < -noise_threshold {
                return Some(i); // Return the index where the photobleaching event occurs
            }
        }
        None // No photobleaching event detected
    }

    pub fn snr(&self) -> Result<f64, IndividualTraceError> {

        if self.values.len() < 2 {return Err(IndividualTraceError::NotEnoughValues)}
        Ok(compute_snr(&self.values))
    }

    pub fn snr_before_after_bleaching(&self, pb_event: usize) -> Result<[f64; 2], IndividualTraceError>{

        if pb_event >= self.values.len() {
            return Err(IndividualTraceError::InvalidTimeStepINdex);
        }

        if pb_event > self.values.len() - 3 {
            return Err(IndividualTraceError::NotEnoughValues);
        }

        let before = compute_snr(&self.values[0..pb_event]);
        let after = compute_snr(&self.values[pb_event + 1 .. self.values.len()]);

        Ok([before, after])
    }

    pub fn noise_post_bleaching(&self, pb_event: usize) -> Result<f64, IndividualTraceError> {

        if pb_event >= self.values.len() - 1{
            return Err(IndividualTraceError::InvalidTimeStepINdex);
        }

        let [_, std] = compute_mean_and_std(&self.values[pb_event + 1 .. self.values.len()]);

        Ok(std)
    }

    pub fn first_value(&self) -> f64 {
        
        self.values[0]
    }

    pub fn max_min_values(&self) -> [f64; 2]{

        // find the max and min values
        let max = *self.values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min = *self.values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();


        [max, min]
    }

    pub fn lifetimes(&self, threshold: f64, min_len: usize) ->  Vec<[usize; 2]>{
        let mut lifetimes_vec: Vec<[usize; 2]> = Vec::new();

        let mut in_run=false;
        let mut curr_run_start: usize = 0;
        for (i, value) in self.values.iter().enumerate() {
            if *value > threshold {
                if !in_run {
                    in_run = true;
                    curr_run_start = i;
                }
            } else {
                if in_run {
                    if i - curr_run_start >= min_len {
                        lifetimes_vec.push([curr_run_start, i]);
                    }
                    in_run = false;
                }
            }
        }

        if in_run && (self.values.len() - curr_run_start >= min_len) {
            lifetimes_vec.push([curr_run_start, self.values.len()]);
        }

        lifetimes_vec
    }

}

// Implementing PartialEq and Eq to compare only based on trace_type
impl PartialEq for IndividualTrace {
    fn eq(&self, other: &Self) -> bool {
        self.trace_type == other.trace_type
    }
}

impl Eq for IndividualTrace {}

// Implementing Hash to use only trace_type for hashing
impl Hash for IndividualTrace {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.trace_type.hash(state);
    }
}


#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum TraceType {
    DemDexc,
    AemDexc,
    AemAexc,
    DemAexc,
    Stoichiometry,
    FRET,
    BackgroundDemDexc,
    BackgroundAemDexc,
    BackgroundAemAexc,
    RawDemDexc,
    RawAemDexc,
    RawAemAexc,
    UncorrectedS,
    UncorrectedE,
    IdealizedE,
    TotalPairIntensity,
}

#[derive(Debug, Clone)]
pub enum IndividualTraceError {
    EmptyTraceVector,
    IncludesNaNs,
    InvalidTimeStepINdex,
    NotEnoughValues,
}


fn check_trace_validity(trace: &[f64]) -> Result<(), IndividualTraceError> {
    if trace.len() == 0 {return Err(IndividualTraceError::EmptyTraceVector)}

    if trace.iter().any(|v| v.is_nan()) { return Err(IndividualTraceError::IncludesNaNs) }

    Ok(())
}