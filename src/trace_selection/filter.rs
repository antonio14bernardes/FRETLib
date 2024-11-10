use super::{individual_trace::PhotobleachingFilterValues, point_traces::{PointTraces, PointTracesError}};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct ValuesToFilter {
    pub donor_photobleaching: Vec<usize>, // (Last event, number of events)
    pub fret_lifetimes: Vec<[usize; 2]>,
    // donor_blinking_events: Vec<usize>,
    pub snr_signal: f64, // SNR of total intensity before donor bleaching
    pub snr_backgroung: f64, // Mean of intensity before donor bleaching divided by s.d. of intensity after donor bleaching
    pub correlation_coef: f64, // Correlation between donor and acceptor
    pub background_std_post_bleach: f64,
    pub intensity_min_man_mean_std: MinMaxMeanStd,
    pub first_fret: f64,
    pub max_fret: f64,
    pub average_fret: f64,
}

#[derive(Debug, Clone)]
pub struct MinMaxMeanStd {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

#[derive(Debug, Clone)]
pub enum FilterTest {
    PhotobleachingSteps,
    DonorLifetime,
    FretLifetime,
    SNRSignal,
    CorrelationCoefficient,
    BackgroundNoise,
    MeanTotalIntensity,
    FRETFirstTimeStep,
    HighestFRET,
    AverageFRET,
}


#[derive(Debug, Clone)]
pub struct FilterSetup {
    photobleaching_steps: Option<Comparison<usize>>,
    donor_lifetime: Option<Comparison<usize>>,
    fret_lifetimes: Option<Comparison<usize>>,
    snr_background: Option<Comparison<f64>>,
    // donor_blinking_events: Option<Comparison<usize>>,
    correlation_coefficient: Option<Comparison<f64>>,
    background_noise: Option<Comparison<f64>>,

    mean_total_intensity: Option<Comparison<MinMaxMeanStd>>,
    snr_signal: Option<Comparison<f64>>,
    fret_first_time_step: Option<Comparison<f64>>,
    highest_fret: Option<Comparison<f64>>,
    average_fret: Option<Comparison<f64>>,
}

impl FilterSetup {
    pub fn default() -> Self {
        Self {
            photobleaching_steps: Some(Comparison::Equal { value: 1 }),
            donor_lifetime: Some(Comparison::Larger { value: 50 }),
            fret_lifetimes: Some(Comparison::Larger { value: 15 }),
            snr_background: Some(Comparison::Larger { value: 8.0 }),
            correlation_coefficient: Some(Comparison::Smaller { value: 0.5 }),
            background_noise: Some(Comparison::Smaller { value: 70.0 }),
            mean_total_intensity: Some(Comparison::WithinNStd {n: 2}),

            snr_signal: None,
            fret_first_time_step: None,
            highest_fret: None,
            average_fret: None,
        }
    }
    
    pub fn check_valid(&self, values_to_filter: ValuesToFilter) -> (bool, Vec<FilterTest>) {
        let mut failed_tests = Vec::new();

        // Photobleaching steps test
        if !run_test(self.photobleaching_steps.as_ref(), values_to_filter.donor_photobleaching.len()) {
            failed_tests.push(FilterTest::PhotobleachingSteps);
        }

        // Donor lifetime test
        let photobleaching_events = &values_to_filter.donor_photobleaching;
        if !photobleaching_events.is_empty() {
            let last_photobleach = photobleaching_events[photobleaching_events.len() - 1];
            if !run_test(self.donor_lifetime.as_ref(), last_photobleach) {
                failed_tests.push(FilterTest::DonorLifetime);
            }
        }

        // FRET lifetime test
        let lifetimes = &values_to_filter.fret_lifetimes;
        if lifetimes.is_empty() && self.fret_lifetimes.is_some() {
            failed_tests.push(FilterTest::FretLifetime);
        } else {
            let largest_lifetime = lifetimes.iter().map(|lifetime| lifetime[1] - lifetime[0]).max().unwrap();
            if !run_test(self.fret_lifetimes.as_ref(), largest_lifetime) {
                failed_tests.push(FilterTest::FretLifetime);
            }
        }

        // SNR signal test
        if !run_test(self.snr_signal.as_ref(), values_to_filter.snr_signal) {
            failed_tests.push(FilterTest::SNRSignal);
        }

        // Correlation coefficient test
        if !run_test(self.correlation_coefficient.as_ref(), values_to_filter.correlation_coef) {
            failed_tests.push(FilterTest::CorrelationCoefficient);
        }

        // Background noise test
        if !run_test(self.background_noise.as_ref(), values_to_filter.background_std_post_bleach) {
            failed_tests.push(FilterTest::BackgroundNoise);
        }

        // Mean total intensity test
        if let Some(test) = &self.mean_total_intensity {
            if !test.compare_min_max_mean_std(&values_to_filter.intensity_min_man_mean_std) {
                failed_tests.push(FilterTest::MeanTotalIntensity);
            }
        }

        // FRET first time step test
        if !run_test(self.fret_first_time_step.as_ref(), values_to_filter.first_fret) {
            failed_tests.push(FilterTest::FRETFirstTimeStep);
        }

        // Highest FRET test
        if !run_test(self.highest_fret.as_ref(), values_to_filter.max_fret) {
            failed_tests.push(FilterTest::HighestFRET);
        }

        // Average FRET test
        if !run_test(self.average_fret.as_ref(), values_to_filter.average_fret) {
            failed_tests.push(FilterTest::AverageFRET);
        }

        // Determine overall success based on whether any tests failed
        let success = failed_tests.is_empty();
        (success, failed_tests)
    }
}

fn run_test<T>(comparison: Option<&Comparison<T>>, value: T) -> bool 
where T: PartialEq + PartialOrd
{
    if let Some(test) = comparison {
        test.compare(value)
    } else {
        true
    }
}

#[derive(Debug, Clone)]
pub enum Comparison<T> 
{
    Larger {value: T},
    LargerEq {value: T},
    Smaller {value: T},
    SmallerEq {value: T},
    Equal {value: T},
    WithinNStd{n: usize},
}

impl<T> Comparison<T> 
where T: PartialOrd + PartialEq
{
    pub fn compare(&self, other_value: T) -> bool {
        match self {
            Self::Larger { value } => other_value > *value,
            Self::LargerEq { value } => other_value >= *value,
            Self::Smaller { value } => other_value < *value,
            Self::SmallerEq { value } => other_value <= *value,
            Self::Equal { value } => other_value == *value,
            _ => false,
        }
    }
}

impl<T> Comparison<T> {
    pub fn compare_min_max_mean_std(&self, other: &MinMaxMeanStd) -> bool {
        match self {
            Self::WithinNStd { n } => {
                let n_std_range = other.std * (*n as f64);
                let min_allowed = other.mean - n_std_range;
                let max_allowed = other.mean + n_std_range;
                other.min >= min_allowed && other.max <= max_allowed
            },
            _ => false,
        }
    }
}