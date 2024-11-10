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
    PhotobleachingSteps { threshold: usize, received: usize },
    DonorLifetime { min_required: usize, received: usize },
    FretLifetime { min_required: usize, largest_received: usize },
    SNRBackground { min_required: f64, received: f64 },
    CorrelationCoefficient { max_allowed: f64, received: f64 },
    BackgroundNoise { max_allowed: f64, received: f64 },
    MeanTotalIntensity { allowed_range: [f64; 2], received_range: [f64; 2] }, // ranges are [min, max]
    SNRSignal { min_required: f64, received: f64 },
    FRETFirstTimeStep { min_required: f64, received: f64 },
    HighestFRET { max_allowed: f64, received: f64 },
    AverageFRET { max_allowed: f64, received: f64 },
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
    
    pub fn check_valid(&self, values_to_filter: &ValuesToFilter) -> (bool, Vec<FilterTest>) {
        let mut failed_tests = Vec::new();

        // Photobleaching steps test
        if let Some(threshold) = self.photobleaching_steps.as_ref() {
            let received = values_to_filter.donor_photobleaching.len();
            if !run_test(Some(threshold), received) {
                failed_tests.push(FilterTest::PhotobleachingSteps { threshold: 1, received });
            }
        }

        // Donor lifetime test
        let photobleaching_events = &values_to_filter.donor_photobleaching;
        if !photobleaching_events.is_empty() {
            let last_photobleach = photobleaching_events[photobleaching_events.len() - 1];
            if let Some(min_required) = self.donor_lifetime.as_ref() {
                if !run_test(Some(min_required), last_photobleach) {
                    failed_tests.push(FilterTest::DonorLifetime { min_required: 50, received: last_photobleach });
                }
            }
        }

        // FRET lifetime test
        let lifetimes = &values_to_filter.fret_lifetimes;
        if lifetimes.is_empty() && self.fret_lifetimes.is_some() {
            failed_tests.push(FilterTest::FretLifetime { min_required: 15, largest_received: 0 });
        } else {
            let largest_lifetime = lifetimes.iter().map(|lifetime| lifetime[1]).max().unwrap_or(0);
            if let Some(min_required) = self.fret_lifetimes.as_ref() {
                if !run_test(Some(min_required), largest_lifetime) {
                    failed_tests.push(FilterTest::FretLifetime { min_required: 15, largest_received: largest_lifetime });
                }
            }
        }

        // SNR background test
        if let Some(min_required) = self.snr_background.as_ref() {
            let received = values_to_filter.snr_backgroung;
            if !run_test(Some(min_required), received) {
                failed_tests.push(FilterTest::SNRBackground { min_required: 8.0, received });
            }
        }

        // Correlation coefficient test
        if let Some(max_allowed) = self.correlation_coefficient.as_ref() {
            let received = values_to_filter.correlation_coef;
            if !run_test(Some(max_allowed), received) {
                failed_tests.push(FilterTest::CorrelationCoefficient { max_allowed: 0.5, received });
            }
        }

        // Background noise test
        if let Some(max_allowed) = self.background_noise.as_ref() {
            let received = values_to_filter.background_std_post_bleach;
            if !run_test(Some(max_allowed), received) {
                failed_tests.push(FilterTest::BackgroundNoise { max_allowed: 70.0, received });
            }
        }

        // Mean total intensity test
        if let Some(test) = &self.mean_total_intensity {

            let min = values_to_filter.intensity_min_man_mean_std.min;
            let max = values_to_filter.intensity_min_man_mean_std.max;
            let mean = values_to_filter.intensity_min_man_mean_std.mean;
            let std = values_to_filter.intensity_min_man_mean_std.std;

            let mut n_fr = 0.0;

            if let Comparison::WithinNStd { n } = test {
                n_fr = *n as f64;
            }

            let allowed_range = [mean - n_fr * std, mean + n_fr * std];
            let received_range = [min, max];
            
            if !test.compare_min_max_mean_std(&values_to_filter.intensity_min_man_mean_std) {
                failed_tests.push(FilterTest::MeanTotalIntensity { allowed_range, received_range });
            }
        }

        // SNR signal test
        if let Some(min_required) = self.snr_signal.as_ref() {
            let received = values_to_filter.snr_signal;
            if !run_test(Some(min_required), received) {
                failed_tests.push(FilterTest::SNRSignal { min_required: 10.0, received });
            }
        }

        // FRET first time step test
        if let Some(min_required) = self.fret_first_time_step.as_ref() {
            let received = values_to_filter.first_fret;
            if !run_test(Some(min_required), received) {
                failed_tests.push(FilterTest::FRETFirstTimeStep { min_required: 0.2, received });
            }
        }

        // Highest FRET test
        if let Some(max_allowed) = self.highest_fret.as_ref() {
            let received = values_to_filter.max_fret;
            if !run_test(Some(max_allowed), received) {
                failed_tests.push(FilterTest::HighestFRET { max_allowed: 0.9, received });
            }
        }

        // Average FRET test
        if let Some(max_allowed) = self.average_fret.as_ref() {
            let received = values_to_filter.average_fret;
            if !run_test(Some(max_allowed), received) {
                failed_tests.push(FilterTest::AverageFRET { max_allowed: 0.7, received });
            }
        }

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