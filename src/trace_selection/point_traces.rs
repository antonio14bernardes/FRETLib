use super::filter::{FilterTest, MinMaxMeanStd, ValuesToFilter};
use super::individual_trace::*;
use super::tools::compute_mean_and_std;
use std::collections::HashSet;


#[derive(Debug, Clone)]
pub struct PointTraces {
    traces: HashSet<IndividualTrace>,
    metadata: Option<PointFileMetadata>,

    // Useful tracker
    don_acc_pair: Option<[TraceType; 2]>, // Pair needs to be made up of either two processed or two raw traces
    donor: Option<TraceType>, // Can be either raw or processed
    acceptor: Option<TraceType>, // Can be either raw or processed


    // Statistics and events 
    acceptor_photobleaching: Option<Vec<usize>>, // (Last event, number of events)
    donor_photobleaching: Option<Vec<usize>>, // (Last event, number of events)
    fret_lifetimes: Option<Vec<[usize; 2]>>,
    // donor_blinking_events: Option<Vec<usize>>,
    snr_signal: Option<f64>, // SNR of total intensity before donor bleaching
    snr_backgroung: Option<f64>, // Mean of intensity before donor bleaching divided by s.d. of intensity after donor bleaching
    correlation_coef: Option<f64>, // Correlation between donor and acceptor
    background_std_post_bleach: Option<f64>,
    intensity_min_max_mean_std: Option<MinMaxMeanStd>,
    first_fret: Option<f64>,
    max_fret: Option<f64>,
    average_fret: Option<f64>,
}

impl PointTraces {
    pub fn new(traces: HashSet<IndividualTrace>, metadata_obj: PointFileMetadata) -> Self {
        let mut new = Self {
            traces,
            metadata: Some(metadata_obj), 

            don_acc_pair: None,
            donor: None,
            acceptor: None,

            donor_photobleaching: None,
            acceptor_photobleaching: None,
            fret_lifetimes: None,
            // donor_blinking_events: None,
            snr_signal: None,
            snr_backgroung: None,
            correlation_coef: None,
            background_std_post_bleach: None,
            intensity_min_max_mean_std: None,
            first_fret: None,
            max_fret: None,
            average_fret: None,

        };

        let _ = new.update_donor_acceptor();

        new
    }

    pub fn new_empty() -> Self {
        Self {
            traces: HashSet::new(), 
            metadata: None, 

            don_acc_pair: None,
            donor: None,
            acceptor: None,

            donor_photobleaching: None,
            acceptor_photobleaching: None,
            fret_lifetimes: None,
            // donor_blinking_events: None,
            snr_signal: None,
            snr_backgroung: None,
            correlation_coef: None,
            background_std_post_bleach: None,
            intensity_min_max_mean_std: None,

            first_fret: None,
            max_fret: None,
            average_fret: None,
        }
    }

    pub fn insert_new_trace(&mut self, values: Vec<f64>, trace_type: TraceType) -> Result<(), PointTracesError> {
        // Try to create a new IndividualTrace
        match IndividualTrace::new(values, trace_type.clone()) {
            Ok(trace) => {
                // Try to insert the trace into the HashSet
                if self.traces.insert(trace) {
                    self.update_donor_acceptor();

                    Ok(())
                } else {
                    Err(PointTracesError::TraceTypeAlreadyPresent{trace_type: trace_type.clone()})
                }
            }
            Err(e) => Err(PointTracesError::IndividualTraceError { error: e, trace_type }),
        }
    }

    pub fn insert_trace(&mut self, trace: IndividualTrace) -> Result<(), PointTracesError> {
        let trace_type = trace.get_type().clone();
        if self.traces.insert(trace)  {
            self.update_donor_acceptor();

            let _ = self.update_donor_acceptor();

            Ok(())
        } else {
            Err(PointTracesError::TraceTypeAlreadyPresent{trace_type: trace_type})
        }
    }

    pub fn remove_trace(&mut self, trace_type: &TraceType) -> Result<(), PointTracesError> {
        let wrapper = IndividualTrace::trace_type_wrapper(trace_type);
        let trace_type = trace_type.clone();
        if self.traces.remove(&wrapper)  {
            self.update_donor_acceptor();
            Ok(())
        } else {
            Err(PointTracesError::TraceTypeNotPresent{trace_type})
        }
    }

    pub fn get_trace(&self, trace_type: &TraceType) -> Option<&IndividualTrace> {
        let wrapper = IndividualTrace::trace_type_wrapper(trace_type);

        self.traces.get(&wrapper)
    }

    pub fn take_trace(&mut self, trace_type: &TraceType) -> Option<IndividualTrace> {
        let wrapper = IndividualTrace::trace_type_wrapper(trace_type);

        let output = self.traces.take(&wrapper);

        self.update_donor_acceptor();

        output
    }

    pub fn get_types(&self) -> Vec<TraceType> {
        let mut types_vec: Vec<TraceType> = Vec::new();

        for trace in self.traces.iter() {
            types_vec.push(trace.get_type().clone());
        }

        types_vec
    }

    pub fn insert_metadata(&mut self, metadata_obj: PointFileMetadata) {
        self.metadata = Some(metadata_obj);
    }

    pub fn get_metadata(&self) -> Option<&PointFileMetadata> {
        self.metadata.as_ref()
    }

    pub fn get_valid_fret(&self) -> Result<&[f64], PointTracesError> {
        // println!("Got to the point traces function");
        // Get the FRET trace
        let fret = self.get_trace(&TraceType::FRET)
        .ok_or(PointTracesError::TraceTypeNotPresent { trace_type: TraceType::FRET })?;

        // Extract values
        let values = fret.get_values();
        // println!("Values: {:?}", values);

        // Extract photobleaching idx
        let donor_pb_option = self.donor_photobleaching.clone();
        let acceptor_pb_option = self.acceptor_photobleaching.clone();

        // if donor_pb_option.is_none() || acceptor_pb_option.is_none() {
        //     return Err(PointTracesError::FilteringNotPerformed)
        // }

        let donor_pb_vec: Vec<usize> = donor_pb_option.unwrap_or(Vec::new());
        let acceptor_db_vec: Vec<usize> = acceptor_pb_option.unwrap_or(Vec::new());

        let donor_pb = if donor_pb_vec.len() > 0 {donor_pb_vec[donor_pb_vec.len() - 1]} else {values.len()-1};
        let acceptor_pb = if acceptor_db_vec.len() > 0 {acceptor_db_vec[acceptor_db_vec.len() - 1]} else {values.len()-1};

        let earliest_pb = if donor_pb < acceptor_pb {donor_pb} else {acceptor_pb};

        Ok(&values[..earliest_pb])
    }

    pub fn check_donor_acceptor_pair(&mut self) -> Result<(), PointTracesError> {

        // Priority is given to a processed pair
        let dem_dex = self.get_trace(&TraceType::DemDexc);
        let aem_dex = self.get_trace(&TraceType::AemDexc);

        if dem_dex.is_some() && aem_dex.is_some() {
            self.don_acc_pair = Some([TraceType::DemDexc, TraceType::AemDexc]);
            return Ok(());
        }

        // If no processed pair, check for a raw pair
        let raw_dem_dex = self.get_trace(&TraceType::RawDemDexc);
        let raw_aem_dex = self.get_trace(&TraceType::RawAemDexc);

        if raw_dem_dex.is_some() && raw_aem_dex.is_some() {
            self.don_acc_pair = Some([TraceType::RawDemDexc, TraceType::RawAemDexc]);
            return Ok(());
        }
        
        self.don_acc_pair = None;
        Err(PointTracesError::NoDonorAcceptorPairFound)
    }

    pub fn check_donor(&mut self) -> Result<(), PointTracesError> {
        // Priority is given to a processed pair
        if self.get_trace(&TraceType::DemDexc).is_some() {
            self.donor = Some(TraceType::DemDexc);
            return Ok(());
        }

        if self.get_trace(&TraceType::RawDemDexc).is_some() {
            self.donor = Some(TraceType::RawDemDexc);
            return Ok(());
        }

        self.donor = None;
        Err(PointTracesError::NoDonorFound)

    }

    pub fn check_acceptor(&mut self) -> Result<(), PointTracesError> {

        // Priority is given to a processed pair
        if self.get_trace(&TraceType::AemDexc).is_some() {
            self.acceptor = Some(TraceType::AemDexc);
            return Ok(());
        }

        if self.get_trace(&TraceType::RawAemDexc).is_some() {
            self.acceptor = Some(TraceType::RawAemDexc);
            return Ok(());
        }

        self.acceptor = None;
        Err(PointTracesError::NoAcceptorFound)

    }

    pub fn check_donor_acceptor(&mut self) -> Result<(), PointTracesError> {
        self.check_donor()?;
        self.check_acceptor()?;

        Ok(())
    }

    pub fn get_donor_acceptor_pair(&self) -> Option<&[TraceType;2]> {
        self.don_acc_pair.as_ref()
    }

    pub fn get_donor_acceptor(&self) -> [Option<&TraceType>;2] {
        [self.donor.as_ref(), self.acceptor.as_ref()]
    }


    pub fn update_donor_acceptor(&mut self) {
        let _ = self.check_donor_acceptor_pair();
        let _ = self.check_donor();
        let _ = self.check_acceptor();
    }


    pub fn detect_photobleaching(&mut self, photobleaching_filter_values: &PhotobleachingFilterValues) -> Result<(), PointTracesError> {
        
        // Retrieve parameters to detect photobleaching
        let window_size = photobleaching_filter_values.median_filter_window_size;
        let threshold = photobleaching_filter_values.noise_threshold_multiple;

        if self.donor.is_some() {
            let donor_type = self.donor.as_ref().unwrap().clone();

            let donor = self.get_trace(&donor_type).unwrap();

            let all_events = donor.detect_photobleaching_events(threshold, window_size);

            self.donor_photobleaching = Some(all_events);
        }

        if self.acceptor.is_some() {
            let acceptor_type = self.acceptor.as_ref().unwrap().clone();

            let acceptor = self.get_trace(&acceptor_type).unwrap();

            let all_events = acceptor.detect_photobleaching_events(threshold, window_size);

            self.acceptor_photobleaching = Some(all_events);
        }

        if self.donor.is_none() {
            return Err(PointTracesError::NoDonorFound);
        }

        if self.acceptor.is_none() {
            return Err(PointTracesError::NoAcceptorFound);
        }

        Ok(())
    }


    pub fn compute_total_pair_intensity(&mut self) -> Result<(), PointTracesError> {

        let [donor, acceptor] = self.don_acc_pair.as_ref().ok_or(PointTracesError::NoDonorAcceptorPairFound)?;

        let d_values = self.get_trace(donor).unwrap().get_values();
        let a_values = self.get_trace(acceptor).unwrap().get_values();
        let mut sum_vec: Vec<f64> = Vec::new();

        for (d, a) in d_values.iter().zip(a_values) {
            sum_vec.push(a + d);
        }

        if let Ok(new_trace) = IndividualTrace::new(sum_vec, TraceType::TotalPairIntensity) {
            self.insert_trace(new_trace)?;
        } 

        Ok(())
    }

    pub fn compute_intensity_snr_and_noise(&mut self) -> Result<(), PointTracesError> {
        // Retrieve donor photobleaching time step
        let donor_pb_vec =
        self.donor_photobleaching.as_ref().ok_or(PointTracesError::PhotobleachingDetectionNotPerformed)?;

        let donor_pb = if donor_pb_vec.len() == 0 {0} else {donor_pb_vec[donor_pb_vec.len() - 1]};
        let pb_input = if donor_pb > 0 {Some(donor_pb)} else {None};

        // Get the intensity trace
        let total_intensity = self.get_trace(&TraceType::TotalPairIntensity)
        .ok_or(PointTracesError::TraceTypeNotPresent { trace_type: TraceType::TotalPairIntensity })?;

        // Get the SNRs
        
        let out = total_intensity.snr_before_after_bleaching(pb_input)
        .map_err(|error| PointTracesError::IndividualTraceError { error, trace_type: TraceType::TotalPairIntensity })?;

        let snr_before = out[0].unwrap().clone();
        let snr_after = out[1].unwrap_or(0.0); // If no photobleaching set to zero

        // Get the noise
        let out = total_intensity.noise_post_bleaching(pb_input)
        .map_err(|error| PointTracesError::IndividualTraceError { error, trace_type: TraceType::TotalPairIntensity })?;

        let noise = out.unwrap_or(0.0);


        // Update struct fields
        self.snr_signal = Some(snr_before);
        self.snr_backgroung = Some(snr_after);
        self.background_std_post_bleach = Some(noise);

        

        Ok(())
    }



    pub fn compute_pair_correlation(&mut self) -> Result<(), PointTracesError> {
        // Ensure donor-acceptor pair exists
        let [donor, acceptor] = self.don_acc_pair.as_ref()
            .ok_or(PointTracesError::NoDonorAcceptorPairFound)?;
    
        // Retrieve donor and acceptor trace values
        let d_values = self.get_trace(donor).unwrap().get_values();
        let a_values = self.get_trace(acceptor).unwrap().get_values();
    
        // Determine the earliest photobleaching index
        let donor_pb_idx = self.donor_photobleaching
            .as_ref()
            .and_then(|v| v.last().copied())
            .unwrap_or(d_values.len());
    
        let acceptor_pb_idx = self.acceptor_photobleaching
            .as_ref()
            .and_then(|v| v.last().copied())
            .unwrap_or(a_values.len());
    
        let earliest_pb = donor_pb_idx.min(acceptor_pb_idx);
    
        // Truncate the values up to the earliest photobleaching index
        let d_values_truncated = &d_values[..earliest_pb];
        let a_values_truncated = &a_values[..earliest_pb];
    
        // Compute mean and standard deviation for truncated values
        let [d_mean, d_std] = compute_mean_and_std(d_values_truncated);
        let [a_mean, a_std] = compute_mean_and_std(a_values_truncated);
    
        // Check for zero standard deviation to prevent division by zero
        let denominator = d_std * a_std;
        if denominator == 0.0 {
            return Err(PointTracesError::StdZero);
        }
    
        // Compute numerator for correlation coefficient
        let mut numerator = 0.0;
        for (d, a) in d_values_truncated.iter().zip(a_values_truncated) {
            numerator += (d - d_mean) * (a - a_mean);
        }
        numerator /= (d_values_truncated.len() - 1) as f64;
    
        // Update correlation coefficient in struct
        self.correlation_coef = Some(numerator / denominator);
    
        Ok(())
    }

    pub fn first_max_average_fret(&mut self) -> Result<(), PointTracesError> {
        // Retrieve donor photobleaching time step
        let donor_pb_vec =
        self.donor_photobleaching.as_ref().ok_or(PointTracesError::PhotobleachingDetectionNotPerformed)?;

        let donor_pb = if donor_pb_vec.len() == 0 {0} else {donor_pb_vec[donor_pb_vec.len() - 1]};
        let pb_input = if donor_pb > 0 {Some(donor_pb)} else {None};



        // Get the fret trace
        let fret = self.get_trace(&TraceType::FRET)
        .ok_or(PointTracesError::TraceTypeNotPresent { trace_type: TraceType::FRET })?;

        // Get the interesting stuff
        let [max, _min] = fret.max_min_values();
        let first = fret.first_value();
        let average = fret.average_before_bleaching(pb_input)
        .map_err(|error| PointTracesError::IndividualTraceError { error, trace_type: TraceType::TotalPairIntensity })?;

        self.max_fret = Some(max);
        self.first_fret = Some(first);
        self.average_fret = Some(average);

        Ok(())
    }

    pub fn get_intensity_min_max_mean_std(&mut self) -> Result<(), PointTracesError> {
        // Get the intensity trace
        let total_intensity = self.get_trace(&TraceType::TotalPairIntensity)
        .ok_or(PointTracesError::TraceTypeNotPresent { trace_type: TraceType::TotalPairIntensity })?;

        let [max, min] = total_intensity.max_min_values();
        let [mean, std] = compute_mean_and_std(total_intensity.get_values());

        let out_struct = MinMaxMeanStd{min, max, mean, std};
        self.intensity_min_max_mean_std = Some(out_struct);

        Ok(())
    }

    pub fn get_fret_lifetimes(&mut self, fret_lifetimes_filter_values: &FretLifetimesFilterValues) -> Result<(), PointTracesError>  {
        // Get the fret trace
        let fret = self.get_trace(&TraceType::FRET)
        .ok_or(PointTracesError::TraceTypeNotPresent { trace_type: TraceType::FRET })?;

        // Get the lifetimes
        let threshold = fret_lifetimes_filter_values.threshold;
        let min_len = fret_lifetimes_filter_values.min_len;

        let lifetimes = fret.lifetimes(threshold, min_len);

        self.fret_lifetimes = Some(lifetimes);

        Ok(())
    }


    pub fn prepare_filter_values(
        &mut self,
        photobleaching_filter_values: &PhotobleachingFilterValues,
        fret_lifetimes_filter_values: &FretLifetimesFilterValues,
    ) -> Result<ValuesToFilter, PointTracesError>{
        self.update_donor_acceptor();

        // if let Err(e) = self.detect_photobleaching(photobleaching_filter_values) {
        //     if let PointTracesError::NoAcceptorFound = e {} // Ignore this error 
        //     else {
        //         return Err(e); // Only propagate if it's not NoAcceptorFound
        //     }
        // }

        self.detect_photobleaching(photobleaching_filter_values)?;
        self.compute_total_pair_intensity()?;
        self.compute_intensity_snr_and_noise()?;
        self.compute_pair_correlation()?;
        self.get_intensity_min_max_mean_std()?;
        self.get_fret_lifetimes(fret_lifetimes_filter_values)?;
        self.first_max_average_fret()?;

        let snr_bkg = self.snr_backgroung.unwrap();
        let std_bkg = self.background_std_post_bleach.unwrap();



        let values_to_filter = ValuesToFilter{
            donor_photobleaching: self.donor_photobleaching.as_ref().unwrap().clone(),
            fret_lifetimes: self.fret_lifetimes.as_ref().unwrap().clone(),
            // donor_blinking_events: Vec<usize>,
            snr_signal: self.snr_signal.unwrap(), 
            snr_background: if snr_bkg != 0.0 {Some(snr_bkg)} else {None}, // snr is 0 if there was no photobleaching event
            correlation_coef: self.correlation_coef.unwrap(),
            background_std_post_bleach: if std_bkg != 0.0 {Some(std_bkg)} else {None}, // snr is 0 if there was no photobleaching event
            intensity_min_man_mean_std: self.intensity_min_max_mean_std.as_ref().unwrap().clone(),
            first_fret: self.first_fret.unwrap(),
            max_fret: self.max_fret.unwrap(),
            average_fret: self.average_fret.unwrap(),
        };

        Ok(values_to_filter)
    }
}

#[derive(Debug, Clone)]
pub struct PointFileMetadata {
    pub file_path: String,
    pub file_date: String,
    pub movie_filename: String,
    pub fret_pair: usize,
}

impl PointFileMetadata {
    pub fn new(file_path: String, file_date: String, movie_filename: String, fret_pair: usize) -> Self {
        Self {
            file_path,
            file_date,
            movie_filename,
            fret_pair,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PointTracesError {
    TraceTypeAlreadyPresent{trace_type: TraceType},
    TraceTypeNotPresent{trace_type: TraceType},
    IndividualTraceError{error: IndividualTraceError, trace_type: TraceType},
    NoDonorAcceptorPairFound,
    NoDonorFound,
    NoAcceptorFound,
    NoFretFound,
    StdZero,

    PhotobleachingDetectionNotPerformed,
    PhotobleachingNotFound,

    FailedFilterTests{ tests_failed: Vec<FilterTest>},
    FilteringNotPerformed,
}