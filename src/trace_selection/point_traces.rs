use super::individual_trace::*;
use std::collections::HashSet;


#[derive(Debug)]
pub struct PointTraces {
    traces: HashSet<IndividualTrace>,
    metadata: Option<PointFileMetadata>,

    // Useful tracker
    don_acc_pair: Option<[TraceType; 2]>, // Pair needs to be made up of either two processed or two raw traces
    donor: Option<TraceType>, // Can be either raw or processed
    acceptor: Option<TraceType>, // Can be either raw or processed


    // Statistics and events 
    acceptor_photobleaching: Option<usize>,
    donor_photobleaching: Option<usize>,
    donor_blinking_events: Option<Vec<usize>>, 
    snr_signal: Option<f64>, // SNR before bleaching
    snr_backgroung: Option<f64>, // SNR after bleaching
    correlation_coef: Option<f64>, // Correlation between donor and acceptor
    background_std_post_bleach: Option<f64>,
    mean_total_intensity: Option<f64>,


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
            donor_blinking_events: None,
            snr_signal: None,
            snr_backgroung: None,
            correlation_coef: None,
            background_std_post_bleach: None,
            mean_total_intensity: None
        };

        let _ = new.check_donor_acceptor();
        let _ = new.check_donor_acceptor_pair();

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
            donor_blinking_events: None,
            snr_signal: None,
            snr_backgroung: None,
            correlation_coef: None,
            background_std_post_bleach: None,
            mean_total_intensity: None,
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

    pub fn insert_metadata(&mut self, metadata_obj: PointFileMetadata) {
        self.metadata = Some(metadata_obj);
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
            self.donor = Some(TraceType::AemDexc);
            return Ok(());
        }

        if self.get_trace(&TraceType::RawAemDexc).is_some() {
            self.donor = Some(TraceType::RawAemDexc);
            return Ok(());
        }

        self.acceptor = None;
        Err(PointTracesError::NoDonorFound)

    }

    pub fn check_donor_acceptor(&mut self) -> Result<(), PointTracesError> {
        self.check_donor()?;
        self.check_acceptor()?;

        Ok(())
    }


    pub fn update_donor_acceptor(&mut self) {
        let _ = self.check_donor_acceptor_pair();
        let _ = self.check_donor();
        let _ = self.check_acceptor();
    }


    pub fn detect_photobleaching(&mut self) -> Result<(), PointTracesError> {
        if self.donor.is_some() {
            let donor_type = self.donor.as_ref().unwrap().clone();

            let mut donor = self.take_trace(&donor_type).unwrap();

            // donor.detect_photobleaching_event(threshold, window_size)
        }

        if self.acceptor.is_some() {
            let acceptor_type = self.acceptor.as_ref().unwrap().clone();

            let mut acceptor = self.take_trace(&acceptor_type).unwrap();

            // acceptor.detect_photobleaching_event(threshold, window_size)
        }



        if self.donor.is_none() {
            return Err(PointTracesError::NoDonorFound);
        }

        if self.acceptor.is_none() {
            return Err(PointTracesError::NoAcceptorFound);
        }

        Ok(())
    }


    
}

#[derive(Debug)]
pub struct PointFileMetadata {
    file_date: String,
    movie_filename: String,
    fret_pair: usize,
}

impl PointFileMetadata {
    pub fn new(file_date: String, movie_filename: String, fret_pair: usize) -> Self {
        Self {
            file_date,
            movie_filename,
            fret_pair,
        }
    }
}

#[derive(Debug)]
pub enum PointTracesError {
    TraceTypeAlreadyPresent{trace_type: TraceType},
    TraceTypeNotPresent{trace_type: TraceType},
    IndividualTraceError{error: IndividualTraceError, trace_type: TraceType},
    NoDonorAcceptorPairFound,
    NoDonorFound,
    NoAcceptorFound,
}