use fret_lib::trace_selection::individual_trace::*;
use fret_lib::trace_selection::point_traces;
use fret_lib::trace_selection::trace_loader::*;
use fret_lib::trace_selection::individual_trace::*;

fn main() {
    let normal_path: &str = "/Users/antonio14bernardes/Documents/Internship/traces_data/Traces_T23_01_sif_pair5.txt";
    let d_bleach_path = "/Users/antonio14bernardes/Documents/Internship/traces_data/D_bleach_T23_01_sif_pair5.txt";
    let a_bleach_path = "/Users/antonio14bernardes/Documents/Internship/traces_data/a_bleach_T23_01_sif_pair7.txt";

    let file_path = a_bleach_path;

    let mut point_traces = parse_file(file_path).unwrap();

    // let trace = point_traces.take_trace(&TraceType::AemDexc).unwrap();

    println!("Output: {:?}",point_traces.compute_pair_correlation());

    
}