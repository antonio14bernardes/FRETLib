use std::collections::HashSet;

use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::optimization_tracker::TerminationCriterium;
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::hmm_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::{baum_welch, HMM};
use fret_lib::signal_analysis::hmm::baum_welch::*;
use rand::seq;
use plotters::prelude::*;



fn main() {
        let real_state1 = State::new(0, 10.0, 1.0).unwrap();
        let real_state2 = State::new(1, 20.0, 2.0).unwrap();
        let real_state3 = State::new(2, 30.0, 3.0).unwrap();

        let real_states = [real_state1, real_state2, real_state3].to_vec();

        let real_start_matrix_raw: Vec<f64> = vec![0.1, 0.8, 0.1];
        let real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
        let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.2, 0.05, 0.75],
        ];
        let real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

        let (sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 600);


        /****** Create slightly off states and matrices ******/


        let fake_state1 = State::new(0, 9.0, 1.2).unwrap();
        let fake_state2 = State::new(1, 23.0, 1.2).unwrap();
        let fake_state3 = State::new(2, 28.0, 2.7).unwrap();

        let fake_states = [fake_state1, fake_state2, fake_state3].to_vec();

        let fake_start_matrix_raw: Vec<f64> = vec![0.4, 0.3, 0.3];
        let fake_start_matrix = StartMatrix::new(fake_start_matrix_raw);

        
        let fake_transition_matrix_raw: Vec<Vec<f64>> = vec![
            vec![0.3, 0.3, 0.4],
            vec![0.4, 0.5, 0.1],
            vec![0.6, 0.1, 0.3],
        ];
        let fake_transition_matrix = TransitionMatrix::new(fake_transition_matrix_raw);

        let mut baum = BaumWelch::new(3);

        baum.set_initial_states(fake_states).unwrap();
        baum.set_initial_start_matrix(fake_start_matrix).unwrap();
        baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();

        let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 100 };

        let output_res = baum.run_optimization(&sequence_values, termination_criterium);

        if output_res.is_err() {
            println!("Optimization failed: {:?}", output_res);
            return
        }



        // For plotting
        let opt_states = baum.take_states().unwrap();
        let opt_start_matrix = baum.take_start_matrix().unwrap();
        let opt_transition_matrix = baum.take_transition_matrix().unwrap();
        let obs_prob = baum.take_observations_prob().unwrap();

        println!("New states: {:?}", &opt_states);
        println!("New start matrix: {:?}", &opt_start_matrix);
        println!("New transition matrix: {:?}", &opt_transition_matrix);
        println!("Final observations prob: {}", &obs_prob);

        let mut viterbi = Viterbi::new(&opt_states, &opt_start_matrix, &opt_transition_matrix);
        viterbi.run(&sequence_values, true);

        let predictions = viterbi.get_prediction().unwrap();

        let sum: u16 = predictions.iter().zip(sequence_ids).map(|(a,b)| if *a == b {1_u16} else {0_u16}).sum();
        let accuracy = (sum as f64) / (predictions.len() as f64);

        println!("Prediction accuracy {}", accuracy);

        let pred_ideal_sequence: Vec<f64> = predictions.iter().map(|id| opt_states[*id].get_value()).collect();

        plot_sequences(&sequence_values, &pred_ideal_sequence).expect("Error while plotting sequences");

}


fn plot_sequences(sequence_values: &[f64], pred_ideal_sequence: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("sequence_plot.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let x_range = 0..sequence_values.len();
    let y_min = sequence_values.iter().cloned().fold(f64::INFINITY, f64::min)
        .min(pred_ideal_sequence.iter().cloned().fold(f64::INFINITY, f64::min));
    let y_max = sequence_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        .max(pred_ideal_sequence.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    let mut chart = ChartBuilder::on(&root_area)
        .caption("Sequence Values and Predicted Sequence", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range.clone(), y_min..y_max)?;

    chart.configure_mesh().draw()?;

    // Plot sequence values as a red line
    chart.draw_series(LineSeries::new(
        sequence_values.iter().enumerate().map(|(i, &v)| (i, v)),
        &RED,
    ))?.label("Sequence Values")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot predicted ideal sequence values as a blue line
    chart.draw_series(LineSeries::new(
        pred_ideal_sequence.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?.label("Predicted Ideal Sequence")
      .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Add a legend to the chart
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}