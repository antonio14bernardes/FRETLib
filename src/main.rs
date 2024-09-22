use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::probability_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::HMM;
use rand::seq;

fn main() {
    let state1 = State::new(0, 10.0, 1.0);
    let state2 = State::new(1, 20.0, 1.0); // Changed the ID to 2
    let state3 = State::new(2, 30.0, 1.0); // Changed the ID to 3

    let states = [state1, state2, state3];

    let start_matrix_raw: Vec<f64> = vec![0.5, 0.25, 0.25];
    let start_matrix = StartMatrix::new(start_matrix_raw);

    
    let transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.5, 0.25, 0.25],
        vec![0.3, 0.4, 0.3],
        vec![0.2, 0.3, 0.5],
    ];
    let mut transition_matrix = TransitionMatrix::new(transition_matrix_raw);

    let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

    let (sequence_ids, sequence_values) = HMM::gen_sequence(&states, &start_matrix, &transition_matrix, 20);

    hmm.run_viterbi(&sequence_values);
    let viterbi_predictions = hmm.get_viterbi_prediction();

    println!("With first set of matrices: \n");
    println!("Real state sequence: {:?}", sequence_ids);
    println!("Predicted state sequence: {:?}", viterbi_predictions);

    // Change one of the process matrices - need to delete the hmm instance to do so and then create a new one
    drop(hmm);

    let transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.25, 0.5, 0.25],
        vec![0.3, 0.3, 0.4],
        vec![0.2, 0.3, 0.5],
    ];

    transition_matrix = TransitionMatrix::new(transition_matrix_raw);

    let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

    hmm.run_forward(&sequence_values).unwrap();
    hmm.run_backward(&sequence_values).unwrap();

    let alphas = hmm.get_forward_probabilities_matrix();
    let betas = hmm.get_backward_probabilities_matrix();

    println!("With second set of matrices: \n");
    println!("alphas: {:?}", alphas);
    println!("betas: {:?}", betas);  





    

}

