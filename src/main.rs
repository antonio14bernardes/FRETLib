use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::probability_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::HMM;

fn main() {
    // Create some State objects with the same IDs for testing
    let state1 = State::new(0, 10.0, 1.0);
    let state2 = State::new(1, 20.0, 1.0); // Changed the ID to 2
    let state3 = State::new(2, 30.0, 1.0); // Changed the ID to 3

    // Store states in an array or slice
    let states = [state1, state2, state3];

    // Create a start matrix where state 1 has 50% chance and the others 25%
    let start_matrix_raw: Vec<f64> = vec![0.5, 0.25, 0.25];
    let start_matrix = StartMatrix::new(start_matrix_raw);

    // Create a transition matrix
    // Let's assume the following transition probabilities:
    // State 1 -> State 1: 0.5, State 1 -> State 2: 0.25, State 1 -> State 3: 0.25
    // State 2 -> State 1: 0.3, State 2 -> State 2: 0.4, State 2 -> State 3: 0.3
    // State 3 -> State 1: 0.2, State 3 -> State 2: 0.3, State 3 -> State 3: 0.5
    let transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.5, 0.25, 0.25],
        vec![0.3, 0.4, 0.3],
        vec![0.2, 0.3, 0.5],
    ];
    let mut transition_matrix = TransitionMatrix::new(transition_matrix_raw);

    let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

    let (sequence_ids, sequence_values) = HMM::gen_sequence(&states, &start_matrix, &transition_matrix, 20);

    let pred = hmm.run_viterbi(&sequence_values);

    let transition_matrix_raw: Vec<Vec<f64>> = vec![
        vec![0.25, 0.5, 0.25],
        vec![0.3, 0.3, 0.4],
        vec![0.2, 0.3, 0.5],
    ];

    drop(hmm);

    transition_matrix = TransitionMatrix::new(transition_matrix_raw);

    let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

    hmm.run_forward(&sequence_values);

}

