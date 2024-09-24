use fret_lib::signal_analysis::hmm::hmm_instance::HMMInstance;
use fret_lib::signal_analysis::hmm::optimization_tracker::TerminationCriterium;
use fret_lib::signal_analysis::hmm::state::State;
use fret_lib::signal_analysis::hmm::probability_matrices::{StartMatrix, TransitionMatrix};
use fret_lib::signal_analysis::hmm::viterbi::Viterbi;
use fret_lib::signal_analysis::hmm::HMM;
use fret_lib::signal_analysis::hmm::baum_welch::*;
use rand::seq;


fn main() {
        let real_state1 = State::new(0, 10.0, 1.0).unwrap();
        let real_state2 = State::new(1, 20.0, 1.0).unwrap(); // Changed the ID to 2
        let real_state3 = State::new(2, 30.0, 1.0).unwrap(); // Changed the ID to 3

        let mut real_states = [real_state1, real_state2, real_state3].to_vec();

        let real_start_matrix_raw: Vec<f64> = vec![0.5, 0.25, 0.25];
        let mut real_start_matrix = StartMatrix::new(real_start_matrix_raw);

        
        let real_transition_matrix_raw: Vec<Vec<f64>> = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.3, 0.4, 0.3],
            vec![0.2, 0.3, 0.5],
        ];
        let mut real_transition_matrix = TransitionMatrix::new(real_transition_matrix_raw);

        let (sequence_ids, sequence_values) = HMM::gen_sequence(&real_states, &real_start_matrix, &real_transition_matrix, 20);


        /****** Create slightly off states and matrices ******/


        let fake_state1 = State::new(0, 9.0, 1.0).unwrap();
        let fake_state2 = State::new(1, 21.0, 1.0).unwrap(); // Changed the ID to 2
        let fake_state3 = State::new(2, 32.0, 1.0).unwrap(); // Changed the ID to 3

        let mut fake_states = [fake_state1, fake_state2, fake_state3].to_vec();

        let fake_start_matrix_raw: Vec<f64> = vec![0.6, 0.20, 0.20];
        let mut fake_start_matrix = StartMatrix::new(fake_start_matrix_raw);

        
        let fake_transition_matrix_raw: Vec<Vec<f64>> = vec![
            vec![0.6, 0.2, 0.2],
            vec![0.35, 0.4, 0.25],
            vec![0.1, 0.4, 0.5],
        ];
        let mut fake_transition_matrix = TransitionMatrix::new(fake_transition_matrix_raw);




        let mut baum = BaumWelch::new(3);

        baum.set_initial_states(fake_states).unwrap();
        baum.set_initial_start_matrix(fake_start_matrix).unwrap();
        baum.set_initial_transition_matrix(fake_transition_matrix).unwrap();

        let termination_criterium = TerminationCriterium::MaxIterations { max_iterations: 300 };

        let result = baum.run_optimization(&sequence_values, termination_criterium);

        println!("result: {:?}", result);


        
}
















// fn main() {
//     let state1 = State::new(0, 10.0, 1.0).unwrap();
//     let state2 = State::new(1, 20.0, 1.0).unwrap(); // Changed the ID to 2
//     let state3 = State::new(2, 30.0, 1.0).unwrap(); // Changed the ID to 3

//     let states = [state1, state2, state3];

//     let start_matrix_raw: Vec<f64> = vec![0.5, 0.25, 0.25];
//     let start_matrix = StartMatrix::new(start_matrix_raw);

    
//     let transition_matrix_raw: Vec<Vec<f64>> = vec![
//         vec![0.5, 0.25, 0.25],
//         vec![0.3, 0.4, 0.3],
//         vec![0.2, 0.3, 0.5],
//     ];
//     let mut transition_matrix = TransitionMatrix::new(transition_matrix_raw);

//     let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

//     let (sequence_ids, sequence_values) = HMM::gen_sequence(&states, &start_matrix, &transition_matrix, 20);

//     hmm.run_viterbi(&sequence_values);
//     let viterbi_predictions = hmm.get_viterbi_prediction();

//     println!("With first set of matrices: \n");
//     println!("Real state sequence: {:?}", sequence_ids);
//     println!("Predicted state sequence: {:?}", viterbi_predictions);

//     // Change one of the process matrices - need to delete the hmm instance to do so and then create a new one
//     drop(hmm);

//     let transition_matrix_raw: Vec<Vec<f64>> = vec![
//         vec![0.25, 0.5, 0.25],
//         vec![0.3, 0.3, 0.4],
//         vec![0.2, 0.3, 0.5],
//     ];

//     transition_matrix = TransitionMatrix::new(transition_matrix_raw);

//     let mut hmm = HMMInstance::new(&states, &start_matrix, &transition_matrix);

//     hmm.run_alphas(&sequence_values).unwrap();
//     hmm.run_betas(&sequence_values).unwrap();
//     hmm.run_gammas(&sequence_values).unwrap();
//     hmm.run_xis(&sequence_values).unwrap();


//     let alphas = hmm.get_alphas();
//     let betas = hmm.get_betas();
//     let gammas = hmm.get_gammas();
//     let xis = hmm.get_xis();

//     println!("With second set of matrices: \n");
//     println!("alphas: {:?}", alphas);
//     println!("betas: {:?}", betas);  
//     println!("gammas: {:?}", gammas);
//     println!("xis: {:?}", xis);  

// }