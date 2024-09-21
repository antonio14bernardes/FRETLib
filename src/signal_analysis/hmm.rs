/********** Hidden Markov Model (HMM) Module for FRET Analysis **********
* This module is designed based on the methodology and algorithm described in the paper:
*
* "Analysis of Single-Molecule FRET Trajectories Using Hidden Markov Modeling"**
*
* Authors: Sean A. McKinney, Chirlmin Joo, and Taekjip Ha  
* Department of Physics and Howard Hughes Medical Institute, University of Illinois at Urbana-Champaign  
* Published in: *Biophysical Journal*, Volume 91, September 2006, Pages 1941–1951  
* DOI: [10.1529/biophysj.106.082487](https://doi.org/10.1529/biophysj.106.082487)
**********/


mod hmm_tools;
mod state;
mod probability_matrices;
mod viterbi;