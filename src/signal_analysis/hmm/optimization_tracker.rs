pub struct OptimizationTracker {
    evals: Vec<f64>,
    iters: u32,

    termination_criterium: TerminationCriterium,

    plateau_count: Option<u16> // Used only for plateau convergence termination criterium

}

impl OptimizationTracker {
    pub fn new(termination_criterium: TerminationCriterium) -> Self {
        let plateau_count = match termination_criterium {
            TerminationCriterium::PlateauConvergence { .. } => Some(0),
            _ => None,
        };

        Self {
            evals: Vec::new(),
            iters: 0,
            termination_criterium,
            plateau_count,
        }
        
    }

    pub fn max_iterations(&self) -> bool {
        match self.termination_criterium {
            TerminationCriterium::MaxIterations { max_iterations } => self.iters > max_iterations,
            TerminationCriterium::OneStepConvergence { max_iterations: Some(max_iter), .. } |
            TerminationCriterium::PlateauConvergence { max_iterations: Some(max_iter), .. } => self.iters > max_iter,
            _ => false,
        }
    }

    pub fn one_step_convergence(&self) -> bool {
        if let TerminationCriterium::OneStepConvergence { epsilon, .. } = self.termination_criterium {
            if self.iters < 2 {
                return false; // Need at least two evaluations to calculate delta
            }
    
            let curr = self.evals[self.iters as usize - 1];
            let prev = self.evals[self.iters as usize - 2];

            if prev.abs() < f64::EPSILON {return true} // Terminate if prev is zero to avoid instability

            let delta = (curr - prev).abs() / prev.abs();

            
    
            return delta < epsilon
        }
    
        false
    }

    pub fn plateau_convergence(&mut self) -> bool {
        if let TerminationCriterium::PlateauConvergence { epsilon, plateau_len,  .. } = self.termination_criterium {
            if self.iters < 2 {
                return false;
            }

            let curr = self.evals[self.iters as usize - 1];
            let prev = self.evals[self.iters as usize - 2];

            if prev.abs() < f64::EPSILON {return true} // Terminate if prev is zero to avoid instability

            let delta = (curr - prev).abs() / prev.abs();

            if delta < epsilon {
                self.plateau_count = Some(self.plateau_count.unwrap_or(0) + 1);
            } else {
                self.plateau_count = Some(0);
            }
            
            return self.plateau_count.unwrap_or(0) >= plateau_len;
        }
    
        false
    }
    
    pub fn step(&mut self, new_eval: f64) -> bool {
        self.evals.push(new_eval);
        self.iters += 1;

        self.max_iterations() || self.one_step_convergence() || self.plateau_convergence()
    }
}

pub enum TerminationCriterium {
    MaxIterations {max_iterations: u32},
    OneStepConvergence {epsilon: f64, max_iterations: Option<u32>},
    OneStepConvergenceAbsolute {epsilon: f64, max_iterations: Option<u32>},
    PlateauConvergence {epsilon: f64, plateau_len: u16, max_iterations: Option<u32>},
    PlateauConvergenceAbsolute {epsilon: f64, plateau_len: u16, max_iterations: Option<u32>},
}