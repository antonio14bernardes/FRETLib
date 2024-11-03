use crate::optimization::amalgam_idea::{AmalgamIdea, AmalgamIdeaError};

pub struct AmalgamWrapper<'a> {
    amalgam: AmalgamIdea<'a>,
}



#[derive(Debug)]
pub enum AmalgamWrapperError {
    AmalgamIdeaError{err: AmalgamIdeaError},
    InvalidNumberOfStates,
    InvalidSequence,
}