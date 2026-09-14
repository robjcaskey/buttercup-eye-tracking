/// Optional upright, screen-facing mounting heuristic, not a measured pose.
/// A below-eye camera favors a screen above its optical axis (sensor-up gaze);
/// an above-eye camera favors sensor-down. Camera roll or a different screen
/// arrangement can invalidate this assumption. Never resolve near-horizontal Y.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMount {
    #[default]
    Flexible,
    BelowEyes,
    AboveEyes,
}

impl CameraMount {
    pub(crate) fn next(self)->Self {match self {Self::Flexible=>Self::BelowEyes,Self::BelowEyes=>Self::AboveEyes,Self::AboveEyes=>Self::Flexible}}
    pub(crate) fn label(self)->&'static str {match self {Self::Flexible=>"flexible",Self::BelowEyes=>"below-eyes",Self::AboveEyes=>"above-eyes"}}
    pub(crate) fn parse(value:&str)->Option<Self> {match value.to_ascii_lowercase().as_str() {"flexible"=>Some(Self::Flexible),"below-eyes"|"below"=>Some(Self::BelowEyes),"above-eyes"|"above"=>Some(Self::AboveEyes),_=>None}}
    pub(crate) fn supports(self,down:f64)->bool {down.is_finite() && match self {Self::Flexible=>true,Self::BelowEyes=>down < -0.05,Self::AboveEyes=>down > 0.05}}
    pub(crate) fn branch(self,down:[f64;2])->Option<usize> {
        if self==Self::Flexible {return None;}
        match down.map(|y|self.supports(y)) {[true,false]=>Some(0),[false,true]=>Some(1),_=>None}
    }
}

