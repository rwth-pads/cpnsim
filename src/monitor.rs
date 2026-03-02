use rhai::{AST, Dynamic};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The type of a monitor — determines its behavior during simulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum MonitorType {
    /// Track token count on watched places (built-in, no script needed)
    MarkingSize,
    /// Count firings of watched transitions (built-in, no script needed)
    TransitionCount,
    /// Stop simulation when a place-based predicate is true
    BreakpointPlace,
    /// Stop simulation when a watched transition fires
    BreakpointTransition,
    /// User-defined data collector with Rhai observation function
    DataCollector,
}

/// Configuration passed from the frontend to register a monitor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorConfig {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub monitor_type: MonitorType,
    pub enabled: bool,
    /// IDs of places this monitor watches
    #[serde(default)]
    pub place_ids: Vec<String>,
    /// IDs of transitions this monitor watches
    #[serde(default)]
    pub transition_ids: Vec<String>,
    /// Rhai script for the observation function.
    /// Has access to: `markings` (map: place_id -> token count),
    /// `step`, `time`, `transition_id`, `transition_name`.
    /// Should return a numeric value.
    #[serde(default)]
    pub observation_script: String,
    /// Rhai script for the predicate (breakpoints / conditional recording).
    /// Same scope as observation_script. Should return bool.
    #[serde(default)]
    pub predicate_script: String,
    /// For built-in breakpoints: "empty", "not-empty", "enabled", "not-enabled"
    #[serde(default)]
    pub stop_condition: Option<String>,
}

/// A single data point recorded by a monitor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorObservation {
    pub step: u64,
    pub time: i64,
    pub value: f64,
}

/// Running statistics for a monitor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorStatistics {
    pub count: u64,
    pub sum: f64,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

impl MonitorStatistics {
    pub fn new() -> Self {
        MonitorStatistics {
            count: 0,
            sum: 0.0,
            avg: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            std_dev: 0.0,
        }
    }

    /// Update statistics with a new value (Welford's online algorithm for variance)
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.avg = self.sum / self.count as f64;

        // Welford's online variance: we track sum-of-squares-of-differences
        // For simplicity, recompute stddev from avg (acceptable for monitor use)
        // A more efficient approach would track M2, but this is fine for typical monitor sizes
    }
}

/// Accumulated results for one monitor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitorResult {
    pub monitor_id: String,
    pub monitor_name: String,
    pub monitor_type: MonitorType,
    pub observations: Vec<MonitorObservation>,
    pub statistics: MonitorStatistics,
    /// Whether this monitor triggered a breakpoint (stop requested)
    pub breakpoint_hit: bool,
}

impl MonitorResult {
    pub fn new(id: &str, name: &str, monitor_type: MonitorType) -> Self {
        MonitorResult {
            monitor_id: id.to_string(),
            monitor_name: name.to_string(),
            monitor_type,
            observations: Vec::new(),
            statistics: MonitorStatistics::new(),
            breakpoint_hit: false,
        }
    }

    pub fn record(&mut self, step: u64, time: i64, value: f64) {
        self.observations.push(MonitorObservation { step, time, value });
        self.statistics.update(value);
    }

    /// Recompute stddev from all observations (called at the end or on demand)
    pub fn finalize_statistics(&mut self) {
        if self.statistics.count == 0 {
            return;
        }
        let avg = self.statistics.avg;
        let variance: f64 = self
            .observations
            .iter()
            .map(|o| (o.value - avg).powi(2))
            .sum::<f64>()
            / self.statistics.count as f64;
        self.statistics.std_dev = variance.sqrt();
    }
}

/// A registered monitor inside the simulator, with compiled ASTs.
#[derive(Debug)]
pub struct CompiledMonitor {
    pub config: MonitorConfig,
    pub observation_ast: Option<AST>,
    pub predicate_ast: Option<AST>,
    pub result: MonitorResult,
    pub step_count: u64, // for transition-count: how many times the transition fired
}

impl CompiledMonitor {
    pub fn new(config: MonitorConfig, observation_ast: Option<AST>, predicate_ast: Option<AST>) -> Self {
        let result = MonitorResult::new(&config.id, &config.name, config.monitor_type.clone());
        CompiledMonitor {
            config,
            observation_ast,
            predicate_ast,
            result,
            step_count: 0,
        }
    }
}

/// Context passed to monitor evaluation — contains simulation state at the moment of firing.
pub struct MonitorContext {
    pub step: u64,
    pub time: i64,
    pub transition_id: String,
    pub transition_name: String,
    /// Token counts per place (place_id -> count)
    pub place_token_counts: HashMap<String, i64>,
    /// Which place IDs had tokens consumed
    pub consumed_place_ids: Vec<String>,
    /// Which place IDs had tokens produced
    pub produced_place_ids: Vec<String>,
    /// The binding variables from the fired transition
    pub binding_variables: HashMap<String, Dynamic>,
}
