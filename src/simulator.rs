use crate::model::{PetriNetData, FiringEventData};
use rhai::{Engine, Scope, Dynamic, AST, Module, EvalAltResult, NativeCallContext};
use std::collections::{HashMap, HashSet};
use std::cell::Cell;
use rand::prelude::*;
use rand_distr::{Distribution, Bernoulli, Beta, Binomial, ChiSquared, Exp, Gamma, Normal, Poisson, StudentT, Uniform, Weibull};
use chrono::{DateTime, Utc, Datelike, Timelike, Duration, Weekday};

// Thread-local storage for simulation time context.
// Rhai script-defined functions cannot access scope variables, so we use
// thread-local state that native Rust functions can read directly.
// This is safe because WASM is single-threaded and native Rust tests are
// typically single-threaded per test.
thread_local! {
    static SIM_CURRENT_TIME: Cell<i64> = Cell::new(0);
    static SIM_EPOCH_MS: Cell<i64> = Cell::new(0);
}

// Represents a potential binding for a transition
#[derive(Debug, Clone)]
struct DeferredArcInfo {
    #[allow(dead_code)] // Potentially needed for future debugging or complex scenarios
    arc_id: String,
    place_id: String,
    #[allow(dead_code)] // Used implicitly as key and in comparisons, but compiler might miss it
    variable_name: String,
}

#[derive(Debug, Clone)]
struct Binding {
    variables: HashMap<String, Dynamic>,
    consumed_tokens_map: HashMap<String, Vec<Dynamic>>,
    deferred_list_arcs: HashMap<String, Vec<DeferredArcInfo>>,
}

impl Binding {
    fn new() -> Self {
        Binding {
            variables: HashMap::new(),
            consumed_tokens_map: HashMap::new(),
            deferred_list_arcs: HashMap::new(),
        }
    }
}

fn check_and_consume_multiset(
    required_tokens: &[Dynamic],
    available_indices: &[usize],
    all_tokens_in_place: &[Dynamic],
) -> Option<Vec<Dynamic>> {
    if required_tokens.is_empty() {
        return Some(vec![]);
    }

    let mut required_counts: HashMap<String, usize> = HashMap::new();
    for token in required_tokens {
        *required_counts.entry(token.to_string()).or_insert(0) += 1;
    }

    let available_tokens_to_match: Vec<(usize, Dynamic)> = available_indices
        .iter()
        .map(|&idx| (idx, all_tokens_in_place[idx].clone()))
        .collect();

    let mut consumed_tokens_for_arc: Vec<Dynamic> = Vec::with_capacity(required_tokens.len());
    let mut consumed_indices_from_available: HashSet<usize> = HashSet::new();

    for (req_token_str, req_count) in required_counts {
        let mut found_count = 0;
        let mut indices_to_consume_this_round = Vec::new();

        for (avail_idx, (_original_place_index, avail_token)) in available_tokens_to_match.iter().enumerate() {
            if consumed_indices_from_available.contains(&avail_idx) {
                continue;
            }

            if avail_token.to_string() == req_token_str {
                indices_to_consume_this_round.push(avail_idx);
                found_count += 1;
                if found_count == req_count {
                    break;
                }
            }
        }

        if found_count < req_count {
            return None;
        } else {
            for idx_in_available in indices_to_consume_this_round {
                if consumed_indices_from_available.insert(idx_in_available) {
                    consumed_tokens_for_arc.push(available_tokens_to_match[idx_in_available].1.clone());
                } else {
                    eprintln!("Internal Warning: Attempted to consume the same available token index twice in check_and_consume_multiset.");
                }
            }
        }
    }

    Some(consumed_tokens_for_arc)
}

fn check_and_consume_multiset_multi_place(
    needed_tokens: &[Dynamic],
    available_tokens_info: &[(String, usize, Dynamic)],
) -> Option<HashMap<String, Vec<Dynamic>>> {
    if needed_tokens.is_empty() {
        return Some(HashMap::new());
    }

    let mut needed_counts = create_count_map(needed_tokens);
    let mut consumed_map: HashMap<String, Vec<Dynamic>> = HashMap::new();
    let mut available_counts_subset = HashMap::new();

    for (_, _, token) in available_tokens_info {
        *available_counts_subset.entry(token.to_string()).or_insert(0) += 1;
    }

    for (token_str, needed_count) in &needed_counts {
        if available_counts_subset.get(token_str).copied().unwrap_or(0) < *needed_count {
            return None;
        }
    }

    let mut consumed_count_total = 0;
    for (place_id, _original_index, token) in available_tokens_info {
        let token_str = token.to_string();
        if let Some(count) = needed_counts.get_mut(&token_str) {
            if *count > 0 {
                consumed_map.entry(place_id.clone()).or_default().push(token.clone());
                *count -= 1;
                consumed_count_total += 1;
                if consumed_count_total == needed_tokens.len() {
                    break;
                }
            }
        }
    }

    if consumed_count_total == needed_tokens.len() {
        Some(consumed_map)
    } else {
        eprintln!("Internal Error: Mismatch between count check and token collection in check_and_consume_multiset_multi_place.");
        None
    }
}

/// Check if a variable name appears as a whole word in an inscription.
/// This avoids false positives like matching "s" in "stop" or "str".
fn is_variable_in_inscription(var_name: &str, inscription: &str) -> bool {
    let chars: Vec<char> = inscription.chars().collect();
    let var_chars: Vec<char> = var_name.chars().collect();
    let var_len = var_chars.len();
    
    if var_len == 0 || chars.len() < var_len {
        return false;
    }
    
    for i in 0..=chars.len() - var_len {
        // Check if the substring matches
        let substring_matches = chars[i..i + var_len].iter().zip(var_chars.iter()).all(|(a, b)| a == b);
        
        if substring_matches {
            // Check if preceded by a non-identifier character (or start of string)
            let preceded_ok = if i == 0 {
                true
            } else {
                let prev_char = chars[i - 1];
                !prev_char.is_ascii_alphanumeric() && prev_char != '_'
            };
            
            // Check if followed by a non-identifier character (or end of string)
            let followed_ok = if i + var_len >= chars.len() {
                true
            } else {
                let next_char = chars[i + var_len];
                !next_char.is_ascii_alphanumeric() && next_char != '_'
            };
            
            if preceded_ok && followed_ok {
                return true;
            }
        }
    }
    
    false
}

fn parse_simple_var_array_inscription(inscription: &str) -> (Option<String>, Option<usize>) {
    let trimmed = inscription.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return (None, None);
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return (None, None);
    }

    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return (None, None);
    }

    let first_var = parts[0];
    if first_var.is_empty() || !first_var.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_') || !first_var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return (None, None);
    }

    if parts.iter().all(|&part| part == first_var) {
        (Some(first_var.to_string()), Some(parts.len()))
    } else {
        (None, None)
    }
}

/// Parses a product-style inscription like `[n, p]` into a list of variable names.
/// Returns None if the inscription is not a valid product pattern.
fn parse_product_inscription(inscription: &str) -> Option<Vec<String>> {
    let trimmed = inscription.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return None;
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return None;
    }

    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.is_empty() || parts.len() < 2 {
        return None;
    }

    // Check that all parts are valid variable names (identifiers)
    let mut var_names = Vec::new();
    for part in parts {
        if part.is_empty() {
            return None;
        }
        // Check if it's a valid identifier (starts with letter or underscore, followed by alphanumeric or underscore)
        if !part.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_') {
            return None;
        }
        if !part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return None;
        }
        var_names.push(part.to_string());
    }

    Some(var_names)
}

/// Checks if a token is a product (array) and extracts its components.
fn extract_product_components(token: &Dynamic) -> Option<Vec<Dynamic>> {
    if let Ok(arr) = token.clone().into_typed_array::<Dynamic>() {
        if arr.len() >= 2 {
            Some(arr)
        } else {
            None
        }
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ColorSetKind {
    Unit,
    Bool,
    Int,
    IntInf,
    Time,
    Real,
    String,
    IntRange,
    List,
    Product,
    Record,
    Unknown,
}

#[derive(Debug, Clone)]
struct ParsedColorSet {
    kind: ColorSetKind,
    base_type_name: Option<String>,
    range: Option<(i64, i64)>,
    /// For product types, the list of component colorset names
    component_types: Option<Vec<String>>,
    /// For record types, the list of (field_name, field_type) pairs
    record_fields: Option<Vec<(String, String)>>,
}

impl ParsedColorSet {
    fn unknown() -> Self {
        ParsedColorSet {
            kind: ColorSetKind::Unknown,
            base_type_name: None,
            range: None,
            record_fields: None,
            component_types: None,
        }
    }
}

fn create_count_map(tokens: &[Dynamic]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.to_string()).or_insert(0) += 1;
    }
    counts
}

fn rust_multiset_equal(list1: Vec<Dynamic>, list2: Vec<Dynamic>) -> bool {
    if list1.len() != list2.len() {
        return false;
    }
    let counts1 = create_count_map(&list1);
    let counts2 = create_count_map(&list2);
    counts1 == counts2
}

/// Converts JSON object notation to Rhai map syntax
/// E.g., {"id":1,"name":"John"} -> #{id: 1, name: "John"}
/// Handles arrays of objects: [{"id":1},{"id":2}] -> [#{id: 1}, #{id: 2}]
fn convert_json_to_rhai(json_str: &str) -> Result<String, String> {
    use serde_json::Value;
    
    let parsed: Value = serde_json::from_str(json_str)
        .map_err(|e| format!("Invalid JSON: {}", e))?;
    
    fn value_to_rhai(v: &Value) -> String {
        match v {
            Value::Null => "()".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => format!("\"{}\"", s.replace('\"', "\\\"")),
            Value::Array(arr) => {
                let elements: Vec<String> = arr.iter().map(value_to_rhai).collect();
                format!("[{}]", elements.join(", "))
            }
            Value::Object(obj) => {
                let fields: Vec<String> = obj.iter()
                    .map(|(k, v)| format!("{}: {}", k, value_to_rhai(v)))
                    .collect();
                format!("#{{{}}}", fields.join(", "))
            }
        }
    }
    
    Ok(value_to_rhai(&parsed))
}

fn rhai_multiset_equal_wrapper(context: NativeCallContext, a: Dynamic, b: Dynamic) -> Result<bool, Box<EvalAltResult>> {
    let vec_a = Simulator::dynamic_to_vec_dynamic(a);
    let vec_b = Simulator::dynamic_to_vec_dynamic(b);
    let _ = context;
    Ok(rust_multiset_equal(vec_a, vec_b))
}

/// Represents a token with an optional timestamp (for timed color sets).
/// The timestamp is in milliseconds relative to the simulation epoch.
/// A token is "ready" when current_time >= timestamp.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TimedToken {
    pub value: Dynamic,
    pub timestamp: i64, // Availability time in milliseconds (0 = immediately available)
}

#[allow(dead_code)]
impl TimedToken {
    pub fn new(value: Dynamic, timestamp: i64) -> Self {
        TimedToken { value, timestamp }
    }
    
    pub fn immediate(value: Dynamic) -> Self {
        TimedToken { value, timestamp: 0 }
    }
    
    pub fn is_ready(&self, current_time: i64) -> bool {
        self.timestamp <= current_time
    }
}

impl std::fmt::Display for TimedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.timestamp == 0 {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{}@{}", self.value, self.timestamp)
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Simulator {
    model: PetriNetData,
    // Each place stores a list of token values (kept as Dynamic for compatibility)
    current_marking: HashMap<String, Vec<Dynamic>>,
    // Parallel structure: token timestamps by place_id, indexed same as current_marking
    token_timestamps: HashMap<String, Vec<i64>>,
    // Current simulation time in milliseconds
    current_time: i64,
    // Simulation epoch as ISO 8601 string (for display purposes)
    simulation_epoch: Option<String>,
    // Simulation epoch as milliseconds since Unix epoch (for calendar calculations)
    epoch_ms: i64,
    // Map of place_id -> is_timed (whether the place's colorset is timed)
    timed_places: HashMap<String, bool>,
    rhai_engine: Engine,
    rhai_scope: Scope<'static>,
    guards: HashMap<String, AST>,
    arc_expressions: HashMap<String, AST>,
    initial_marking_expressions: HashMap<String, AST>,
    // Compiled transition time expressions (transition_id -> AST)
    time_expressions: HashMap<String, AST>,
    declared_variables: HashMap<String, String>,
    parsed_color_sets: HashMap<String, ParsedColorSet>,
}

// ============================================================================
// Rhai delay module - provides time delay functions for timed Petri nets
// All delays are expressed in milliseconds internally
// ============================================================================

/// Delay in milliseconds
fn delay_ms(ms: i64) -> i64 {
    ms
}

/// Delay in seconds
fn delay_sec(seconds: i64) -> i64 {
    seconds * 1000
}

/// Delay in minutes  
fn delay_min(minutes: i64) -> i64 {
    minutes * 60 * 1000
}

/// Delay in hours
fn delay_hours(hours: i64) -> i64 {
    hours * 60 * 60 * 1000
}

/// Delay in days
fn delay_days(days: i64) -> i64 {
    days * 24 * 60 * 60 * 1000
}

/// Register the delay module with a Rhai engine
fn register_delay_module(engine: &mut Engine) {
    // Register delay functions with "delay_" prefix for direct access
    engine.register_fn("delay_ms", delay_ms);
    engine.register_fn("delay_sec", delay_sec);
    engine.register_fn("delay_min", delay_min);
    engine.register_fn("delay_hours", delay_hours);
    engine.register_fn("delay_days", delay_days);
}

// ============================================================================
// Calendar / time utility functions for timed Petri nets
// These functions convert simulation time (ms since epoch) to calendar units.
// The Rhai scope provides __sim_time__ (current simulation time in ms since
// sim start) and __sim_epoch_ms__ (simulation epoch as ms since Unix epoch).
// Absolute time = __sim_epoch_ms__ + __sim_time__
// ============================================================================

/// Convert simulation-relative ms to a chrono DateTime<Utc>
fn sim_time_to_datetime(sim_time_ms: i64, epoch_ms: i64) -> DateTime<Utc> {
    let abs_ms = epoch_ms + sim_time_ms;
    DateTime::<Utc>::from_timestamp_millis(abs_ms)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap())
}

/// hour_of_day(t) -> 0-23
fn rhai_hour_of_day(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.hour() as i64
}

/// minute_of_hour(t) -> 0-59
fn rhai_minute_of_hour(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.minute() as i64
}

/// second_of_minute(t) -> 0-59
fn rhai_second_of_minute(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.second() as i64
}

/// day_of_week(t) -> 0=Sunday, 1=Monday, ..., 6=Saturday
fn rhai_day_of_week(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    match dt.weekday() {
        Weekday::Sun => 0,
        Weekday::Mon => 1,
        Weekday::Tue => 2,
        Weekday::Wed => 3,
        Weekday::Thu => 4,
        Weekday::Fri => 5,
        Weekday::Sat => 6,
    }
}

/// day_of_month(t) -> 1-31
fn rhai_day_of_month(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.day() as i64
}

/// month(t) -> 1-12
fn rhai_month(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.month() as i64
}

/// year(t) -> e.g. 2026
fn rhai_year(sim_time: i64, epoch_ms: i64) -> i64 {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    dt.year() as i64
}

/// is_weekend(t) -> true if Saturday or Sunday
fn rhai_is_weekend(sim_time: i64, epoch_ms: i64) -> bool {
    let dt = sim_time_to_datetime(sim_time, epoch_ms);
    matches!(dt.weekday(), Weekday::Sat | Weekday::Sun)
}

/// is_workday(t) -> true if Monday through Friday
fn rhai_is_workday(sim_time: i64, epoch_ms: i64) -> bool {
    !rhai_is_weekend(sim_time, epoch_ms)
}

/// next_weekday_at(dow, h, m) -> absolute sim time (ms) of the next occurrence
/// dow: 0=Sunday, 1=Monday, ..., 6=Saturday
/// h: hour (0-23), m: minute (0-59)
/// If the current time is before the target time on the same weekday, returns today.
/// Otherwise returns next week's occurrence.
fn rhai_next_weekday_at(sim_time: i64, epoch_ms: i64, dow: i64, h: i64, m: i64) -> i64 {
    let now = sim_time_to_datetime(sim_time, epoch_ms);
    let _target_weekday = match dow {
        0 => Weekday::Sun,
        1 => Weekday::Mon,
        2 => Weekday::Tue,
        3 => Weekday::Wed,
        4 => Weekday::Thu,
        5 => Weekday::Fri,
        6 => Weekday::Sat,
        _ => Weekday::Mon, // default
    };
    
    // Calculate days until target weekday
    let current_dow = now.weekday().num_days_from_sunday() as i64;
    let target_dow = dow;
    let mut days_ahead = target_dow - current_dow;
    if days_ahead < 0 {
        days_ahead += 7;
    }
    
    // Build candidate datetime
    let candidate = now.date_naive() + Duration::days(days_ahead);
    let candidate_dt = candidate.and_hms_opt(h as u32, m as u32, 0)
        .unwrap_or(candidate.and_hms_opt(0, 0, 0).unwrap());
    let candidate_utc = DateTime::<Utc>::from_naive_utc_and_offset(candidate_dt, Utc);
    
    // If candidate is in the past or at current time, go to next week
    let result = if candidate_utc.timestamp_millis() <= now.timestamp_millis() {
        let next_week = candidate + Duration::days(7);
        let next_dt = next_week.and_hms_opt(h as u32, m as u32, 0)
            .unwrap_or(next_week.and_hms_opt(0, 0, 0).unwrap());
        DateTime::<Utc>::from_naive_utc_and_offset(next_dt, Utc)
    } else {
        candidate_utc
    };
    
    // Return as simulation-relative time
    result.timestamp_millis() - epoch_ms
}

/// next_weekday(dow) -> next occurrence at midnight
fn rhai_next_weekday(sim_time: i64, epoch_ms: i64, dow: i64) -> i64 {
    rhai_next_weekday_at(sim_time, epoch_ms, dow, 0, 0)
}

/// next_hour(h) -> sim time (ms) of the next occurrence of hour h:00
fn rhai_next_hour(sim_time: i64, epoch_ms: i64, h: i64) -> i64 {
    let now = sim_time_to_datetime(sim_time, epoch_ms);
    let today = now.date_naive();
    let candidate = today.and_hms_opt(h as u32, 0, 0)
        .unwrap_or(today.and_hms_opt(0, 0, 0).unwrap());
    let candidate_utc = DateTime::<Utc>::from_naive_utc_and_offset(candidate, Utc);
    
    let result = if candidate_utc.timestamp_millis() <= now.timestamp_millis() {
        // Already past this hour today, use tomorrow
        let tomorrow = today + Duration::days(1);
        let dt = tomorrow.and_hms_opt(h as u32, 0, 0)
            .unwrap_or(tomorrow.and_hms_opt(0, 0, 0).unwrap());
        DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
    } else {
        candidate_utc
    };
    
    result.timestamp_millis() - epoch_ms
}

/// Register calendar functions with Rhai engine.
/// All user-facing functions are registered as native Rust functions that read
/// current simulation time and epoch from thread-local storage (SIM_CURRENT_TIME
/// and SIM_EPOCH_MS). This avoids the Rhai limitation where script-defined
/// functions cannot access scope variables.
fn register_calendar_module(engine: &mut Engine) {
    // Helper to get current sim time and epoch from thread-local
    fn get_sim_ctx() -> (i64, i64) {
        let t = SIM_CURRENT_TIME.with(|c| c.get());
        let e = SIM_EPOCH_MS.with(|c| c.get());
        (t, e)
    }

    // current_time() -> current simulation time in ms
    engine.register_fn("current_time", || -> i64 {
        SIM_CURRENT_TIME.with(|c| c.get())
    });

    // Calendar decomposition: take a sim_time parameter
    engine.register_fn("hour_of_day", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_hour_of_day(t, e)
    });
    engine.register_fn("minute_of_hour", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_minute_of_hour(t, e)
    });
    engine.register_fn("second_of_minute", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_second_of_minute(t, e)
    });
    engine.register_fn("day_of_week", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_day_of_week(t, e)
    });
    engine.register_fn("day_of_month", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_day_of_month(t, e)
    });
    engine.register_fn("month", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_month(t, e)
    });
    engine.register_fn("year", |t: i64| -> i64 {
        let (_, e) = get_sim_ctx(); rhai_year(t, e)
    });
    engine.register_fn("is_weekend", |t: i64| -> bool {
        let (_, e) = get_sim_ctx(); rhai_is_weekend(t, e)
    });
    engine.register_fn("is_workday", |t: i64| -> bool {
        let (_, e) = get_sim_ctx(); rhai_is_workday(t, e)
    });

    // Scheduling: next_weekday_at(dow, h, m), etc. - use current sim time implicitly
    engine.register_fn("next_weekday_at", |dow: i64, h: i64, m: i64| -> i64 {
        let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, dow, h, m)
    });
    engine.register_fn("next_weekday", |dow: i64| -> i64 {
        let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, dow)
    });
    engine.register_fn("next_hour", |h: i64| -> i64 {
        let (t, e) = get_sim_ctx(); rhai_next_hour(t, e, h)
    });

    // Convenience: next_monday() .. next_sunday()
    engine.register_fn("next_monday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 1) });
    engine.register_fn("next_tuesday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 2) });
    engine.register_fn("next_wednesday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 3) });
    engine.register_fn("next_thursday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 4) });
    engine.register_fn("next_friday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 5) });
    engine.register_fn("next_saturday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 6) });
    engine.register_fn("next_sunday", || -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday(t, e, 0) });

    // Convenience: next_monday_at(h, m) .. next_sunday_at(h, m)
    engine.register_fn("next_monday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 1, h, m) });
    engine.register_fn("next_tuesday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 2, h, m) });
    engine.register_fn("next_wednesday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 3, h, m) });
    engine.register_fn("next_thursday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 4, h, m) });
    engine.register_fn("next_friday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 5, h, m) });
    engine.register_fn("next_saturday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 6, h, m) });
    engine.register_fn("next_sunday_at", |h: i64, m: i64| -> i64 { let (t, e) = get_sim_ctx(); rhai_next_weekday_at(t, e, 0, h, m) });

    // time_until(target_time) -> ms until target_time from current sim time
    engine.register_fn("time_until", |target: i64| -> i64 {
        let t = SIM_CURRENT_TIME.with(|c| c.get());
        target - t
    });
}

// ============================================================================
// Rhai random distribution module - provides CPN Tools compatible distributions
// See: https://cpntools.org/2018/01/18/random-distribution-functions/
// ============================================================================

/// Bernoulli distribution: returns 1 with probability p, 0 with probability 1-p
/// Raises exception if p < 0.0 or p > 1.0
fn dist_bernoulli(p: f64) -> Result<i64, Box<EvalAltResult>> {
    if p < 0.0 || p > 1.0 {
        return Err(format!("Bernoulli: p must be in [0, 1], got {}", p).into());
    }
    let dist = Bernoulli::new(p).map_err(|e| format!("Bernoulli error: {}", e))?;
    let mut rng = rand::rng();
    Ok(if dist.sample(&mut rng) { 1 } else { 0 })
}

/// Beta distribution with shape parameters a and b
/// Raises exception if a <= 0.0 or b <= 0.0
fn dist_beta(a: f64, b: f64) -> Result<f64, Box<EvalAltResult>> {
    if a <= 0.0 || b <= 0.0 {
        return Err(format!("Beta: a and b must be > 0, got a={}, b={}", a, b).into());
    }
    let dist = Beta::new(a, b).map_err(|e| format!("Beta error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Binomial distribution: number of successes in n trials with probability p
/// Raises exception if n < 1 or p < 0.0 or p > 1.0
fn dist_binomial(n: i64, p: f64) -> Result<i64, Box<EvalAltResult>> {
    if n < 1 {
        return Err(format!("Binomial: n must be >= 1, got {}", n).into());
    }
    if p < 0.0 || p > 1.0 {
        return Err(format!("Binomial: p must be in [0, 1], got {}", p).into());
    }
    let dist = Binomial::new(n as u64, p).map_err(|e| format!("Binomial error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng) as i64)
}

/// Chi-squared distribution with n degrees of freedom
/// Raises exception if n < 1
fn dist_chisq(n: i64) -> Result<f64, Box<EvalAltResult>> {
    if n < 1 {
        return Err(format!("Chisq: n must be >= 1, got {}", n).into());
    }
    let dist = ChiSquared::new(n as f64).map_err(|e| format!("Chisq error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Discrete uniform distribution: returns integer in [a, b]
/// Raises exception if a > b
fn dist_discrete(a: i64, b: i64) -> Result<i64, Box<EvalAltResult>> {
    if a > b {
        return Err(format!("Discrete: a must be <= b, got a={}, b={}", a, b).into());
    }
    let mut rng = rand::rng();
    Ok(rng.random_range(a..=b))
}

/// Erlang distribution with shape n and rate r
/// This is a Gamma distribution with integer shape parameter
/// Raises exception if n < 1 or r <= 0.0
fn dist_erlang(n: i64, r: f64) -> Result<f64, Box<EvalAltResult>> {
    if n < 1 {
        return Err(format!("Erlang: n must be >= 1, got {}", n).into());
    }
    if r <= 0.0 {
        return Err(format!("Erlang: r must be > 0, got {}", r).into());
    }
    // Erlang(n, r) = Gamma(n, 1/r) where r is the rate
    let dist = Gamma::new(n as f64, 1.0 / r).map_err(|e| format!("Erlang error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Exponential distribution with rate r (mean = 1/r)
/// Raises exception if r <= 0.0
fn dist_exponential(r: f64) -> Result<f64, Box<EvalAltResult>> {
    if r <= 0.0 {
        return Err(format!("Exponential: r must be > 0, got {}", r).into());
    }
    let dist = Exp::new(r).map_err(|e| format!("Exponential error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Gamma distribution with scale l (lambda) and shape k
/// Raises exception if l <= 0.0 or k <= 0.0
fn dist_gamma(l: f64, k: f64) -> Result<f64, Box<EvalAltResult>> {
    if l <= 0.0 || k <= 0.0 {
        return Err(format!("Gamma: l and k must be > 0, got l={}, k={}", l, k).into());
    }
    // rand_distr::Gamma uses (shape, scale) = (k, l)
    let dist = Gamma::new(k, l).map_err(|e| format!("Gamma error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Normal (Gaussian) distribution with mean n and variance v
/// Raises exception if v < 0.0
fn dist_normal(n: f64, v: f64) -> Result<f64, Box<EvalAltResult>> {
    if v < 0.0 {
        return Err(format!("Normal: variance must be >= 0, got {}", v).into());
    }
    if v == 0.0 {
        return Ok(n); // Zero variance means constant value
    }
    // Normal::new takes (mean, std_dev), variance = std_dev^2
    let std_dev = v.sqrt();
    let dist = Normal::new(n, std_dev).map_err(|e| format!("Normal error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Poisson distribution with mean m
/// Raises exception if m <= 0.0
fn dist_poisson(m: f64) -> Result<i64, Box<EvalAltResult>> {
    if m <= 0.0 {
        return Err(format!("Poisson: m must be > 0, got {}", m).into());
    }
    let dist = Poisson::new(m).map_err(|e| format!("Poisson error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng) as i64)
}

/// Rayleigh distribution with scale parameter s
/// Raises exception if s < 0.0
fn dist_rayleigh(s: f64) -> Result<f64, Box<EvalAltResult>> {
    if s < 0.0 {
        return Err(format!("Rayleigh: s must be >= 0, got {}", s).into());
    }
    if s == 0.0 {
        return Ok(0.0); // Zero scale means constant 0
    }
    // Rayleigh is a special case of Weibull with shape=2
    // Or we can use: X = s * sqrt(-2 * ln(U)) where U is uniform(0,1)
    let mut rng = rand::rng();
    let u: f64 = rng.random();
    Ok(s * (-2.0 * u.ln()).sqrt())
}

/// Student's t-distribution with n degrees of freedom
/// Raises exception if n < 1
fn dist_student(n: i64) -> Result<f64, Box<EvalAltResult>> {
    if n < 1 {
        return Err(format!("Student: n must be >= 1, got {}", n).into());
    }
    let dist = StudentT::new(n as f64).map_err(|e| format!("Student error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Continuous uniform distribution on [a, b]
/// Raises exception if a > b
fn dist_uniform(a: f64, b: f64) -> Result<f64, Box<EvalAltResult>> {
    if a > b {
        return Err(format!("Uniform: a must be <= b, got a={}, b={}", a, b).into());
    }
    if a == b {
        return Ok(a);
    }
    let dist = Uniform::new_inclusive(a, b).map_err(|e| format!("Uniform error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Weibull distribution with scale lambda and shape k
/// Raises exception if lambda <= 0.0 or k <= 0.0
fn dist_weibull(lambda: f64, k: f64) -> Result<f64, Box<EvalAltResult>> {
    if lambda <= 0.0 || k <= 0.0 {
        return Err(format!("Weibull: lambda and k must be > 0, got lambda={}, k={}", lambda, k).into());
    }
    let dist = Weibull::new(lambda, k).map_err(|e| format!("Weibull error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng))
}

/// Register all random distribution functions with a Rhai engine
/// Function names match CPN Tools convention
fn register_distribution_module(engine: &mut Engine) {
    engine.register_fn("bernoulli", dist_bernoulli);
    engine.register_fn("beta", dist_beta);
    engine.register_fn("binomial", dist_binomial);
    engine.register_fn("chisq", dist_chisq);
    engine.register_fn("discrete", dist_discrete);
    engine.register_fn("erlang", dist_erlang);
    engine.register_fn("exponential", dist_exponential);
    engine.register_fn("gamma", dist_gamma);
    engine.register_fn("normal", dist_normal);
    engine.register_fn("poisson", dist_poisson);
    engine.register_fn("rayleigh", dist_rayleigh);
    engine.register_fn("student", dist_student);
    engine.register_fn("uniform", dist_uniform);
    engine.register_fn("weibull", dist_weibull);
}

/// Extract the actual value from a token, handling both timed and non-timed formats.
/// For timed tokens (Rhai maps with "value" and "timestamp" fields), returns the "value" field.
/// For non-timed tokens, returns the token as-is.
fn extract_token_value(token: &Dynamic) -> Dynamic {
    // Check if this is a Rhai map (object) with a "value" field
    if token.is_map() {
        if let Some(map) = token.read_lock::<rhai::Map>() {
            if let Some(value) = map.get("value") {
                return value.clone();
            }
        }
    }
    // Not a timed token map, return as-is
    token.clone()
}

/// Extract the timestamp from a timed token (Rhai map).
/// Returns 0 if not a timed token or if timestamp field is missing.
fn extract_token_timestamp(token: &Dynamic) -> i64 {
    if token.is_map() {
        if let Some(map) = token.read_lock::<rhai::Map>() {
            if let Some(ts) = map.get("timestamp") {
                return ts.as_int().unwrap_or(0);
            }
        }
    }
    0
}

/// Create a timed token (Rhai map with "value" and "timestamp" fields)
fn create_timed_token(value: Dynamic, timestamp: i64) -> Dynamic {
    let mut map = rhai::Map::new();
    map.insert("value".into(), value);
    map.insert("timestamp".into(), Dynamic::from(timestamp));
    Dynamic::from(map)
}

impl Simulator {
    /// Update simulation time context for Rhai expression evaluation.
    /// Sets both thread-local state (for native calendar functions) and
    /// scope variable (for direct access in expressions).
    /// Must be called before any guard/arc/time expression evaluation.
    fn update_scope_time(&mut self) {
        SIM_CURRENT_TIME.with(|c| c.set(self.current_time));
        SIM_EPOCH_MS.with(|c| c.set(self.epoch_ms));
    }

    pub fn new(model_data: PetriNetData) -> Result<Self, String> {
        let mut engine = Engine::new();
        let mut scope = Scope::new();

        engine.register_fn("multiset_equal", rhai_multiset_equal_wrapper);
        
        // Register delay module for timed simulations
        register_delay_module(&mut engine);
        
        // Register random distribution functions (CPN Tools compatible)
        register_distribution_module(&mut engine);

        let mut guards = HashMap::new();
        let mut arc_expressions = HashMap::new();
        let mut initial_marking_expressions = HashMap::new();
        let mut time_expressions: HashMap<String, AST> = HashMap::new();

        let mut current_marking: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut token_timestamps: HashMap<String, Vec<i64>> = HashMap::new();
        let mut declared_variables: HashMap<String, String> = HashMap::new();
        let mut parsed_color_sets: HashMap<String, ParsedColorSet> = HashMap::new();
        let mut timed_colorsets: HashSet<String> = HashSet::new();

        for var in &model_data.variables {
            declared_variables.insert(var.name.clone(), var.color_set.clone());
        }

        // First pass: identify timed color sets
        for cs in &model_data.color_sets {
            if cs.timed {
                timed_colorsets.insert(cs.name.clone());
            }
        }

        for cs in &model_data.color_sets {
            let name = cs.name.clone();
            let definition = cs.definition.trim();
            let mut parsed_cs = ParsedColorSet::unknown();

            if definition.ends_with("= unit;") {
                parsed_cs.kind = ColorSetKind::Unit;
            } else if definition.ends_with("= bool;") {
                parsed_cs.kind = ColorSetKind::Bool;
            } else if definition.ends_with("= int;") {
                parsed_cs.kind = ColorSetKind::Int;
            } else if definition.ends_with("= intinf;") {
                parsed_cs.kind = ColorSetKind::IntInf;
            } else if definition.ends_with("= time;") {
                parsed_cs.kind = ColorSetKind::Time;
            } else if definition.ends_with("= real;") {
                parsed_cs.kind = ColorSetKind::Real;
            } else if definition.ends_with("= string;") {
                parsed_cs.kind = ColorSetKind::String;
            } else if definition.contains("= int with") && definition.contains("..") && definition.ends_with(";") {
                if let Some(range_part) = definition.split(" with ").nth(1) {
                    if let Some(range_str) = range_part.split(';').next() {
                        let bounds: Vec<&str> = range_str.split("..").collect();
                        if bounds.len() == 2 {
                            if let (Ok(start), Ok(end)) = (bounds[0].trim().parse::<i64>(), bounds[1].trim().parse::<i64>()) {
                                parsed_cs.kind = ColorSetKind::IntRange;
                                parsed_cs.range = Some((start, end));
                                parsed_cs.base_type_name = Some("INT".to_string());
                            } else {
                                eprintln!("Warning: Could not parse range bounds in definition: {}", definition);
                            }
                        } else {
                            eprintln!("Warning: Could not parse range format in definition: {}", definition);
                        }
                    }
                }
            } else if definition.starts_with("colset ") && definition.contains("= list ") && definition.ends_with(";") {
                if let Some(list_part) = definition.split("= list ").nth(1) {
                    if let Some(element_type) = list_part.split(';').next() {
                        parsed_cs.kind = ColorSetKind::List;
                        parsed_cs.base_type_name = Some(element_type.trim().to_string());
                    } else {
                        eprintln!("Warning: Could not parse list element type in definition: {}", definition);
                    }
                }
            } else if definition.starts_with("colset ") && definition.contains("= product ") && definition.ends_with(";") {
                // Parse product types like "colset INTxDATA = product INT * DATA;"
                if let Some(product_part) = definition.split("= product ").nth(1) {
                    if let Some(types_str) = product_part.split(';').next() {
                        let component_types: Vec<String> = types_str
                            .split('*')
                            .map(|s| s.trim().to_string())
                            .collect();
                        if !component_types.is_empty() {
                            parsed_cs.kind = ColorSetKind::Product;
                            parsed_cs.component_types = Some(component_types);
                        } else {
                            eprintln!("Warning: Could not parse product component types in definition: {}", definition);
                        }
                    }
                }
            } else if definition.starts_with("colset ") && definition.contains("= record ") && definition.ends_with(";") {
                // Parse record types like "colset PERSON = record name: STRING * age: INT;"
                if let Some(record_part) = definition.split("= record ").nth(1) {
                    if let Some(fields_str) = record_part.split(';').next() {
                        let fields: Vec<(String, String)> = fields_str
                            .split('*')
                            .filter_map(|field| {
                                let parts: Vec<&str> = field.split(':').map(|s| s.trim()).collect();
                                if parts.len() == 2 {
                                    Some((parts[0].to_string(), parts[1].to_string()))
                                } else {
                                    eprintln!("Warning: Could not parse record field: {}", field);
                                    None
                                }
                            })
                            .collect();
                        if !fields.is_empty() {
                            parsed_cs.kind = ColorSetKind::Record;
                            parsed_cs.record_fields = Some(fields);
                        } else {
                            eprintln!("Warning: Could not parse record fields in definition: {}", definition);
                        }
                    }
                }
            } else {
                eprintln!("Warning: Unrecognized colorset definition format for '{}': {}", name, definition);
            }

            if parsed_cs.kind != ColorSetKind::Unknown {
                parsed_color_sets.insert(name, parsed_cs);
            } else {
                if !definition.starts_with("colset ") {
                    eprintln!("Warning: Failed to parse colorset definition for '{}': {}", name, definition);
                }
            }
        }

        let mut all_fn_code = String::new();
        for func in &model_data.functions {
            all_fn_code.push_str(&func.code);
            all_fn_code.push('\n');
        }

        if !all_fn_code.is_empty() {
            match engine.compile(&all_fn_code) {
                Ok(ast) => {
                    let fn_scope = Scope::new();
                    match Module::eval_ast_as_new(fn_scope, &ast, &engine) {
                        Ok(compiled_module) => {
                            engine.register_global_module(compiled_module.into());
                        }
                        Err(e) => {
                            let err_msg = format!("Error evaluating function AST into new module: {}", e);
                            eprintln!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error compiling combined functions: {}", e);
                    eprintln!("{}", err_msg);
                    return Err(err_msg);
                }
            }
        }

        // Process "uses" (constant/value definitions) and add them to the scope
        for use_def in &model_data.uses {
            let content = use_def.content.trim();
            if content.is_empty() {
                continue;
            }
            
            // Parse SML-style val definitions like: val stop = "########";
            // and convert them to Rhai constant definitions
            if content.starts_with("val ") && content.contains('=') {
                // Parse: val name = value;
                if let Some(rest) = content.strip_prefix("val ") {
                    let parts: Vec<&str> = rest.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        let var_name = parts[0].trim();
                        let mut value_str = parts[1].trim();
                        // Remove trailing semicolon if present
                        if value_str.ends_with(';') {
                            value_str = value_str[..value_str.len()-1].trim();
                        }
                        
                        // Evaluate the value expression
                        match engine.eval_expression::<Dynamic>(value_str) {
                            Ok(value) => {
                                scope.push_constant(var_name, value);
                            }
                            Err(e) => {
                                eprintln!("Warning: Could not evaluate use definition '{}': {}", use_def.name, e);
                            }
                        }
                    }
                }
            } else {
                // Try to evaluate as Rhai code directly (for let/const statements)
                // This won't work for Module compilation, so we try expression evaluation
                eprintln!("Warning: Unrecognized use definition format for '{}': {}", use_def.name, content);
            }
        }

        // Register calendar functions as native Rust functions.
        // These read sim time/epoch from thread-local state (set by update_scope_time).
        register_calendar_module(&mut engine);

        for net in &model_data.petri_nets {
            for place in &net.places {
                if !place.initial_marking.is_empty() && place.initial_marking != "[]" {
                    // Convert JSON object notation to Rhai map syntax if needed
                    let marking_expr = if place.initial_marking.trim_start().starts_with('[') && 
                                           place.initial_marking.contains('{') {
                        // Looks like JSON with objects - convert to Rhai
                        match convert_json_to_rhai(&place.initial_marking) {
                            Ok(rhai_expr) => rhai_expr,
                            Err(e) => {
                                eprintln!("Warning: Failed to convert JSON marking for place {}: {}. Trying original.", place.name, e);
                                place.initial_marking.clone()
                            }
                        }
                    } else {
                        place.initial_marking.clone()
                    };
                    
                    match engine.compile_expression(&marking_expr) {
                        Ok(ast) => {
                            initial_marking_expressions.insert(place.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling initial marking for place {}: {}", place.name, e);
                            eprintln!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
                current_marking.insert(place.id.clone(), vec![]);
            }
            for transition in &net.transitions {
                if !transition.guard.is_empty() && transition.guard.to_lowercase() != "true" {
                    match engine.compile(&transition.guard) {
                        Ok(ast) => {
                            guards.insert(transition.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling guard for transition {}: {}", transition.name, e);
                            eprintln!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
                // Compile transition time expression if present
                // Time expressions can have "@+" prefix (CPN Tools format) - strip it
                let time_expr = transition.time.trim();
                let time_expr = if time_expr.starts_with("@+") {
                    time_expr[2..].trim()
                } else if time_expr.starts_with("@") {
                    time_expr[1..].trim()
                } else {
                    time_expr
                };
                if !time_expr.is_empty() {
                    match engine.compile_expression(time_expr) {
                        Ok(ast) => {
                            time_expressions.insert(transition.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling time expression for transition {}: {}", transition.name, e);
                            eprintln!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
            }
            for arc in &net.arcs {
                if !arc.inscription.is_empty() {
                    match engine.compile_expression(&arc.inscription) {
                        Ok(ast) => {
                            arc_expressions.insert(arc.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling inscription for arc {}: {}", arc.id, e);
                            eprintln!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
            }
        }

        for (place_id, ast) in &initial_marking_expressions {
            match engine.eval_ast_with_scope::<Dynamic>(&mut scope, ast) {
                Ok(result) => {
                    let tokens: Vec<Dynamic> = if let Ok(token_values) = result.clone().into_typed_array::<Dynamic>() {
                        token_values
                    } else if result.is_unit() {
                        // A single unit value () represents one unit token
                        vec![result]
                    } else {
                        vec![result]
                    };
                    // Initialize timestamps to 0 (immediately available)
                    let timestamps: Vec<i64> = vec![0; tokens.len()];
                    current_marking.insert(place_id.clone(), tokens);
                    token_timestamps.insert(place_id.clone(), timestamps);
                }
                Err(e) => {
                    let err_msg = format!("Error evaluating initial marking for place {}: {}", place_id, e);
                    eprintln!("{}", err_msg);
                    return Err(err_msg);
                }
            }
        }

        // Build timed_places map by checking each place's colorset
        let mut timed_places: HashMap<String, bool> = HashMap::new();
        for net in &model_data.petri_nets {
            for place in &net.places {
                let is_timed = timed_colorsets.contains(&place.color_set);
                timed_places.insert(place.id.clone(), is_timed);
                // Initialize token_timestamps for each place if not already done
                token_timestamps.entry(place.id.clone()).or_insert_with(Vec::new);
            }
        }

        // Get simulation epoch from model
        let simulation_epoch = model_data.simulation_epoch.clone();
        
        // Parse epoch string to milliseconds since Unix epoch
        // Supports both RFC 3339 ("2026-02-05T23:55:47.356Z") and
        // naive ISO 8601 without timezone ("2026-02-05T23:55:47.356"), treated as UTC
        let epoch_ms: i64 = simulation_epoch.as_ref()
            .and_then(|s| {
                // First try RFC 3339 (with timezone)
                DateTime::parse_from_rfc3339(s).ok()
                    .map(|dt| dt.timestamp_millis())
                    .or_else(|| {
                        // Try appending Z for naive ISO 8601 strings
                        let with_z = format!("{}Z", s);
                        DateTime::parse_from_rfc3339(&with_z).ok()
                            .map(|dt| dt.timestamp_millis())
                    })
                    .or_else(|| {
                        // Try parsing as NaiveDateTime directly
                        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok()
                            .map(|ndt| ndt.and_utc().timestamp_millis())
                    })
            })
            .unwrap_or(0);
        
        // Initialize thread-local simulation context for calendar functions
        SIM_CURRENT_TIME.with(|c| c.set(0));
        SIM_EPOCH_MS.with(|c| c.set(epoch_ms));

        Ok(Simulator {
            model: model_data,
            current_marking,
            token_timestamps,
            current_time: 0,
            simulation_epoch,
            epoch_ms,
            timed_places,
            rhai_engine: engine,
            rhai_scope: scope,
            guards,
            arc_expressions,
            initial_marking_expressions,
            time_expressions,
            declared_variables,
            parsed_color_sets,
        })
    }

    pub fn run_step(&mut self) -> Option<FiringEventData> {
        let mut enabled_bindings = self.find_enabled_bindings();
        
        // If no bindings found, keep advancing time until we find enabled transitions or run out of future tokens
        while enabled_bindings.is_empty() {
            if let Some(earliest_future_time) = self.find_earliest_future_token_time() {
                // Advance simulation time to when the next token becomes available
                #[cfg(target_arch = "wasm32")]
                {
                    web_sys::console::log_1(&format!(
                        "[WASM] No transition enabled at time {}, advancing to {}",
                        self.current_time, earliest_future_time
                    ).into());
                }
                self.current_time = earliest_future_time;
                
                // Try finding bindings again at the new time
                enabled_bindings = self.find_enabled_bindings();
            } else {
                // No more future tokens, simulation is truly done
                break;
            }
        }
        
        if enabled_bindings.is_empty() {
            return None;
        }

        let (selected_transition_id, selected_binding) = self.select_transition_and_binding(&enabled_bindings);

        let transition_to_fire = self.model.find_transition(&selected_transition_id)
            .expect("Selected transition ID not found in model - internal error");

        let mut consumed_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut produced_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();

        // tokens_to_consume contains Dynamic values
        for (place_id, tokens_to_consume) in &selected_binding.consumed_tokens_map {
            if let Some(available_tokens) = self.current_marking.get_mut(place_id) {
                let mut consumed_for_event: Vec<Dynamic> = Vec::new();
                let mut remaining_tokens: Vec<Dynamic> = Vec::new();
                let mut needed_counts: HashMap<String, usize> = HashMap::new();
                for token in tokens_to_consume {
                    *needed_counts.entry(token.to_string()).or_insert(0) += 1;
                }

                let mut current_counts: HashMap<String, usize> = HashMap::new();

                for token in available_tokens.iter() {
                    let token_str = token.to_string();
                    let needed_count = needed_counts.get(&token_str).copied().unwrap_or(0);
                    let current_count = current_counts.entry(token_str).or_insert(0);

                    if *current_count < needed_count {
                        consumed_for_event.push(token.clone());
                        *current_count += 1;
                    } else {
                        remaining_tokens.push(token.clone());
                    }
                }

                let mut consumed_event_counts: HashMap<String, usize> = HashMap::new();
                for token in &consumed_for_event {
                    *consumed_event_counts.entry(token.to_string()).or_insert(0) += 1;
                }

                if consumed_event_counts == needed_counts {
                    *available_tokens = remaining_tokens;
                    consumed_tokens_event.insert(place_id.clone(), consumed_for_event);
                } else {
                    eprintln!(
                        "Internal Error: Mismatch during consumption for transition {}. Place: {}. Expected counts: {:?}, Actual consumed counts: {:?}. Binding: {:?}. Available before: {:?}. Available after attempt: {:?}.",
                        transition_to_fire.name,
                        place_id,
                        needed_counts,
                        consumed_event_counts,
                        selected_binding.variables,
                        self.current_marking.get(place_id),
                        remaining_tokens
                    );
                    println!("Current marking state: {:?}", self.current_marking);
                    panic!("Consumption failed unexpectedly due to count mismatch!");
                }
            } else {
                eprintln!("Internal Error: Place {} (expected for consumption) not found in marking map.", place_id);
                panic!("Consumption failed unexpectedly!");
            }
        }

        let mut firing_scope = self.rhai_scope.clone();
        for (var, value) in &selected_binding.variables {
            firing_scope.push_constant(var, value.clone());
        }

        // Collect all output arc inscriptions to find unbound range variables
        // and bind them to random values within their range
        let mut rng = rand::rng();
        let mut already_bound_random: std::collections::HashSet<String> = std::collections::HashSet::new();
        if let Some(net) = self.model.petri_nets.first() {
            for arc in &net.arcs {
                // Check if this is an output arc (source is transition, or bidirectional with target as transition)
                let is_output_arc = arc.source == selected_transition_id || 
                    (arc.is_bidirectional && arc.target == selected_transition_id);
                
                if is_output_arc {
                    // Find unbound variables used in output arcs that have int range colorsets
                    for (var_name, colorset_name) in &self.declared_variables {
                        // Skip if already bound from input arcs
                        if selected_binding.variables.contains_key(var_name) {
                            continue;
                        }
                        // Skip if already bound randomly in this loop
                        if already_bound_random.contains(var_name) {
                            continue;
                        }
                        // Check if the variable is used in this arc inscription (as a whole word, not substring)
                        // Use regex-like word boundary check: variable must be preceded and followed by non-identifier chars
                        if !is_variable_in_inscription(var_name, &arc.inscription) {
                            continue;
                        }
                        // Check if this variable has an int range colorset
                        if let Some(parsed_cs) = self.parsed_color_sets.get(colorset_name) {
                            if parsed_cs.kind == ColorSetKind::IntRange {
                                if let Some((start, end)) = parsed_cs.range {
                                    // Generate random value within range
                                    let random_value = rng.random_range(start..=end);
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        // Log to browser console in WASM
                                        web_sys::console::log_1(&format!("[WASM] Binding random variable {} to {} (from colorset {} with range {}..{})", var_name, random_value, colorset_name, start, end).into());
                                    }
                                    firing_scope.push_constant(var_name, Dynamic::from(random_value));
                                    already_bound_random.insert(var_name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Evaluate transition time expression to get delay for produced tokens
        let time_delay: i64 = if let Some(time_ast) = self.time_expressions.get(&selected_transition_id) {
            match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, time_ast) {
                Ok(result) => {
                    let delay = result.as_int().unwrap_or(0);
                    #[cfg(target_arch = "wasm32")]
                    {
                        web_sys::console::log_1(&format!(
                            "[WASM] Transition {} time delay: {} ms",
                            transition_to_fire.name, delay
                        ).into());
                    }
                    delay
                }
                Err(e) => {
                    eprintln!("Error evaluating time expression for transition {}: {}", transition_to_fire.name, e);
                    0
                }
            }
        } else {
            0
        };

        // Calculate the timestamp for produced tokens (current_time + time_delay)
        let produced_token_time = self.current_time + time_delay;

        if let Some(net) = self.model.petri_nets.first() {
            for arc in &net.arcs {
                // Output arcs: source is transition, OR bidirectional with target as transition
                let is_output_arc = arc.source == selected_transition_id || 
                    (arc.is_bidirectional && arc.target == selected_transition_id);
                
                if is_output_arc {
                    // For bidirectional arcs where target is the transition, the place is the source
                    let place_id = if arc.is_bidirectional && arc.target == selected_transition_id {
                        &arc.source
                    } else {
                        &arc.target
                    };
                    if let Some(ast) = self.arc_expressions.get(&arc.id) {
                        eprintln!("[DEBUG] Evaluating output arc {} to place {}: inscription = '{}'", arc.id, place_id, arc.inscription);
                        eprintln!("[DEBUG] Firing scope variables: {:?}", firing_scope.iter().map(|(n, _, v)| format!("{}={}", n, v)).collect::<Vec<_>>());
                        match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, ast) {
                            Ok(produced_tokens_dynamic) => {
                                eprintln!("[DEBUG] Output arc {} produced: {:?}", arc.id, produced_tokens_dynamic);
                                // Check if the target place has a product colorset
                                let is_product_place = self.is_place_product_type(place_id);
                                
                                let tokens_to_add = if is_product_place {
                                    // For product places, check if the result is an array representing a single product token
                                    // or an array of product tokens (e.g., from a conditional that returns multiple)
                                    if let Ok(arr) = produced_tokens_dynamic.clone().into_typed_array::<Dynamic>() {
                                        // Check if this looks like a product value (array with mixed types)
                                        // vs an array of product values (array of arrays)
                                        if !arr.is_empty() {
                                            // If first element is also an array, it's likely an array of products
                                            if arr[0].clone().into_typed_array::<Dynamic>().is_ok() {
                                                // Array of product tokens
                                                arr
                                            } else {
                                                // Single product token - wrap it
                                                vec![produced_tokens_dynamic]
                                            }
                                        } else {
                                            // Empty array means no tokens
                                            vec![]
                                        }
                                    } else if produced_tokens_dynamic.is_unit() {
                                        // Unit value represents one unit token
                                        vec![produced_tokens_dynamic]
                                    } else {
                                        vec![produced_tokens_dynamic]
                                    }
                                } else {
                                    // For non-product places, use the original logic
                                    Self::dynamic_to_vec_dynamic(produced_tokens_dynamic)
                                };
                                
                                if !tokens_to_add.is_empty() {
                                    // Check if target place is timed
                                    let is_timed = self.timed_places.get(place_id).copied().unwrap_or(false);
                                    
                                    // For timed places, wrap tokens as timed token objects
                                    // Use produced_token_time which includes the transition's time delay
                                    let tokens_for_marking: Vec<Dynamic> = if is_timed {
                                        tokens_to_add.iter().map(|t| {
                                            // If it's already a timed token, preserve it; otherwise create one
                                            if t.is_map() {
                                                if let Some(map) = t.read_lock::<rhai::Map>() {
                                                    if map.contains_key("value") && map.contains_key("timestamp") {
                                                        return t.clone();
                                                    }
                                                }
                                            }
                                            create_timed_token(t.clone(), produced_token_time)
                                        }).collect()
                                    } else {
                                        tokens_to_add.clone()
                                    };
                                    
                                    // Store timed tokens in both event and marking
                                    produced_tokens_event.entry(place_id.clone()).or_default().extend(tokens_for_marking.clone());
                                    self.current_marking.entry(place_id.clone()).or_default().extend(tokens_for_marking.clone());
                                    
                                    // Also update parallel timestamp array for consistency
                                    let timestamps = self.token_timestamps.entry(place_id.clone()).or_default();
                                    for _ in 0..tokens_to_add.len() {
                                        timestamps.push(produced_token_time);
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("  Error evaluating output arc {} inscription: {}", arc.id, e);
                                return None;
                            }
                        }
                    } else if !arc.inscription.is_empty() {
                        eprintln!("  Internal Error: Compiled AST not found for non-empty output arc {}", arc.id);
                        return None;
                    }
                }
            }
        } else {
            return None;
        }

        let event_data = FiringEventData {
            transition_id: selected_transition_id,
            transition_name: transition_to_fire.name.clone(),
            simulation_time: self.current_time,
            consumed: consumed_tokens_event,
            produced: produced_tokens_event,
        };

        Some(event_data)
    }

    #[allow(dead_code)]
    pub fn run(&mut self) {
        println!("Starting simulation run...");
        println!("Initial Marking: {:?}", self.current_marking);
        let mut step = 0;
        while let Some(event_data) = self.run_step() {
            step += 1;
            println!("--- Step {} ---", step);
            println!("  Fired: {} ({})", event_data.transition_name, event_data.transition_id);
            println!("  Consumed: {:?}", event_data.consumed);
            println!("  Produced: {:?}", event_data.produced);
            println!("Current Marking: {:?}", self.current_marking);
            if step > 100 {
                println!("Stopping run after 100 steps.");
                break;
            }
        }
        println!("--- Simulation run finished ---");
    }

    pub fn dynamic_to_vec_dynamic(d: Dynamic) -> Vec<Dynamic> {
        if let Ok(arr) = d.clone().into_typed_array::<Dynamic>() {
            arr
        } else {
            // Single value (including unit) - wrap in vec as one token
            vec![d]
        }
    }

    fn is_list_variable(&self, var_name: &str) -> bool {
        self.declared_variables.get(var_name)
            .and_then(|colorset_name| self.parsed_color_sets.get(colorset_name))
            .map_or(false, |cs| cs.kind == ColorSetKind::List)
    }

    fn get_available_token_indices_with_place<'a>(
        place_id: &'a str,
        all_tokens_in_place: &'a [Dynamic],
        token_timestamps: &'a [i64],
        consumed_in_binding: Option<&Vec<Dynamic>>,
        current_time: i64,
    ) -> Vec<(&'a str, usize, Dynamic)> {
        let consumed_counts = consumed_in_binding
            .map(|tokens| create_count_map(tokens))
            .unwrap_or_default();

        let mut available_tokens_info = Vec::new();
        let mut current_token_counts_in_place = HashMap::new();
        for (index, token) in all_tokens_in_place.iter().enumerate() {
            // Check for timestamp - first try parallel array, then check embedded timestamp in token
            let timestamp = if let Some(&ts) = token_timestamps.get(index) {
                if ts != 0 { ts } else { extract_token_timestamp(token) }
            } else {
                extract_token_timestamp(token)
            };
            
            // Only consider tokens that are ready (timestamp <= current_time)
            if timestamp > current_time {
                continue;
            }
            let token_str = token.to_string();
            let consumed_count = consumed_counts.get(&token_str).copied().unwrap_or(0);
            let current_count = current_token_counts_in_place.entry(token_str.clone()).or_insert(0);

            if *current_count < consumed_count {
                *current_count += 1;
            } else {
                available_tokens_info.push((place_id, index, token.clone()));
            }
        }
        available_tokens_info
    }

    fn find_enabled_bindings(&mut self) -> Vec<(String, Binding)> {
        // Update simulation time in Rhai scope for calendar functions
        self.update_scope_time();
        
        let mut final_enabled_bindings = Vec::new();
        if let Some(net) = self.model.petri_nets.first() {
            for transition in &net.transitions {
                let mut potential_bindings: Vec<Binding> = vec![Binding::new()];
                // Input arcs: target is transition, OR bidirectional with source as transition
                let input_arcs: Vec<_> = net.arcs.iter().filter(|a| {
                    a.target == transition.id || 
                    (a.is_bidirectional && a.source == transition.id)
                }).collect();

                for arc in &input_arcs {
                    // For bidirectional arcs where source is the transition, the place is the target
                    let place_id = if arc.is_bidirectional && arc.source == transition.id {
                        &arc.target
                    } else {
                        &arc.source
                    };
                    let inscription = &arc.inscription;
                    let mut next_bindings_for_arc = Vec::new();

                    let all_tokens_in_place = match self.current_marking.get(place_id) {
                        Some(tokens) => tokens,
                        None => {
                            potential_bindings = vec![];
                            break;
                        }
                    };
                    
                    let timestamps_for_place = self.token_timestamps.get(place_id.as_str())
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);

                    for current_binding in &potential_bindings {
                        let current_time = self.current_time;
                        let available_tokens_here: Vec<_> = Self::get_available_token_indices_with_place(
                            place_id,
                            all_tokens_in_place,
                            timestamps_for_place,
                            current_binding.consumed_tokens_map.get(place_id),
                            current_time,
                        ).into_iter().map(|(_, _, token)| token).collect();

                        let available_token_indices: Vec<usize> = Self::get_available_token_indices_with_place(
                            place_id,
                            all_tokens_in_place,
                            timestamps_for_place,
                            current_binding.consumed_tokens_map.get(place_id),
                            current_time,
                        ).into_iter().map(|(_, index, _)| index).collect();

                        if !inscription.is_empty() && self.declared_variables.contains_key(inscription) {
                            let var_name = inscription;
                            if self.is_list_variable(var_name) {
                                /*
                                [DEBUG] Deferring list variable '{}' for arc from place '{}' to transition '{}'
                                */
                                let mut new_binding = current_binding.clone();
                                let deferred_info = DeferredArcInfo {
                                    arc_id: arc.id.clone(),
                                    place_id: place_id.clone(),
                                    variable_name: var_name.clone(),
                                };
                                new_binding.deferred_list_arcs
                                    .entry(var_name.clone())
                                    .or_default()
                                    .push(deferred_info);
                                next_bindings_for_arc.push(new_binding);
                            } else {
                                /*
                                [DEBUG] Attempting to bind variable '{}' from place '{}' for transition '{}'
                                */
                                if let Some(bound_value) = current_binding.variables.get(var_name) {
                                    let bound_value_str = bound_value.to_string();
                                    let mut found_match = false;
                                    for token in &available_tokens_here {
                                        // For timed tokens, compare the extracted value
                                        let token_value = extract_token_value(token);
                                        if token_value.to_string() == bound_value_str {
                                            /*
                                            [DEBUG] Found matching token '{}' for variable '{}' in place '{}'
                                            */
                                            let mut new_binding = current_binding.clone();
                                            new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                            next_bindings_for_arc.push(new_binding);
                                            found_match = true;
                                            break;
                                        }
                                    }
                                    if !found_match {
                                        /*
                                        [DEBUG] No matching token for already bound variable '{}' in place '{}'
                                        */
                                    }
                                } else {
                                    let mut unique_tokens_processed = HashSet::new();
                                    for token in &available_tokens_here {
                                        // Extract the actual value from timed tokens for binding
                                        let token_value = extract_token_value(token);
                                        if unique_tokens_processed.insert(token_value.to_string()) {
                                            /*
                                            [DEBUG] Binding variable '{}' to token '{}' from place '{}' for transition '{}'
                                            */
                                            let mut new_binding = current_binding.clone();
                                            // Bind the extracted value, not the whole timed token
                                            new_binding.variables.insert(var_name.clone(), token_value);
                                            new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                            next_bindings_for_arc.push(new_binding);
                                        }
                                    }
                                }
                            }
                        } else if inscription.is_empty() || inscription == "[]" {
                            next_bindings_for_arc.push(current_binding.clone());
                        } else {
                            if let Some(inscription_ast) = self.arc_expressions.get(&arc.id) {
                                let mut binding_scope = self.rhai_scope.clone();
                                let mut depends_on_deferred = false;
                                for (var, val) in &current_binding.variables {
                                    binding_scope.push_constant(var, val.clone());
                                }
                                for deferred_var in current_binding.deferred_list_arcs.keys() {
                                    if inscription.contains(deferred_var) {
                                        depends_on_deferred = true;
                                        break;
                                    }
                                }

                                if depends_on_deferred {
                                    eprintln!("Warning: Complex inscription '{}' might depend on deferred list variable. Deferral not fully implemented.", inscription);
                                    next_bindings_for_arc.push(current_binding.clone());
                                } else {
                                    match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut binding_scope, inscription_ast) {
                                        Ok(result_dynamic) => {
                                            let required_tokens = Self::dynamic_to_vec_dynamic(result_dynamic);
                                            match check_and_consume_multiset(&required_tokens, &available_token_indices, all_tokens_in_place) {
                                                Some(consumed_for_this_arc) => {
                                                    let mut new_binding = current_binding.clone();
                                                    if !consumed_for_this_arc.is_empty() {
                                                        new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().extend(consumed_for_this_arc);
                                                    }
                                                    next_bindings_for_arc.push(new_binding);
                                                }
                                                None => {}
                                            }
                                        }
                                        Err(e) => {
                                            if let EvalAltResult::ErrorVariableNotFound(ref name, ..) = *e {
                                                // First, try to parse as a product inscription like [n, p]
                                                if let Some(var_names) = parse_product_inscription(inscription) {
                                                    // Check if this is a product-typed place and the first missing var matches
                                                    if self.is_place_product_type(place_id) && var_names.contains(&name.to_string()) {
                                                        // This is a product inscription - try to bind from product tokens
                                                        let mut unique_tokens_processed = HashSet::new();
                                                        for token in &available_tokens_here {
                                                            // Extract the actual value from timed tokens (unwrap {value, timestamp} maps)
                                                            let token_value = extract_token_value(token);
                                                            let token_str = token_value.to_string();
                                                            if unique_tokens_processed.insert(token_str) {
                                                                // Try to extract product components from the unwrapped value
                                                                if let Some(components) = extract_product_components(&token_value) {
                                                                    if components.len() == var_names.len() {
                                                                        // Check if bound variables match, and bind unbound ones
                                                                        let mut new_binding = current_binding.clone();
                                                                        let mut all_match = true;

                                                                        for (i, var_name) in var_names.iter().enumerate() {
                                                                            let component = &components[i];
                                                                            if let Some(existing_val) = current_binding.variables.get(var_name) {
                                                                                // Variable already bound - check if it matches
                                                                                if existing_val.to_string() != component.to_string() {
                                                                                    all_match = false;
                                                                                    break;
                                                                                }
                                                                            } else {
                                                                                // Bind the variable to this component
                                                                                new_binding.variables.insert(var_name.clone(), component.clone());
                                                                            }
                                                                        }

                                                                        if all_match {
                                                                            new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                                                            next_bindings_for_arc.push(new_binding);
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        // Continue to next binding even if no matches found (it will be filtered out naturally)
                                                        continue;
                                                    }
                                                }

                                                // Fall back to simple var array inscription handling
                                                let (var_name_opt, required_count_opt) = parse_simple_var_array_inscription(inscription);
                                                if let (Some(var_name), Some(required_count)) = (var_name_opt, required_count_opt) {
                                                    if var_name == *name && !current_binding.variables.contains_key(&var_name) && !self.is_list_variable(&var_name) {
                                                        let mut available_tokens_grouped: HashMap<String, Vec<usize>> = HashMap::new();
                                                        for &index in &available_token_indices {
                                                            available_tokens_grouped
                                                                .entry(all_tokens_in_place[index].to_string())
                                                                .or_default()
                                                                .push(index);
                                                        }
                                                        for (_token_str, indices) in available_tokens_grouped {
                                                            if indices.len() >= required_count {
                                                                let token_val = all_tokens_in_place[indices[0]].clone();
                                                                let mut new_binding = current_binding.clone();
                                                                new_binding.variables.insert(var_name.clone(), token_val.clone());
                                                                let consumed_tokens_for_arc: Vec<Dynamic> = std::iter::repeat(token_val).take(required_count).collect();
                                                                new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().extend(consumed_tokens_for_arc);
                                                                next_bindings_for_arc.push(new_binding);
                                                            }
                                                        }
                                                    } else {
                                                        eprintln!("Eval Error (VarNotFound, Mismatch/Bound/List): Arc {}, Inscription '{}', Transition {}: {}. Variable '{}' involved.", arc.id, inscription, transition.name, e, name);
                                                    }
                                                } else {
                                                    // Check if the missing variable is a declared variable that should be bound from this place
                                                    // This handles complex inscriptions like "if n == k && p != stop { str + p } else { str }"
                                                    // where 'str' needs to be bound from the place's tokens
                                                    let missing_var = name.to_string();
                                                    if self.declared_variables.contains_key(&missing_var) && !current_binding.variables.contains_key(&missing_var) {
                                                        // Check if the missing variable's colorset matches the place's colorset
                                                        let var_colorset = self.declared_variables.get(&missing_var);
                                                        let place_colorset = self.get_place_colorset(place_id);
                                                        
                                                        if var_colorset == place_colorset.as_ref() {
                                                            // Bind the variable from available tokens and consume one token
                                                            let mut unique_tokens_processed = HashSet::new();
                                                            for token in &available_tokens_here {
                                                                if unique_tokens_processed.insert(token.to_string()) {
                                                                    let mut new_binding = current_binding.clone();
                                                                    new_binding.variables.insert(missing_var.clone(), token.clone());
                                                                    new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                                                    next_bindings_for_arc.push(new_binding);
                                                                }
                                                            }
                                                        } else {
                                                            eprintln!("Eval Error (VarNotFound, Type Mismatch): Arc {}, Inscription '{}', Transition {}: Variable '{}' has colorset {:?} but place has {:?}.", 
                                                                arc.id, inscription, transition.name, missing_var, var_colorset, place_colorset);
                                                        }
                                                    } else {
                                                        eprintln!("Eval Error (VarNotFound, Unhandled Pattern): Arc {}, Inscription '{}', Transition {}: {}. Requires unbound var '{}'.", arc.id, inscription, transition.name, e, name);
                                                    }
                                                }
                                            } else {
                                                eprintln!("Eval Error (Other): Arc {}, Inscription '{}', Transition {}: {}", arc.id, inscription, transition.name, e);
                                            }
                                        }
                                    }
                                }
                            } else {
                                eprintln!("Internal Error: Compiled AST not found for non-empty input arc inscription '{}' (Arc ID: {})", inscription, arc.id);
                            }
                        }
                    }

                    potential_bindings = next_bindings_for_arc;
                    if potential_bindings.is_empty() {
                        break;
                    }
                }

                // --- Second Pass: Evaluate Guards and Resolve Deferred List Bindings ---
                let mut enabled_bindings_for_transition = Vec::new();
                for mut binding in potential_bindings {
                    if binding.deferred_list_arcs.is_empty() {
                        let guard_passes = match self.guards.get(&transition.id) {
                            Some(guard_ast) => {
                                let mut binding_scope = self.rhai_scope.clone();
                                for (var, val) in &binding.variables {
                                    binding_scope.push_constant(var, val.clone());
                                }
                                match self.rhai_engine.eval_ast_with_scope::<bool>(&mut binding_scope, guard_ast) {
                                    Ok(result) => result,
                                    Err(e) => {
                                        eprintln!("Error evaluating guard (no deferred) for transition {}: {}", transition.name, e);
                                        false
                                    }
                                }
                            }
                            None => true,
                        };
                        if guard_passes {
                            enabled_bindings_for_transition.push(binding);
                        }
                    } else {
                        /*
                        [DEBUG] Deferred list variables for transition '{}': {:?}
                        */
                        let guard_ast_option = self.guards.get(&transition.id);

                        if guard_ast_option.is_none() {
                            eprintln!("Warning: Transition {} has deferred list variables but no guard to drive binding. Discarding binding.", transition.name);
                            continue;
                        }

                        let guard_ast = guard_ast_option.unwrap();
                        let mut binding_scope = self.rhai_scope.clone();
                        for (var, val) in &binding.variables {
                            binding_scope.push_constant(var, val.clone());
                        }

                        match self.rhai_engine.eval_ast_with_scope::<bool>(&mut binding_scope, guard_ast) {
                            Ok(result) => {
                                if result {
                                    eprintln!("Warning: Guard for transition {} passed but deferred list variables remain unbound. Consumption unclear. Keeping binding.", transition.name);
                                    binding.deferred_list_arcs.clear();
                                    enabled_bindings_for_transition.push(binding);
                                }
                            }
                            Err(e) => {
                                if let EvalAltResult::ErrorVariableNotFound(ref missing_var, ..) = *e {
                                    if let Some(deferred_arc_infos) = binding.deferred_list_arcs.get(missing_var) {
                                        /*
                                        [DEBUG] Guard needs deferred variable '{}' for transition '{}'
                                        */
                                        let deferred_arc_infos_clone = deferred_arc_infos.clone();

                                        let guard_string = &transition.guard;
                                        let parts: Vec<&str> = guard_string.split('=').map(|s| s.trim()).collect();
                                        if parts.len() == 2 {
                                            let potential_list_var: Option<&str>;
                                            let potential_expr_str: Option<&str>;

                                            if parts[0] == missing_var && !parts[1].contains(missing_var) {
                                                potential_list_var = Some(parts[0]);
                                                potential_expr_str = Some(parts[1]);
                                            } else if parts[1] == missing_var && !parts[0].contains(missing_var) {
                                                potential_list_var = Some(parts[1]);
                                                potential_expr_str = Some(parts[0]);
                                            } else {
                                                potential_list_var = None;
                                                potential_expr_str = None;
                                            }

                                            if let (Some(list_var_name), Some(expr_str)) = (potential_list_var, potential_expr_str) {
                                                if list_var_name == missing_var {
                                                    match self.rhai_engine.compile_expression(expr_str) {
                                                        Ok(expr_ast) => {
                                                            match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut binding_scope, &expr_ast) {
                                                                Ok(required_dynamic) => {
                                                                    let required_multiset = Self::dynamic_to_vec_dynamic(required_dynamic);
                                                                    /*
                                                                    [DEBUG] For transition '{}', guard requires '{}' = {:?}
                                                                    */
                                                                    let mut all_available_tokens_info = Vec::new();
                                                                    for info in &deferred_arc_infos_clone {
                                                                        let place_id = &info.place_id;
                                                                        if let Some(tokens_in_place) = self.current_marking.get(place_id) {
                                                                            /*
                                                                            [DEBUG] Available tokens in place '{}': {:?}
                                                                            */
                                                                            let timestamps_for_place = self.token_timestamps.get(place_id.as_str())
                                                                                .map(|v| v.as_slice())
                                                                                .unwrap_or(&[]);
                                                                            let available_here = Self::get_available_token_indices_with_place(
                                                                                place_id,
                                                                                tokens_in_place,
                                                                                timestamps_for_place,
                                                                                binding.consumed_tokens_map.get(place_id),
                                                                                self.current_time,
                                                                            );
                                                                            all_available_tokens_info.extend(available_here.into_iter().map(|(_, idx, tok)| (place_id.clone(), idx, tok)));
                                                                        }
                                                                    }

                                                                    match check_and_consume_multiset_multi_place(&required_multiset, &all_available_tokens_info) {
                                                                        Some(consumption_map) => {
                                                                            /*
                                                                            [DEBUG] Consuming for '{}' from places: {:?}
                                                                            */
                                                                            binding.variables.insert(list_var_name.to_string(), Dynamic::from(required_multiset.clone()));
                                                                            for (place_id, consumed_tokens) in consumption_map {
                                                                                if !consumed_tokens.is_empty() {
                                                                                    binding.consumed_tokens_map.entry(place_id).or_default().extend(consumed_tokens);
                                                                                }
                                                                            }

                                                                            binding.deferred_list_arcs.remove(list_var_name);

                                                                            let comparison_guard_str = format!("{} == {}", parts[0], parts[1]);
                                                                            match self.rhai_engine.compile_expression(&comparison_guard_str) {
                                                                                Ok(comp_ast) => {
                                                                                    let mut re_eval_scope = self.rhai_scope.clone();
                                                                                    for (var, val) in &binding.variables {
                                                                                        re_eval_scope.push_constant(var, val.clone());
                                                                                    }

                                                                                    match self.rhai_engine.eval_ast_with_scope::<bool>(&mut re_eval_scope, &comp_ast) {
                                                                                        Ok(guard_result) => {
                                                                                            if guard_result {
                                                                                                enabled_bindings_for_transition.push(binding);
                                                                                            } else {
                                                                                                eprintln!("Guard re-evaluation failed for {} after resolving {}", transition.name, list_var_name);
                                                                                            }
                                                                                        }
                                                                                        Err(re_eval_err) => {
                                                                                            eprintln!("Error re-evaluating guard for {} as comparison: {}", transition.name, re_eval_err);
                                                                                        }
                                                                                    }
                                                                                }
                                                                                Err(_comp_err) => {
                                                                                    eprintln!("Error compiling comparison guard for {}: {}", transition.name, _comp_err);
                                                                                }
                                                                            }
                                                                        }
                                                                        None => {
                                                                            /*
                                                                            [DEBUG] Required multiset {:?} not available in any combination of input places for '{}'
                                                                            */
                                                                        }
                                                                    }
                                                                }
                                                                Err(_eval_err) => { /*
                                                                [DEBUG] Error evaluating expression '{}' in guard for {}: {}
                                                                */ }
                                                            }
                                                        }
                                                        Err(_compile_err) => { /*
                                                        [DEBUG] Error compiling expression '{}' in guard for {}: {}
                                                        */ }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        eprintln!("Error evaluating guard for transition {}: Variable '{}' not found (and not deferred).", transition.name, missing_var);
                                    }
                                } else {
                                    eprintln!("Error evaluating guard for transition {}: {}", transition.name, e);
                                }
                            }
                        }
                    }
                }

                for enabled_binding in enabled_bindings_for_transition {
                    if !enabled_binding.deferred_list_arcs.is_empty() {
                        eprintln!("Internal Error: Enabled binding for {} still has deferred arcs: {:?}", transition.name, enabled_binding.deferred_list_arcs.keys());
                    } else {
                        final_enabled_bindings.push((transition.id.clone(), enabled_binding));
                    }
                }
            }
        }
        final_enabled_bindings
    }

    fn select_transition_and_binding(
        &self,
        enabled: &[(String, Binding)]
    ) -> (String, Binding) {
        if enabled.is_empty() {
            panic!("Cannot select from empty list of enabled transitions/bindings");
        }

        let mut rng = rand::rng();
        let chosen_pair = enabled.choose(&mut rng)
            .expect("Internal error: Failed to choose from non-empty enabled bindings");

        (chosen_pair.0.clone(), chosen_pair.1.clone())
    }

    pub fn get_marking(&self, place_id: &str) -> Option<&Vec<Dynamic>> {
        self.current_marking.get(place_id)
    }

    pub fn get_all_markings(&self) -> &HashMap<String, Vec<Dynamic>> {
        &self.current_marking
    }

    /// Get the current simulation time in milliseconds
    pub fn get_current_time(&self) -> i64 {
        self.current_time
    }

    /// Set the current simulation time in milliseconds
    pub fn set_current_time(&mut self, time: i64) {
        self.current_time = time;
    }

    /// Get the simulation epoch (ISO 8601 string)
    pub fn get_simulation_epoch(&self) -> Option<&String> {
        self.simulation_epoch.as_ref()
    }

    /// Set the simulation epoch (ISO 8601 string)
    pub fn set_simulation_epoch(&mut self, epoch: Option<String>) {
        self.simulation_epoch = epoch;
    }

    /// Advance simulation time by a given delta in milliseconds
    pub fn advance_time(&mut self, delta_ms: i64) {
        self.current_time += delta_ms;
    }

    /// Find the earliest timestamp of any token that is in the future (after current_time).
    /// Returns None if there are no future tokens.
    fn find_earliest_future_token_time(&self) -> Option<i64> {
        let mut earliest: Option<i64> = None;
        
        for (place_id, tokens) in &self.current_marking {
            let timestamps = self.token_timestamps.get(place_id);
            
            for (index, token) in tokens.iter().enumerate() {
                // Get timestamp from parallel array or embedded in token
                let timestamp = if let Some(ts_vec) = timestamps {
                    if let Some(&ts) = ts_vec.get(index) {
                        if ts != 0 { ts } else { extract_token_timestamp(token) }
                    } else {
                        extract_token_timestamp(token)
                    }
                } else {
                    extract_token_timestamp(token)
                };
                
                // Only consider future tokens
                if timestamp > self.current_time {
                    earliest = Some(match earliest {
                        None => timestamp,
                        Some(e) => std::cmp::min(e, timestamp),
                    });
                }
            }
        }
        
        earliest
    }

    /// Get the list of currently enabled transitions.
    /// Returns a vector of (transition_id, transition_name) pairs.
    pub fn get_enabled_transitions(&self) -> Vec<(String, String)> {
        let mut enabled = Vec::new();
        if let Some(net) = self.model.petri_nets.first() {
            // We need to check which transitions have at least one valid binding
            // For now, we use a simplified approach by checking the enabled bindings
            let mut temp_self = self.clone_for_check();
            let enabled_bindings = temp_self.find_enabled_bindings();
            
            // Collect unique transition IDs that have at least one binding
            let mut seen_transitions = std::collections::HashSet::new();
            for (transition_id, _) in enabled_bindings {
                if !seen_transitions.contains(&transition_id) {
                    if let Some(transition) = net.transitions.iter().find(|t| t.id == transition_id) {
                        enabled.push((transition_id.clone(), transition.name.clone()));
                        seen_transitions.insert(transition_id);
                    }
                }
            }
        }
        enabled
    }

    /// Fire a specific transition by ID.
    /// Returns the firing event data if successful, None if the transition is not enabled.
    pub fn fire_transition(&mut self, transition_id: &str) -> Option<FiringEventData> {
        let enabled_bindings = self.find_enabled_bindings();
        
        // Find bindings for the specified transition
        let bindings_for_transition: Vec<_> = enabled_bindings
            .into_iter()
            .filter(|(tid, _)| tid == transition_id)
            .collect();
        
        if bindings_for_transition.is_empty() {
            return None;
        }
        
        // Pick a random binding for this transition (could be extended to allow specific binding selection)
        let mut rng = rand::rng();
        let (selected_transition_id, selected_binding) = bindings_for_transition
            .into_iter()
            .choose(&mut rng)
            .unwrap();
        
        // Fire the transition with the selected binding
        self.fire_transition_with_binding(&selected_transition_id, selected_binding)
    }

    /// Clone self for read-only checking (used by get_enabled_transitions)
    fn clone_for_check(&self) -> Self {
        Simulator {
            model: self.model.clone(),
            current_marking: self.current_marking.clone(),
            token_timestamps: self.token_timestamps.clone(),
            current_time: self.current_time,
            simulation_epoch: self.simulation_epoch.clone(),
            epoch_ms: self.epoch_ms,
            timed_places: self.timed_places.clone(),
            rhai_engine: Engine::new(), // Fresh engine for checking
            rhai_scope: self.rhai_scope.clone(),
            guards: self.guards.clone(),
            arc_expressions: self.arc_expressions.clone(),
            initial_marking_expressions: self.initial_marking_expressions.clone(),
            time_expressions: self.time_expressions.clone(),
            declared_variables: self.declared_variables.clone(),
            parsed_color_sets: self.parsed_color_sets.clone(),
        }
    }

    /// Internal method to fire a transition with a specific binding
    fn fire_transition_with_binding(&mut self, transition_id: &str, selected_binding: Binding) -> Option<FiringEventData> {
        // Update simulation time in Rhai scope for calendar functions in time/arc expressions
        self.update_scope_time();
        
        let transition_to_fire = self.model.find_transition(transition_id)?;
        
        let mut consumed_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut produced_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();

        // Consume tokens
        for (place_id, tokens_to_consume) in &selected_binding.consumed_tokens_map {
            if let Some(available_tokens) = self.current_marking.get_mut(place_id) {
                let mut consumed_for_event = Vec::new();
                let mut remaining_tokens = Vec::new();
                let mut needed_counts: HashMap<String, usize> = HashMap::new();
                for token in tokens_to_consume {
                    *needed_counts.entry(token.to_string()).or_insert(0) += 1;
                }

                let mut current_counts: HashMap<String, usize> = HashMap::new();

                for token in available_tokens.iter() {
                    let token_str = token.to_string();
                    let needed_count = needed_counts.get(&token_str).copied().unwrap_or(0);
                    let current_count = current_counts.entry(token_str).or_insert(0);

                    if *current_count < needed_count {
                        consumed_for_event.push(token.clone());
                        *current_count += 1;
                    } else {
                        remaining_tokens.push(token.clone());
                    }
                }

                *available_tokens = remaining_tokens;
                consumed_tokens_event.insert(place_id.clone(), consumed_for_event);
            }
        }

        // Set up firing scope with bound variables
        let mut firing_scope = self.rhai_scope.clone();
        for (var, value) in &selected_binding.variables {
            firing_scope.push_constant(var, value.clone());
        }

        // Handle unbound range variables (same logic as run_step)
        let mut rng = rand::rng();
        let mut already_bound_random: std::collections::HashSet<String> = std::collections::HashSet::new();
        if let Some(net) = self.model.petri_nets.first() {
            for arc in &net.arcs {
                let is_output_arc = arc.source == *transition_id || 
                    (arc.is_bidirectional && arc.target == *transition_id);
                
                if is_output_arc {
                    for (var_name, colorset_name) in &self.declared_variables {
                        if selected_binding.variables.contains_key(var_name) {
                            continue;
                        }
                        if already_bound_random.contains(var_name) {
                            continue;
                        }
                        if !is_variable_in_inscription(var_name, &arc.inscription) {
                            continue;
                        }
                        if let Some(parsed_cs) = self.parsed_color_sets.get(colorset_name) {
                            if parsed_cs.kind == ColorSetKind::IntRange {
                                if let Some((start, end)) = parsed_cs.range {
                                    let random_value = rng.random_range(start..=end);
                                    firing_scope.push_constant(var_name, Dynamic::from(random_value));
                                    already_bound_random.insert(var_name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Evaluate transition time expression to get delay for produced tokens
        let time_delay: i64 = if let Some(time_ast) = self.time_expressions.get(transition_id) {
            match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, time_ast) {
                Ok(result) => {
                    let delay = result.as_int().unwrap_or(0);
                    #[cfg(target_arch = "wasm32")]
                    {
                        web_sys::console::log_1(&format!(
                            "[WASM] fire_transition_with_binding: Transition {} time delay: {} ms",
                            transition_to_fire.name, delay
                        ).into());
                    }
                    delay
                }
                Err(e) => {
                    eprintln!("Error evaluating time expression for transition {}: {}", transition_to_fire.name, e);
                    0
                }
            }
        } else {
            0
        };

        // Calculate the timestamp for produced tokens (current_time + time_delay)
        let produced_token_time = self.current_time + time_delay;

        // Produce tokens on output arcs
        if let Some(net) = self.model.petri_nets.first() {
            let output_arcs: Vec<_> = net.arcs.iter().filter(|a| {
                a.source == *transition_id || 
                (a.is_bidirectional && a.target == *transition_id)
            }).collect();

            for arc in output_arcs {
                let target_place_id = if arc.is_bidirectional && arc.target == *transition_id {
                    &arc.source
                } else {
                    &arc.target
                };

                if let Some(ast) = self.arc_expressions.get(&arc.id) {
                    match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, ast) {
                        Ok(produced_tokens_dynamic) => {
                            let is_product_place = self.is_place_product_type(target_place_id);

                            let tokens_to_add = if is_product_place {
                                if let Ok(arr) = produced_tokens_dynamic.clone().into_typed_array::<Dynamic>() {
                                    if !arr.is_empty() {
                                        if arr[0].clone().into_typed_array::<Dynamic>().is_ok() {
                                            arr
                                        } else {
                                            vec![produced_tokens_dynamic]
                                        }
                                    } else {
                                        vec![]
                                    }
                                } else if produced_tokens_dynamic.is_unit() {
                                    vec![produced_tokens_dynamic]
                                } else {
                                    vec![produced_tokens_dynamic]
                                }
                            } else {
                                Self::dynamic_to_vec_dynamic(produced_tokens_dynamic)
                            };

                            if !tokens_to_add.is_empty() {
                                let is_timed = self.timed_places.get(target_place_id).copied().unwrap_or(false);

                                let tokens_for_marking: Vec<Dynamic> = if is_timed {
                                    tokens_to_add.iter().map(|t| {
                                        if t.is_map() {
                                            if let Some(map) = t.read_lock::<rhai::Map>() {
                                                if map.contains_key("value") && map.contains_key("timestamp") {
                                                    return t.clone();
                                                }
                                            }
                                        }
                                        create_timed_token(t.clone(), produced_token_time)
                                    }).collect()
                                } else {
                                    tokens_to_add.clone()
                                };

                                produced_tokens_event.entry(target_place_id.clone()).or_default().extend(tokens_for_marking.clone());
                                self.current_marking.entry(target_place_id.clone()).or_default().extend(tokens_for_marking.clone());

                                let timestamps = self.token_timestamps.entry(target_place_id.clone()).or_default();
                                for _ in 0..tokens_to_add.len() {
                                    timestamps.push(produced_token_time);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error evaluating output arc {}: {}", arc.id, e);
                        }
                    }
                }
            }
        }

        Some(FiringEventData {
            transition_id: transition_id.to_string(),
            transition_name: transition_to_fire.name.clone(),
            simulation_time: self.current_time,
            consumed: consumed_tokens_event,
            produced: produced_tokens_event,
        })
    }

    #[allow(dead_code)]
    fn get_parsed_colorset(&self, name: &str) -> Option<&ParsedColorSet> {
        self.parsed_color_sets.get(name)
    }

    #[allow(dead_code)]
    fn is_list_colorset(&self, name: &str) -> bool {
        self.parsed_color_sets.get(name).map_or(false, |cs| cs.kind == ColorSetKind::List)
    }

    #[allow(dead_code)]
    fn get_list_element_type(&self, name: &str) -> Option<&String> {
        self.parsed_color_sets.get(name).and_then(|cs| {
            if cs.kind == ColorSetKind::List {
                cs.base_type_name.as_ref()
            } else {
                None
            }
        })
    }

    #[allow(dead_code)]
    fn get_int_range(&self, name: &str) -> Option<(i64, i64)> {
        self.parsed_color_sets.get(name).and_then(|cs| {
            if cs.kind == ColorSetKind::IntRange {
                cs.range
            } else {
                None
            }
        })
    }

    /// Checks if a colorset name refers to a product type.
    fn is_product_colorset(&self, name: &str) -> bool {
        self.parsed_color_sets.get(name).map_or(false, |cs| cs.kind == ColorSetKind::Product)
    }

    /// Gets the component types of a product colorset.
    #[allow(dead_code)]
    fn get_product_component_types(&self, name: &str) -> Option<&Vec<String>> {
        self.parsed_color_sets.get(name).and_then(|cs| {
            if cs.kind == ColorSetKind::Product {
                cs.component_types.as_ref()
            } else {
                None
            }
        })
    }

    /// Checks if a place (by ID) has a product colorset.
    fn is_place_product_type(&self, place_id: &str) -> bool {
        if let Some(net) = self.model.petri_nets.first() {
            if let Some(place) = net.places.iter().find(|p| p.id == place_id) {
                return self.is_product_colorset(&place.color_set);
            }
        }
        false
    }

    /// Gets the colorset name for a place by ID.
    fn get_place_colorset(&self, place_id: &str) -> Option<String> {
        if let Some(net) = self.model.petri_nets.first() {
            if let Some(place) = net.places.iter().find(|p| p.id == place_id) {
                return Some(place.color_set.clone());
            }
        }
        None
    }

    /// Checks if all variables in a product inscription are declared and their types 
    /// match the product colorset component types.
    #[allow(dead_code)]
    fn validate_product_binding(&self, var_names: &[String], place_id: &str) -> bool {
        let colorset_name = match self.get_place_colorset(place_id) {
            Some(name) => name,
            None => return false,
        };

        let component_types = match self.get_product_component_types(&colorset_name) {
            Some(types) => types,
            None => return false,
        };

        if var_names.len() != component_types.len() {
            return false;
        }

        // Check that each variable is declared and its type matches the component type
        for (i, var_name) in var_names.iter().enumerate() {
            if let Some(var_colorset) = self.declared_variables.get(var_name) {
                // The variable's colorset should match the component type
                if var_colorset != &component_types[i] {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}
