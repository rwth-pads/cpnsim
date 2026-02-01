use crate::model::{PetriNetData, FiringEventData};
use rhai::{Engine, Scope, Dynamic, AST, Module, EvalAltResult, NativeCallContext};
use std::collections::{HashMap, HashSet};
use rand::prelude::*;

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

#[derive(Debug)]
#[allow(dead_code)]
pub struct Simulator {
    model: PetriNetData,
    current_marking: HashMap<String, Vec<Dynamic>>,
    rhai_engine: Engine,
    rhai_scope: Scope<'static>,
    guards: HashMap<String, AST>,
    arc_expressions: HashMap<String, AST>,
    initial_marking_expressions: HashMap<String, AST>,
    declared_variables: HashMap<String, String>,
    parsed_color_sets: HashMap<String, ParsedColorSet>,
}

impl Simulator {
    pub fn new(model_data: PetriNetData) -> Result<Self, String> {
        let mut engine = Engine::new();
        let mut scope = Scope::new();

        engine.register_fn("multiset_equal", rhai_multiset_equal_wrapper);

        let mut guards = HashMap::new();
        let mut arc_expressions = HashMap::new();
        let mut initial_marking_expressions = HashMap::new();

        let mut current_marking: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut declared_variables: HashMap<String, String> = HashMap::new();
        let mut parsed_color_sets: HashMap<String, ParsedColorSet> = HashMap::new();

        for var in &model_data.variables {
            declared_variables.insert(var.name.clone(), var.color_set.clone());
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
                    if let Ok(tokens) = result.clone().into_typed_array::<Dynamic>() {
                        current_marking.insert(place_id.clone(), tokens);
                    } else if result.is_unit() {
                        // A single unit value () represents one unit token
                        current_marking.insert(place_id.clone(), vec![result]);
                    } else {
                        current_marking.insert(place_id.clone(), vec![result]);
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error evaluating initial marking for place {}: {}", place_id, e);
                    eprintln!("{}", err_msg);
                    return Err(err_msg);
                }
            }
        }

        Ok(Simulator {
            model: model_data,
            current_marking,
            rhai_engine: engine,
            rhai_scope: scope,
            guards,
            arc_expressions,
            initial_marking_expressions,
            declared_variables,
            parsed_color_sets,
        })
    }

    pub fn run_step(&mut self) -> Option<FiringEventData> {
        let enabled_bindings = self.find_enabled_bindings();
        if enabled_bindings.is_empty() {
            return None;
        }

        let (selected_transition_id, selected_binding) = self.select_transition_and_binding(&enabled_bindings);

        let transition_to_fire = self.model.find_transition(&selected_transition_id)
            .expect("Selected transition ID not found in model - internal error");

        let mut consumed_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut produced_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();

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
                                    produced_tokens_event.entry(place_id.clone()).or_default().extend(tokens_to_add.clone());
                                    self.current_marking.entry(place_id.clone()).or_default().extend(tokens_to_add);
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
        consumed_in_binding: Option<&Vec<Dynamic>>,
    ) -> Vec<(&'a str, usize, Dynamic)> {
        let consumed_counts = consumed_in_binding
            .map(|tokens| create_count_map(tokens))
            .unwrap_or_default();

        let mut available_tokens_info = Vec::new();
        let mut current_token_counts_in_place = HashMap::new();
        for (index, token) in all_tokens_in_place.iter().enumerate() {
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

                    for current_binding in &potential_bindings {
                        let available_tokens_here: Vec<_> = Self::get_available_token_indices_with_place(
                            place_id,
                            all_tokens_in_place,
                            current_binding.consumed_tokens_map.get(place_id),
                        ).into_iter().map(|(_, _, token)| token).collect();

                        let available_token_indices: Vec<usize> = Self::get_available_token_indices_with_place(
                            place_id,
                            all_tokens_in_place,
                            current_binding.consumed_tokens_map.get(place_id),
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
                                        if token.to_string() == bound_value_str {
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
                                        if unique_tokens_processed.insert(token.to_string()) {
                                            /*
                                            [DEBUG] Binding variable '{}' to token '{}' from place '{}' for transition '{}'
                                            */
                                            let mut new_binding = current_binding.clone();
                                            new_binding.variables.insert(var_name.clone(), token.clone());
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
                                                            let token_str = token.to_string();
                                                            if unique_tokens_processed.insert(token_str) {
                                                                // Try to extract product components
                                                                if let Some(components) = extract_product_components(token) {
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
                                                                            let available_here = Self::get_available_token_indices_with_place(
                                                                                place_id,
                                                                                tokens_in_place,
                                                                                binding.consumed_tokens_map.get(place_id),
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
            rhai_engine: Engine::new(), // Fresh engine for checking
            rhai_scope: self.rhai_scope.clone(),
            guards: self.guards.clone(),
            arc_expressions: self.arc_expressions.clone(),
            initial_marking_expressions: self.initial_marking_expressions.clone(),
            declared_variables: self.declared_variables.clone(),
            parsed_color_sets: self.parsed_color_sets.clone(),
        }
    }

    /// Internal method to fire a transition with a specific binding
    fn fire_transition_with_binding(&mut self, transition_id: &str, selected_binding: Binding) -> Option<FiringEventData> {
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
                        Ok(result) => {
                            let tokens_to_produce = if result.is_array() {
                                result.into_array().unwrap_or_default()
                            } else {
                                vec![result]
                            };

                            let place_tokens = self.current_marking.entry(target_place_id.clone()).or_default();
                            for token in &tokens_to_produce {
                                place_tokens.push(token.clone());
                            }
                            produced_tokens_event.entry(target_place_id.clone())
                                .or_default()
                                .extend(tokens_to_produce);
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
