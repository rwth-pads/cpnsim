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
    Unknown,
}

#[derive(Debug, Clone)]
struct ParsedColorSet {
    kind: ColorSetKind,
    base_type_name: Option<String>,
    range: Option<(i64, i64)>,
}

impl ParsedColorSet {
    fn unknown() -> Self {
        ParsedColorSet {
            kind: ColorSetKind::Unknown,
            base_type_name: None,
            range: None,
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

        for net in &model_data.petri_nets {
            for place in &net.places {
                if !place.initial_marking.is_empty() && place.initial_marking != "[]" {
                    match engine.compile_expression(&place.initial_marking) {
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
                        current_marking.insert(place_id.clone(), vec![]);
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

        if let Some(net) = self.model.petri_nets.first() {
            for arc in &net.arcs {
                if arc.source == selected_transition_id {
                    let place_id = &arc.target;
                    if let Some(ast) = self.arc_expressions.get(&arc.id) {
                        match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, ast) {
                            Ok(produced_tokens_dynamic) => {
                                let tokens_to_add = Self::dynamic_to_vec_dynamic(produced_tokens_dynamic);
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
        } else if d.is_unit() {
            vec![]
        } else {
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
                let input_arcs: Vec<_> = net.arcs.iter().filter(|a| a.target == transition.id).collect();

                for arc in &input_arcs {
                    let place_id = &arc.source;
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
                                                    eprintln!("Eval Error (VarNotFound, Unhandled Pattern): Arc {}, Inscription '{}', Transition {}: {}. Requires unbound var '{}'.", arc.id, inscription, transition.name, e, name);
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
}
