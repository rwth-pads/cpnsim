use crate::model::{PetriNetData, FiringEventData};
// Add EvalAltResult
use rhai::{Engine, Scope, Dynamic, AST, Module, EvalAltResult};
use std::collections::{HashMap, HashSet};
use rand::prelude::*;

// Represents a potential binding for a transition
#[derive(Debug, Clone)]
struct Binding {
    variables: HashMap<String, Dynamic>,
    // Map Place ID -> List of specific token values consumed from that place for this binding
    consumed_tokens_map: HashMap<String, Vec<Dynamic>>,
}

impl Binding {
    fn new() -> Self {
        Binding {
            variables: HashMap::new(),
            consumed_tokens_map: HashMap::new(),
        }
    }
}

// Helper function to check multiset availability and identify tokens to consume.
// Returns Some(consumed_tokens) if successful, None otherwise.
// WARNING: Uses to_string() for comparison, unsuitable for complex types.
fn check_and_consume_multiset(
    required_tokens: &[Dynamic],
    available_indices: &[usize],
    all_tokens_in_place: &[Dynamic],
) -> Option<Vec<Dynamic>> {
    if required_tokens.is_empty() {
        // Empty requirement is always met, consumes nothing.
        return Some(vec![]);
    }

    let mut required_counts: HashMap<String, usize> = HashMap::new();
    for token in required_tokens {
        *required_counts.entry(token.to_string()).or_insert(0) += 1;
    }

    // Create a list of available tokens (value and original index) for matching
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

// Helper function to parse inscriptions like "[v]", "[v,v]", "[v,v,v]" etc.
// Returns (Some(variable_name), Some(count)) if it matches the pattern, otherwise (None, None).
fn parse_simple_var_array_inscription(inscription: &str) -> (Option<String>, Option<usize>) {
    let trimmed = inscription.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return (None, None);
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return (None, None); // Empty array "[]" is handled differently
    }

    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.is_empty() {
         return (None, None); // Should not happen if inner is not empty, but safety check
    }

    let first_var = parts[0];
    // Basic check: is it a valid variable name? (Starts with letter or _, contains letters, numbers, _)
    if first_var.is_empty() || !first_var.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_') || !first_var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return (None, None); // First part doesn't look like a variable
    }

    // Check if all parts are identical to the first part
    if parts.iter().all(|&part| part == first_var) {
        (Some(first_var.to_string()), Some(parts.len()))
    } else {
        (None, None) // Contains different variables or other elements
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Allow unused fields for now
pub struct Simulator {
    model: PetriNetData,
    current_marking: HashMap<String, Vec<Dynamic>>,
    rhai_engine: Engine,
    rhai_scope: Scope<'static>,
    // Uncomment Rhai AST maps
    guards: HashMap<String, AST>,
    arc_expressions: HashMap<String, AST>,
    initial_marking_expressions: HashMap<String, AST>,
    // Store declared variable names and their types (ColorSet names)
    declared_variables: HashMap<String, String>,
}

impl Simulator {
    // Change return type to use String for error
    pub fn new(model_data: PetriNetData) -> Result<Self, String> {
        let mut engine = Engine::new(); // Make engine mutable for registration
        let mut scope = Scope::new(); // Make scope mutable for evaluation

        let mut guards = HashMap::new();
        let mut arc_expressions = HashMap::new();
        let mut initial_marking_expressions = HashMap::new();

        let mut current_marking: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut declared_variables: HashMap<String, String> = HashMap::new();

        for var in &model_data.variables {
            declared_variables.insert(var.name.clone(), var.color_set.clone());
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

    fn dynamic_to_vec_dynamic(d: Dynamic) -> Vec<Dynamic> {
        if let Ok(arr) = d.clone().into_typed_array::<Dynamic>() {
            arr
        } else if d.is_unit() {
            vec![]
        } else {
            vec![d]
        }
    }

    fn find_enabled_bindings(&mut self) -> Vec<(String, Binding)> {
        let mut all_enabled_bindings = Vec::new();
        if let Some(net) = self.model.petri_nets.first() {
            for transition in &net.transitions {
                let mut potential_bindings: Vec<Binding> = vec![Binding::new()];

                let input_arcs: Vec<_> = net.arcs.iter().filter(|a| a.target == transition.id).collect();

                for arc in input_arcs {
                    let place_id = &arc.source;
                    let inscription = &arc.inscription;
                    let mut next_bindings = Vec::new();

                    let all_tokens_in_place = match self.current_marking.get(place_id) {
                        Some(tokens) => tokens,
                        None => {
                            potential_bindings = vec![];
                            break;
                        }
                    };

                    for current_binding in &potential_bindings {
                        let consumed_counts = current_binding.consumed_tokens_map.get(place_id)
                            .map(|tokens| {
                                let mut counts = HashMap::new();
                                for token in tokens {
                                    *counts.entry(token.to_string()).or_insert(0) += 1;
                                }
                                counts
                            })
                            .unwrap_or_default();

                        let mut available_token_indices = Vec::new();
                        let mut current_token_counts_in_place = HashMap::new();
                        for (index, token) in all_tokens_in_place.iter().enumerate() {
                            let token_str = token.to_string();
                            let consumed_count = consumed_counts.get(&token_str).copied().unwrap_or(0);
                            let current_count = current_token_counts_in_place.entry(token_str.clone()).or_insert(0);

                            if *current_count < consumed_count {
                                *current_count += 1;
                            } else {
                                available_token_indices.push(index);
                            }
                        }

                        if !inscription.is_empty() && self.declared_variables.contains_key(inscription) {
                            let var_name = inscription;

                            if let Some(bound_value) = current_binding.variables.get(var_name) {
                                let bound_value_str = bound_value.to_string();
                                let mut found_match_index = None;
                                for &index in &available_token_indices {
                                    if all_tokens_in_place[index].to_string() == bound_value_str {
                                        found_match_index = Some(index);
                                        break;
                                    }
                                }

                                if let Some(index) = found_match_index {
                                    let mut new_binding = current_binding.clone();
                                    let consumed_token = all_tokens_in_place[index].clone();
                                    new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(consumed_token);
                                    next_bindings.push(new_binding);
                                }
                            } else {
                                if available_token_indices.is_empty() {
                                } else {
                                    for &index in &available_token_indices {
                                        let token = &all_tokens_in_place[index];
                                        let mut new_binding = current_binding.clone();
                                        new_binding.variables.insert(var_name.clone(), token.clone());
                                        new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                        next_bindings.push(new_binding);
                                    }
                                }
                            }
                        } else if inscription.is_empty() {
                            if !available_token_indices.is_empty() {
                                next_bindings.push(current_binding.clone());
                            }
                        } else {
                            if let Some(inscription_ast) = self.arc_expressions.get(&arc.id) {
                                let mut binding_scope = self.rhai_scope.clone();
                                for (var, val) in &current_binding.variables {
                                    binding_scope.push_constant(var, val.clone());
                                }

                                match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut binding_scope, inscription_ast) {
                                    Ok(result_dynamic) => {
                                        let required_tokens: Vec<Dynamic> = Self::dynamic_to_vec_dynamic(result_dynamic);

                                        match check_and_consume_multiset(&required_tokens, &available_token_indices, all_tokens_in_place) {
                                            Some(consumed_for_this_arc) => {
                                                let mut new_binding = current_binding.clone();
                                                if !consumed_for_this_arc.is_empty() {
                                                    new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().extend(consumed_for_this_arc);
                                                }
                                                next_bindings.push(new_binding);
                                            }
                                            None => {
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        // Use 'ref name' to borrow instead of move
                                        if let EvalAltResult::ErrorVariableNotFound(ref name, ..) = *e {
                                            if self.declared_variables.contains_key(name) { // Use borrowed 'name' directly
                                                let (var_name_opt, required_count_opt) = parse_simple_var_array_inscription(inscription);

                                                if let (Some(var_name), Some(required_count)) = (var_name_opt, required_count_opt) {
                                                    // Dereference 'name' for comparison with 'var_name' (String)
                                                    if var_name == *name && !current_binding.variables.contains_key(&var_name) {
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

                                                                next_bindings.push(new_binding);
                                                            }
                                                        }
                                                    } else {
                                                        eprintln!(
                                                            "Eval Error (VarNotFound, Mismatch): Arc {}, Inscription '{}', Transition {}: {}. Variable '{}' involved.",
                                                            arc.id, inscription, transition.name, e, name
                                                        );
                                                    }
                                                } else {
                                                    eprintln!(
                                                        "Eval Error (VarNotFound, Unhandled Pattern): Arc {}, Inscription '{}', Transition {}: {}. Requires unbound var '{}'.",
                                                        arc.id, inscription, transition.name, e, name
                                                    );
                                                }
                                            } else {
                                                eprintln!(
                                                    "Eval Error (VarNotFound, Undeclared): Arc {}, Inscription '{}', Transition {}: {}. Variable '{}' not declared.",
                                                    arc.id, inscription, transition.name, e, name
                                                );
                                            }
                                        } else {
                                            eprintln!(
                                                "Eval Error (Other): Arc {}, Inscription '{}', Transition {}: {}",
                                                arc.id, inscription, transition.name, e
                                            );
                                        }
                                    }
                                }
                            } else if !inscription.is_empty() {
                                eprintln!(
                                    "Internal Error: Compiled AST not found for non-empty input arc inscription '{}' (Arc ID: {})",
                                    inscription, arc.id
                                );
                            } else {
                                eprintln!(
                                    "Internal Warning: Reached complex inscription case with empty inscription for arc {}",
                                    arc.id
                                );
                                next_bindings.push(current_binding.clone());
                            }
                        }
                    }

                    potential_bindings = next_bindings;
                    if potential_bindings.is_empty() {
                        break;
                    }
                }

                let mut enabled_bindings_for_transition = Vec::new();
                for binding in potential_bindings {
                    let mut binding_scope = self.rhai_scope.clone();
                    for (var, val) in &binding.variables {
                        binding_scope.push_constant(var, val.clone());
                    }

                    let guard_passes = match self.guards.get(&transition.id) {
                        Some(guard_ast) => {
                            match self.rhai_engine.eval_ast_with_scope::<bool>(&mut binding_scope, guard_ast) {
                                Ok(result) => {
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Error evaluating guard for transition {}: {}", transition.name, e);
                                    false
                                }
                            }
                        }
                        None => true,
                    };

                    if guard_passes {
                        enabled_bindings_for_transition.push(binding);
                    }
                }

                for enabled_binding in enabled_bindings_for_transition {
                    all_enabled_bindings.push((transition.id.clone(), enabled_binding));
                }
            }
        }
        all_enabled_bindings
    }

    fn select_transition_and_binding(
        &self,
        enabled: &[(String, Binding)]
    ) -> (String, Binding) {
        if enabled.is_empty() {
            panic!("Cannot select from empty list of enabled transitions/bindings");
        }

        let mut priorities: HashMap<i64, Vec<&(String, Binding)>> = HashMap::new();
        let default_priority = self.model.get_priority_level("P_NORMAL").unwrap_or(1000);

        for binding_pair in enabled {
            let transition = self.model.find_transition(&binding_pair.0)
                .expect("Enabled transition ID not found in model - internal error");
            let level = transition.priority.is_empty()
                .then(|| default_priority)
                .or_else(|| self.model.get_priority_level(&transition.priority))
                .unwrap_or(default_priority);

            priorities.entry(level).or_default().push(binding_pair);
        }

        let lowest_level = priorities.keys().min().copied()
            .expect("Internal error: Failed to find minimum priority level");

        let highest_priority_options = priorities.get(&lowest_level)
            .expect("Internal error: Lowest priority level not found in map");

        let mut rng = rand::rng();
        let chosen_pair = highest_priority_options.choose(&mut rng)
            .expect("Internal error: Failed to choose from non-empty priority options");

        (chosen_pair.0.clone(), chosen_pair.1.clone())
    }

    pub fn get_marking(&self, place_id: &str) -> Option<&Vec<Dynamic>> {
        self.current_marking.get(place_id)
    }

    pub fn get_all_markings(&self) -> &HashMap<String, Vec<Dynamic>> {
        &self.current_marking
    }
}
