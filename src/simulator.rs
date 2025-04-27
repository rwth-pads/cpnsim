use crate::model::{PetriNetData, FiringEventData};
// Remove unused ParseError import
use rhai::{Engine, Scope, Dynamic, AST, Module}; // Remove Position import if no longer needed
use std::collections::{HashMap, HashSet};
// Import the rand prelude for Rng and SliceRandom traits
use rand::prelude::*;
use rand::rng;

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

        // Uncomment maps
        let mut guards = HashMap::new();
        let mut arc_expressions = HashMap::new();
        let mut initial_marking_expressions = HashMap::new();

        let mut current_marking: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut declared_variables: HashMap<String, String> = HashMap::new();

        // --- Register Custom Types/Functions (Optional but Recommended) ---
        // Example: If you have custom token types like `Product { id: String, price: f64 }`
        // engine.register_type_with_name::<Product>("Product")
        //       .register_get("id", Product::get_id)
        //       .register_get("price", Product::get_price);
        // Need to implement these methods on your struct.

        // --- Load Declared Variables ---
        for var in &model_data.variables {
            declared_variables.insert(var.name.clone(), var.color_set.clone());
        }
        println!("Declared variables: {:?}", declared_variables);

        // --- Compile and Register Functions ---
        // Module will be created from AST evaluation below if functions exist
        let mut all_fn_code = String::new();
        for func in &model_data.functions {
            // Append the code of each function definition.
            // Ensure there's separation, e.g., a newline, although Rhai often handles this.
            all_fn_code.push_str(&func.code);
            all_fn_code.push('\n');
        }

        if !all_fn_code.is_empty() {
            println!("Compiling all functions together...");
            match engine.compile(&all_fn_code) {
                Ok(ast) => {
                    // Evaluate the AST containing all functions to define them within a *new* module.
                    // This replaces the non-existent set_script_ast.
                    // Use Module::eval_ast_as_new which takes the scope, ast, and engine.
                    let fn_scope = Scope::new(); // Create a temporary scope for function definition evaluation
                    // Pass scope by value/immutable borrow as expected by eval_ast_as_new
                    match Module::eval_ast_as_new(fn_scope, &ast, &engine) {
                        Ok(compiled_module) => {
                            // Register the newly created module containing the functions globally
                            engine.register_global_module(compiled_module.into());
                        }
                        Err(e) => {
                            let err_msg = format!("Error evaluating function AST into new module: {}", e);
                            println!("{}", err_msg);
                            return Err(err_msg);
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error compiling combined functions: {}", e);
                    println!("{}", err_msg);
                    return Err(err_msg);
                }
            }
        }

        // Module is registered above if functions were compiled and evaluated successfully

        // --- Compile Expressions ---
        for net in &model_data.petri_nets {
            // Compile Initial Markings
            for place in &net.places {
                if !place.initial_marking.is_empty() && place.initial_marking != "[]" {
                    println!("Compiling initial marking for place {}: {}", place.name, place.initial_marking);
                    match engine.compile_expression(&place.initial_marking) {
                        Ok(ast) => {
                            initial_marking_expressions.insert(place.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling initial marking for place {}: {}", place.name, e);
                            println!("{}", err_msg);
                            // Map error to String
                            return Err(err_msg);
                        }
                    }
                }
                // Initialize marking map entry even if no initial marking expression
                current_marking.insert(place.id.clone(), vec![]);
            }
            // Compile Guards
            for transition in &net.transitions {
                // Only compile non-empty and non-"true" guards
                if !transition.guard.is_empty() && transition.guard.to_lowercase() != "true" {
                    println!("Compiling guard for transition {}: {}", transition.name, transition.guard);
                    match engine.compile(&transition.guard) {
                        Ok(ast) => {
                            guards.insert(transition.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling guard for transition {}: {}", transition.name, e);
                            println!("{}", err_msg);
                            // Map error to String
                            return Err(err_msg);
                        }
                    }
                }
            }
            // Compile Arc Inscriptions
            for arc in &net.arcs {
                // Only compile non-empty inscriptions
                if !arc.inscription.is_empty() {
                    println!("Compiling inscription for arc {}: {}", arc.id, arc.inscription);
                    // Use compile_expression as arc inscriptions often evaluate to values/collections
                    match engine.compile_expression(&arc.inscription) {
                        Ok(ast) => {
                            arc_expressions.insert(arc.id.clone(), ast);
                        }
                        Err(e) => {
                            let err_msg = format!("Error compiling inscription for arc {}: {}", arc.id, e);
                            println!("{}", err_msg);
                            // Map error to String
                            return Err(err_msg);
                        }
                    }
                }
            }
        }

        // --- Evaluate Initial Markings ---
        for (place_id, ast) in &initial_marking_expressions {
            println!("Evaluating initial marking for place {}", place_id);
            match engine.eval_ast_with_scope::<Dynamic>(&mut scope, ast) {
                Ok(result) => {
                    // Expecting the result to be a Vec<Dynamic> or convertible to it
                    // Clone result before calling into_typed_array
                    if let Ok(tokens) = result.clone().into_typed_array::<Dynamic>() {
                        println!("  -> Initial tokens: {:?}", tokens);
                        current_marking.insert(place_id.clone(), tokens);
                    } else if result.is_unit() {
                        // Handle case where expression might evaluate to () for empty set
                        current_marking.insert(place_id.clone(), vec![]);
                    } else {
                        // Handle single token case or other types if necessary
                        println!("  -> Initial single token: {:?}", result);
                        current_marking.insert(place_id.clone(), vec![result]);
                        // Alternatively, return an error if the type is unexpected
                        // return Err(Box::new(EvalAltResult::ErrorMismatchDataType(...)));
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error evaluating initial marking for place {}: {}", place_id, e);
                    println!("{}", err_msg);
                    // Map error to String
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
            println!("No enabled transitions/bindings found.");
            return None;
        }

        // selected_binding is now BindingInfo
        let (selected_transition_id, selected_binding) = self.select_transition_and_binding(&enabled_bindings);

        let transition_to_fire = self.model.find_transition(&selected_transition_id)
            .expect("Selected transition ID not found in model - internal error");

        println!(
            "Firing transition: {} ({}) with binding: {:?}",
            transition_to_fire.name,
            selected_transition_id,
            selected_binding.variables // Log the bound variables
        );

        let mut consumed_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();
        let mut produced_tokens_event: HashMap<String, Vec<Dynamic>> = HashMap::new();

        // --- Consume Tokens (using the pre-calculated binding) ---
        for (place_id, tokens_to_consume) in &selected_binding.consumed_tokens_map {
            if let Some(available_tokens) = self.current_marking.get_mut(place_id) {
                println!("  Consuming {:?} from place {}", tokens_to_consume, place_id);
                let mut consumed_for_event = Vec::new();
                let mut remaining_tokens = Vec::new();
                let mut consumed_indices = HashSet::new(); // Track indices to remove efficiently

                // Create a temporary list of tokens to consume for matching
                let mut tokens_needed = tokens_to_consume.clone();

                // Iterate through available tokens to find matches
                for (index, token) in available_tokens.iter().enumerate() {
                    let mut found_match = false;
                    if let Some(needed_index) = tokens_needed.iter().position(|needed| {
                        // WARNING: Using to_string() for comparison is a simplification!
                        // This will NOT work correctly for complex types (structs, maps).
                        // Proper deep comparison or implementing PartialEq for Dynamic might be needed.
                        token.to_string() == needed.to_string()
                    }) {
                        // Found a token to consume
                        consumed_indices.insert(index);
                        consumed_for_event.push(token.clone());
                        let _ = tokens_needed.remove(needed_index); // Remove the matched needed token
                        found_match = true;
                    }

                    if !found_match && !consumed_indices.contains(&index) {
                        // Keep tokens that were not consumed
                        remaining_tokens.push(token.clone());
                    }
                }

                if tokens_needed.is_empty() {
                    // Successfully found all required tokens
                    *available_tokens = remaining_tokens; // Update the marking
                    consumed_tokens_event.insert(place_id.clone(), consumed_for_event);
                } else {
                    // This should NOT happen if find_enabled_bindings is correct
                    eprintln!(
                        "Internal Error: Could not find expected tokens {:?} to consume from place {} for binding {:?}. Available: {:?}",
                        tokens_to_consume, place_id, selected_binding.variables, self.current_marking.get(place_id)
                    );
                    // Attempt to log current state for debugging
                    println!("Current marking state: {:?}", self.current_marking);
                    panic!("Consumption failed unexpectedly!"); // Panic for now, indicates logic error
                    // return None; // Or handle more gracefully
                }
            } else {
                eprintln!("Internal Error: Place {} (expected for consumption) not found in marking map.", place_id);
                panic!("Consumption failed unexpectedly!");
                // return None;
            }
        }

        // --- Produce Tokens ---
        // Create a scope for this specific firing, including the selected binding variables
        let mut firing_scope = self.rhai_scope.clone(); // Clone global scope
        for (var, value) in &selected_binding.variables {
            firing_scope.push_constant(var, value.clone()); // Add bound variables
        }

        if let Some(net) = self.model.petri_nets.first() { // Assuming single net
            for arc in &net.arcs {
                if arc.source == selected_transition_id { // Output arc
                    let place_id = &arc.target;
                    if let Some(ast) = self.arc_expressions.get(&arc.id) {
                        match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut firing_scope, ast) {
                            Ok(produced_tokens_dynamic) => {
                                // Use helper to ensure result is always Vec<Dynamic>
                                let tokens_to_add = Self::dynamic_to_vec_dynamic(produced_tokens_dynamic);
                                println!("  Arc {} produces: {:?}", arc.id, tokens_to_add);
                                if !tokens_to_add.is_empty() {
                                    // Add to event data
                                    produced_tokens_event.entry(place_id.clone()).or_default().extend(tokens_to_add.clone());
                                    // Add to current marking
                                    self.current_marking.entry(place_id.clone()).or_default().extend(tokens_to_add);
                                }
                            }
                            Err(e) => {
                                eprintln!("  Error evaluating output arc {} inscription: {}", arc.id, e);
                                return None; // Abort step on evaluation error
                            }
                        }
                    } else if !arc.inscription.is_empty() {
                        eprintln!("  Internal Error: Compiled AST not found for non-empty output arc {}", arc.id);
                        return None; // Abort step
                    }
                    // Handle arcs with empty inscriptions if necessary (produce nothing)
                }
            }
        } else {
            return None; // No net found
        }

        let event_data = FiringEventData {
            transition_id: selected_transition_id,
            transition_name: transition_to_fire.name.clone(),
            consumed: consumed_tokens_event, // Use the map built during consumption
            produced: produced_tokens_event, // Use the map built during production
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
            // Add a step limit for safety during development
            if step > 100 {
                println!("Stopping run after 100 steps.");
                break;
            }
        }
        println!("--- Simulation run finished ---");
    }

    // Helper to convert Dynamic (single or array) to Vec<Dynamic>
    fn dynamic_to_vec_dynamic(d: Dynamic) -> Vec<Dynamic> {
        if let Ok(arr) = d.clone().into_typed_array::<Dynamic>() {
            arr
        } else if d.is_unit() {
            vec![]
        } else {
            vec![d] // Assume single value if not an array or unit
        }
    }

    // --- find_enabled_bindings (Revised) ---
    // Finds transitions that can fire and the specific bindings (variable assignments
    // and consumed tokens) that enable them.
    // NOTE: This implementation handles multiple input arcs with simple variable inscriptions.
    // It needs extension for complex inscriptions (tuples, expressions), proper token comparison,
    // and potentially performance optimizations.
    fn find_enabled_bindings(&mut self) -> Vec<(String, Binding)> {
        let mut all_enabled_bindings = Vec::new();
        if let Some(net) = self.model.petri_nets.first() { // Assuming single net
            for transition in &net.transitions {
                let mut potential_bindings: Vec<Binding> = vec![Binding::new()]; // Start with one empty binding

                // --- Step 1: Generate potential bindings based on input arcs ---
                let input_arcs: Vec<_> = net.arcs.iter().filter(|a| a.target == transition.id).collect();

                for arc in input_arcs {
                    let place_id = &arc.source;
                    let inscription = &arc.inscription;
                    let mut next_bindings = Vec::new(); // Bindings satisfying arcs up to this one

                    // Check if the source place exists in the current marking
                    let all_tokens_in_place = match self.current_marking.get(place_id) {
                        Some(tokens) => tokens,
                        None => {
                            println!("Warning: Place {} for input arc {} not found in marking.", place_id, arc.id);
                            potential_bindings = vec![]; // Cannot satisfy arc if place doesn't exist
                            break; // Stop processing arcs for this transition
                        }
                    };

                    for current_binding in &potential_bindings {
                        // Determine tokens already consumed from this place *by this specific binding*
                        // We need to know which *specific instances* are consumed if duplicates exist.
                        // Using counts for now, assuming simple comparable types.
                        // WARNING: This comparison logic (using to_string) is a placeholder
                        // and will fail for complex, non-comparable token types.
                        let consumed_counts = current_binding.consumed_tokens_map.get(place_id)
                            .map(|tokens| {
                                let mut counts = HashMap::new();
                                for token in tokens {
                                    *counts.entry(token.to_string()).or_insert(0) += 1;
                                }
                                counts
                            })
                            .unwrap_or_default();

                        // Filter available tokens based on what's already consumed by this binding path
                        let mut available_token_indices = Vec::new();
                        let mut current_counts = HashMap::new();
                        for (index, token) in all_tokens_in_place.iter().enumerate() {
                            let token_str = token.to_string();
                            let consumed_count = consumed_counts.get(&token_str).copied().unwrap_or(0);
                            let current_count = current_counts.entry(token_str).or_insert(0);
                            if *current_count < consumed_count {
                                // This specific instance is already marked as consumed by this binding
                                *current_count += 1;
                            } else {
                                // This instance is available for this binding
                                available_token_indices.push(index);
                            }
                        }

                        // --- Handle inscription type ---
                        if !inscription.is_empty() && self.declared_variables.contains_key(inscription) {
                            // --- Case 1: Simple Variable Inscription ---
                            let var_name = inscription;

                            if let Some(bound_value) = current_binding.variables.get(var_name) {
                                // Variable already bound: Check if a matching *available* token exists
                                let bound_value_str = bound_value.to_string(); // Simplistic comparison
                                let mut found_match = false;
                                for &index in &available_token_indices {
                                    let token = &all_tokens_in_place[index];
                                    if token.to_string() == bound_value_str {
                                        let mut new_binding = current_binding.clone();
                                        new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().push(token.clone());
                                        next_bindings.push(new_binding);
                                        found_match = true;
                                        break; // Consume only one matching token per arc
                                    }
                                }
                                if !found_match {
                                     // Discard binding path: Required token (matching bound value) not available
                                }
                            } else {
                                // Variable not bound yet: Create new bindings for each *available* token
                                if available_token_indices.is_empty() {
                                    // Discard binding path: Place has no available tokens to bind
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
                            // --- Case 2: Empty Inscription ---
                            // CPN Semantics: Usually requires *a* token but doesn't bind/consume.
                            // For simulation, we often treat it as needing the place to be non-empty.
                            // Let's assume it requires at least one *available* token.
                            if !available_token_indices.is_empty() {
                                // Keep the current binding as is, constraint met.
                                next_bindings.push(current_binding.clone());
                            } else {
                                // Discard binding path: Place empty, cannot satisfy empty inscription arc.
                            }
                        } else {
                            // --- Case 3: Complex Inscription (Tuple, Expression, etc.) ---
                            // Evaluate the inscription to determine the required tokens.
                            if let Some(inscription_ast) = self.arc_expressions.get(&arc.id) {
                                // Create a scope specific to this binding path for evaluation
                                let mut binding_scope = self.rhai_scope.clone();
                                for (var, val) in &current_binding.variables {
                                    binding_scope.push_constant(var, val.clone());
                                }

                                match self.rhai_engine.eval_ast_with_scope::<Dynamic>(&mut binding_scope, inscription_ast) {
                                    Ok(result_dynamic) => {
                                        let required_tokens = Self::dynamic_to_vec_dynamic(result_dynamic);
                                        if required_tokens.is_empty() {
                                            // If the expression evaluates to an empty list, it means no specific tokens are required (like a test arc).
                                            // Keep the current binding as is, constraint met.
                                            next_bindings.push(current_binding.clone());
                                        } else {
                                            // We need to find if the required tokens are available among the 'available_token_indices'
                                            // WARNING: This matching logic relies on to_string() and might be incorrect for complex types.
                                            // It also needs to handle duplicates correctly (multiset matching).
                                            let mut available_tokens_to_match: Vec<_> = available_token_indices
                                                .iter()
                                                .map(|&idx| all_tokens_in_place[idx].clone())
                                                .collect();
                                            let mut consumed_for_this_arc = Vec::new();
                                            let mut possible = true;

                                            for required_token in &required_tokens {
                                                let required_token_str = required_token.to_string(); // Simplistic comparison
                                                if let Some(pos) = available_tokens_to_match.iter().position(|avail| avail.to_string() == required_token_str) {
                                                    // Found a match, remove it from available and add to consumed
                                                    let matched_token = available_tokens_to_match.remove(pos);
                                                    consumed_for_this_arc.push(matched_token);
                                                } else {
                                                    // Required token not found among available ones
                                                    possible = false;
                                                    break;
                                                }
                                            }

                                            if possible {
                                                // All required tokens were found
                                                let mut new_binding = current_binding.clone();
                                                new_binding.consumed_tokens_map.entry(place_id.clone()).or_default().extend(consumed_for_this_arc);
                                                next_bindings.push(new_binding);
                                            } else {
                                                // Discard binding path: Required tokens not available.
                                                // println!("Debug: Discarding binding - required tokens {:?} not found in available {:?}", required_tokens, available_tokens_to_match);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Error evaluating input arc {} inscription '{}' for transition {}: {}",
                                            arc.id, inscription, transition.name, e
                                        );
                                        // Discard binding path on evaluation error
                                    }
                                }
                            } else if !inscription.is_empty() {
                                // This case should ideally not happen if compilation succeeded
                                eprintln!(
                                    "Internal Error: Compiled AST not found for non-empty input arc inscription '{}' (Arc ID: {})",
                                    inscription, arc.id
                                );
                                // Discard binding path
                            } else {
                                // Should be handled by Case 2 (Empty Inscription) above.
                                // If reached, it might indicate an issue in the logic flow.
                                eprintln!(
                                    "Internal Warning: Reached complex inscription case with empty inscription for arc {}",
                                    arc.id
                                );
                                next_bindings.push(current_binding.clone()); // Treat as Case 2 for safety?
                            }
                        }

                    } // end loop over current_bindings

                    potential_bindings = next_bindings; // Update bindings for the next arc iteration
                    if potential_bindings.is_empty() {
                        // If any arc cannot be satisfied, no bindings are possible for this transition
                        break;
                    }
                } // end loop over input_arcs


                // --- Step 2: Check Guards for each potential binding ---
                let mut enabled_bindings_for_transition = Vec::new();
                for binding in potential_bindings {
                    // Create a scope specific to this binding
                    let mut binding_scope = self.rhai_scope.clone();
                    for (var, val) in &binding.variables {
                        // Use push_constant for immutability within the guard check
                        binding_scope.push_constant(var, val.clone());
                    }

                    let guard_passes = match self.guards.get(&transition.id) {
                        Some(guard_ast) => {
                            match self.rhai_engine.eval_ast_with_scope::<bool>(&mut binding_scope, guard_ast) {
                                Ok(result) => {
                                    if !result {
                                        // Optional: Log guard failures for debugging
                                        // println!("  Guard failed for transition {} with binding {:?}", transition.name, binding.variables);
                                    }
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Error evaluating guard for transition {}: {}", transition.name, e);
                                    false // Treat evaluation error as guard failure
                                }
                            }
                        }
                        None => true, // No guard (or guard is "true") means it passes
                    };

                    if guard_passes {
                        // This binding satisfies all input arcs and the guard
                        enabled_bindings_for_transition.push(binding);
                    }
                }

                // Add the fully validated bindings for this transition to the overall list
                for enabled_binding in enabled_bindings_for_transition {
                     println!("  Transition {} enabled with binding {:?}", transition.name, enabled_binding.variables);
                     all_enabled_bindings.push((transition.id.clone(), enabled_binding));
                }

            } // end loop over transitions
        }
        all_enabled_bindings
    }

    // Selects a transition and binding based on priority (and potentially randomness)
    fn select_transition_and_binding(
        &self,
        enabled: &[(String, Binding)]
    ) -> (String, Binding) {
        if enabled.is_empty() {
            panic!("Cannot select from empty list of enabled transitions/bindings");
        }

        // Group bindings by priority level
        let mut priorities: HashMap<i64, Vec<&(String, Binding)>> = HashMap::new();
        let default_priority = self.model.get_priority_level("P_NORMAL").unwrap_or(1000);

        for binding_pair in enabled {
            let transition = self.model.find_transition(&binding_pair.0)
                .expect("Enabled transition ID not found in model - internal error");
            // Use the transition's priority field, fall back to default if empty or not found
            let level = transition.priority.is_empty()
                .then(|| default_priority)
                .or_else(|| self.model.get_priority_level(&transition.priority))
                .unwrap_or(default_priority);

            priorities.entry(level).or_default().push(binding_pair);
        }

        // Find the lowest priority level (highest priority) among the enabled ones
        let lowest_level = priorities.keys().min().copied()
            .expect("Internal error: Failed to find minimum priority level");

        let highest_priority_options = priorities.get(&lowest_level)
            .expect("Internal error: Lowest priority level not found in map");

        // --- Random Choice (Non-deterministic) ---
        // Get a thread-local random number generator
        let mut rng = rng();
        // Choose a random element using the SliceRandom trait from the prelude
        let chosen_pair = highest_priority_options.choose(&mut rng)
            .expect("Internal error: Failed to choose from non-empty priority options");

        // --- Deterministic Choice (Commented out) ---
        // let chosen_pair = highest_priority_options.first()
        //      .expect("Internal error: Failed to get first element from non-empty priority options");


        // Clone the selected transition ID and the Binding struct
        (chosen_pair.0.clone(), chosen_pair.1.clone())
    }

    // --- New Public Method ---
    /// Returns the current marking (list of tokens) for a given place ID.
    /// Returns `None` if the place ID is not found.
    pub fn get_marking(&self, place_id: &str) -> Option<&Vec<Dynamic>> {
        self.current_marking.get(place_id)
    }

    /// Returns the current marking (list of tokens) for all places.
    pub fn get_all_markings(&self) -> &HashMap<String, Vec<Dynamic>> {
        &self.current_marking
    }

}
