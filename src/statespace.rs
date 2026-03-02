use rhai::Dynamic;
use serde::Serialize;
use std::collections::{HashMap, HashSet, BTreeMap};

/// A canonical representation of a marking (used as hashable key for visited set).
/// We sort place IDs and within each place sort tokens by their string representation
/// to ensure deterministic equality.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalMarking {
    /// Sorted list of (place_id, sorted token strings)
    places: Vec<(String, Vec<String>)>,
    /// Current simulation time (relevant for timed nets)
    time: i64,
}

impl CanonicalMarking {
    pub fn from_marking(marking: &HashMap<String, Vec<Dynamic>>, time: i64, include_time: bool) -> Self {
        let mut places: Vec<(String, Vec<String>)> = marking
            .iter()
            .filter(|(_, tokens)| !tokens.is_empty())
            .map(|(place_id, tokens)| {
                let mut token_strs: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
                token_strs.sort();
                (place_id.clone(), token_strs)
            })
            .collect();
        places.sort_by(|a, b| a.0.cmp(&b.0));
        CanonicalMarking {
            places,
            time: if include_time { time } else { 0 },
        }
    }
}

/// A node in the state space graph.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateNode {
    /// Unique state ID (sequential: 1, 2, 3, ...)
    pub id: u32,
    /// Marking: place_id -> list of token strings
    pub marking: BTreeMap<String, Vec<String>>,
    /// Simulation time at this state
    pub time: i64,
}

/// An arc in the state space graph.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateArc {
    /// Source state ID
    pub from: u32,
    /// Target state ID
    pub to: u32,
    /// Transition ID that caused this state change
    pub transition_id: String,
    /// Transition name
    pub transition_name: String,
    /// Binding description (variable assignments)
    pub binding: String,
}

/// Bounds information for a single place.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaceBounds {
    pub place_id: String,
    pub place_name: String,
    pub upper_bound: usize,
    pub lower_bound: usize,
    /// Upper bound in terms of multi-set size
    pub upper_multi_set_bound: usize,
    pub lower_multi_set_bound: usize,
}

/// A strongly connected component.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SccComponent {
    pub id: u32,
    /// State IDs in this SCC
    pub states: Vec<u32>,
}

/// Full state space report (mirrors CPN Tools report format).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSpaceReport {
    /// Statistics
    pub num_states: u32,
    pub num_arcs: u32,
    pub num_scc: u32,
    /// Whether the state space exploration completed (vs hit a limit)
    pub is_full: bool,
    /// Exploration was stopped due to limit
    pub limit_reached: bool,
    /// Calculation time in milliseconds
    pub calc_time_ms: u64,

    /// Boundedness properties — per-place upper/lower bounds
    pub place_bounds: Vec<PlaceBounds>,

    /// Home properties — markings reachable from all other markings
    pub home_markings: Vec<u32>,

    /// Liveness properties
    /// Dead markings (no transitions enabled)
    pub dead_markings: Vec<u32>,
    /// Dead transition instances (never enabled in any reachable state)
    pub dead_transitions: Vec<String>,
    /// Live transitions (enabled in at least one reachable state)
    pub live_transitions: Vec<String>,

    /// Fairness — for each transition, how often it fires relative to others
    pub transition_fire_counts: Vec<TransitionFireCount>,

    /// SCC graph
    pub scc_graph: Vec<SccComponent>,
    /// Terminal SCC (no outgoing arcs to other SCCs)
    pub terminal_scc: Vec<u32>,
}

/// Transition firing statistics across the state space.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransitionFireCount {
    pub transition_id: String,
    pub transition_name: String,
    /// Number of arcs in the state space labeled with this transition
    pub fire_count: u32,
}

/// The full state space graph (for visualization).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSpaceGraph {
    pub nodes: Vec<StateNode>,
    pub arcs: Vec<StateArc>,
}

/// Result of state space calculation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSpaceResult {
    pub report: StateSpaceReport,
    pub graph: StateSpaceGraph,
}

/// Configuration for state space calculation.
#[derive(Debug, Clone)]
pub struct StateSpaceConfig {
    /// Maximum number of states to explore (0 = unlimited)
    pub max_states: u32,
    /// Maximum number of arcs (0 = unlimited)
    pub max_arcs: u32,
    /// Whether the net is timed (include time in state key)
    pub is_timed: bool,
    /// Maximum simulation time to explore for timed nets (0 = unlimited)
    pub max_time: i64,
}

impl Default for StateSpaceConfig {
    fn default() -> Self {
        StateSpaceConfig {
            max_states: 10_000,
            max_arcs: 50_000,
            is_timed: false,
            max_time: 0,
        }
    }
}

/// Analysis helper: compute the state space report from nodes and arcs.
pub fn compute_report(
    nodes: &[StateNode],
    arcs: &[StateArc],
    is_full: bool,
    limit_reached: bool,
    calc_time_ms: u64,
    all_transition_ids: &[(String, String)], // (id, name)
    explored: &HashSet<u32>, // states that were actually expanded during BFS
) -> StateSpaceReport {
    let num_states = nodes.len() as u32;
    let num_arcs = arcs.len() as u32;

    // --- Boundedness: per-place bounds ---
    let mut place_bounds_map: HashMap<String, (usize, usize)> = HashMap::new(); // place_id -> (max, min)
    // Collect all place IDs from all states
    let mut all_place_ids: HashSet<String> = HashSet::new();
    for node in nodes {
        for pid in node.marking.keys() {
            all_place_ids.insert(pid.clone());
        }
    }
    // Initialize bounds
    for pid in &all_place_ids {
        place_bounds_map.insert(pid.clone(), (0, usize::MAX));
    }
    // Compute bounds across all states
    for node in nodes {
        for pid in &all_place_ids {
            let count = node.marking.get(pid).map(|v| v.len()).unwrap_or(0);
            let entry = place_bounds_map.get_mut(pid).unwrap();
            entry.0 = entry.0.max(count); // upper bound
            entry.1 = entry.1.min(count); // lower bound
        }
    }
    let mut place_bounds: Vec<PlaceBounds> = place_bounds_map
        .into_iter()
        .map(|(pid, (upper, lower))| PlaceBounds {
            place_id: pid.clone(),
            place_name: pid, // TODO: use actual place name if available
            upper_bound: upper,
            lower_bound: if lower == usize::MAX { 0 } else { lower },
            upper_multi_set_bound: upper,
            lower_multi_set_bound: if lower == usize::MAX { 0 } else { lower },
        })
        .collect();
    place_bounds.sort_by(|a, b| a.place_id.cmp(&b.place_id));

    // --- Dead markings (states with no outgoing arcs) ---
    // Only consider states that were actually explored (expanded during BFS).
    // Frontier states (discovered but not yet expanded due to limits) are excluded
    // to avoid false dead markings in truncated state spaces.
    let states_with_outgoing: HashSet<u32> = arcs.iter().map(|a| a.from).collect();
    let dead_markings: Vec<u32> = nodes
        .iter()
        .filter(|n| explored.contains(&n.id) && !states_with_outgoing.contains(&n.id))
        .map(|n| n.id)
        .collect();

    // --- Transition liveness ---
    let transitions_that_fire: HashSet<String> = arcs.iter().map(|a| a.transition_id.clone()).collect();
    let dead_transitions: Vec<String> = all_transition_ids
        .iter()
        .filter(|(tid, _)| !transitions_that_fire.contains(tid))
        .map(|(tid, _)| tid.clone())
        .collect();
    let live_transitions: Vec<String> = all_transition_ids
        .iter()
        .filter(|(tid, _)| transitions_that_fire.contains(tid))
        .map(|(tid, _)| tid.clone())
        .collect();

    // --- Transition fire counts ---
    let mut fire_counts: HashMap<String, u32> = HashMap::new();
    for arc in arcs {
        *fire_counts.entry(arc.transition_id.clone()).or_insert(0) += 1;
    }
    let transition_fire_counts: Vec<TransitionFireCount> = all_transition_ids
        .iter()
        .map(|(tid, tname)| TransitionFireCount {
            transition_id: tid.clone(),
            transition_name: tname.clone(),
            fire_count: fire_counts.get(tid).copied().unwrap_or(0),
        })
        .collect();

    // --- SCC computation (Tarjan's algorithm) ---
    // Only include explored states in SCC analysis. Frontier states (discovered but
    // not expanded due to limits) have incomplete successor information and would
    // create many false trivial terminal SCCs.
    let explored_nodes: Vec<&StateNode> = nodes.iter()
        .filter(|n| explored.contains(&n.id))
        .collect();
    let explored_arcs: Vec<&StateArc> = arcs.iter()
        .filter(|a| explored.contains(&a.from) && explored.contains(&a.to))
        .collect();
    let (scc_graph, terminal_scc) = compute_scc(&explored_nodes, &explored_arcs);
    let num_scc = scc_graph.len() as u32;

    // --- Home markings ---
    // A marking M is a home marking if it's reachable from every other marking.
    // In practice: states that belong to a single terminal SCC (if there's exactly one terminal SCC)
    let home_markings = if terminal_scc.len() == 1 {
        let terminal_scc_id = terminal_scc[0];
        scc_graph
            .iter()
            .find(|scc| scc.id == terminal_scc_id)
            .map(|scc| scc.states.clone())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    StateSpaceReport {
        num_states,
        num_arcs,
        num_scc,
        is_full,
        limit_reached,
        calc_time_ms,
        place_bounds,
        home_markings,
        dead_markings,
        dead_transitions,
        live_transitions,
        transition_fire_counts,
        scc_graph,
        terminal_scc,
    }
}

/// Compute strongly connected components using Tarjan's algorithm.
fn compute_scc(nodes: &[&StateNode], arcs: &[&StateArc]) -> (Vec<SccComponent>, Vec<u32>) {
    let n = nodes.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Build adjacency list
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for node in nodes {
        adj.entry(node.id).or_default();
    }
    for arc in arcs {
        adj.entry(arc.from).or_default().push(arc.to);
    }

    // Tarjan's algorithm
    let mut index_counter: u32 = 0;
    let mut stack: Vec<u32> = Vec::new();
    let mut on_stack: HashSet<u32> = HashSet::new();
    let mut indices: HashMap<u32, u32> = HashMap::new();
    let mut lowlinks: HashMap<u32, u32> = HashMap::new();
    let mut sccs: Vec<SccComponent> = Vec::new();
    let mut scc_id_counter: u32 = 0;

    fn strongconnect(
        v: u32,
        adj: &HashMap<u32, Vec<u32>>,
        index_counter: &mut u32,
        stack: &mut Vec<u32>,
        on_stack: &mut HashSet<u32>,
        indices: &mut HashMap<u32, u32>,
        lowlinks: &mut HashMap<u32, u32>,
        sccs: &mut Vec<SccComponent>,
        scc_id_counter: &mut u32,
    ) {
        indices.insert(v, *index_counter);
        lowlinks.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        if let Some(neighbors) = adj.get(&v) {
            for &w in neighbors {
                if !indices.contains_key(&w) {
                    strongconnect(w, adj, index_counter, stack, on_stack, indices, lowlinks, sccs, scc_id_counter);
                    let low_w = *lowlinks.get(&w).unwrap();
                    let low_v = lowlinks.get_mut(&v).unwrap();
                    *low_v = (*low_v).min(low_w);
                } else if on_stack.contains(&w) {
                    let idx_w = *indices.get(&w).unwrap();
                    let low_v = lowlinks.get_mut(&v).unwrap();
                    *low_v = (*low_v).min(idx_w);
                }
            }
        }

        if lowlinks.get(&v) == indices.get(&v) {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            component.sort();
            sccs.push(SccComponent {
                id: *scc_id_counter,
                states: component,
            });
            *scc_id_counter += 1;
        }
    }

    let all_ids: Vec<u32> = nodes.iter().map(|n| n.id).collect();
    for &v in &all_ids {
        if !indices.contains_key(&v) {
            strongconnect(
                v,
                &adj,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlinks,
                &mut sccs,
                &mut scc_id_counter,
            );
        }
    }

    // Determine terminal SCCs (no outgoing edges to states in other SCCs)
    let mut state_to_scc: HashMap<u32, u32> = HashMap::new();
    for scc in &sccs {
        for &state_id in &scc.states {
            state_to_scc.insert(state_id, scc.id);
        }
    }
    let mut scc_has_outgoing: HashSet<u32> = HashSet::new();
    for &arc in arcs {
        let from_scc = state_to_scc.get(&arc.from);
        let to_scc = state_to_scc.get(&arc.to);
        if let (Some(&from), Some(&to)) = (from_scc, to_scc) {
            if from != to {
                scc_has_outgoing.insert(from);
            }
        }
    }
    let terminal_scc: Vec<u32> = sccs
        .iter()
        .filter(|scc| !scc_has_outgoing.contains(&scc.id))
        .map(|scc| scc.id)
        .collect();

    (sccs, terminal_scc)
}
