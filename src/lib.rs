mod utils;
mod model;
mod simulator;

pub use model::{PetriNetData, FiringEventData}; // Correctly export FiringEventData from model
pub use simulator::Simulator;

use rhai::Dynamic; // Import Dynamic
use std::collections::HashMap; // Import HashMap

// Conditionally include the wasm module only when targeting wasm32
cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub mod wasm;
    }
}

/// The core logic struct, no wasm specifics here.
pub struct CoreSimulator {
    simulator: Simulator,
}

impl CoreSimulator {
    // Change return type to use String for error to avoid Send/Sync issues
    pub fn new(ocpn_json_string: &str) -> Result<Self, String> {
        // Map serde error to String
        let model_data: PetriNetData = serde_json::from_str(ocpn_json_string)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        // Map Rhai error to String
        let simulator = Simulator::new(model_data)
            .map_err(|e| format!("Failed to initialize simulator: {}", e))?;

        Ok(CoreSimulator { simulator })
    }

    // Update return type to use FiringEventData
    pub fn run_step(&mut self) -> Option<FiringEventData> {
        self.simulator.run_step()
    }

    /// Returns a reference to the entire current marking map.
    /// Keys are place IDs, values are vectors of tokens (Dynamic).
    pub fn get_all_markings(&self) -> &HashMap<String, Vec<Dynamic>> {
        self.simulator.get_all_markings()
    }

    /// Get the list of currently enabled transitions.
    /// Returns a vector of (transition_id, transition_name) pairs.
    pub fn get_enabled_transitions(&self) -> Vec<(String, String)> {
        self.simulator.get_enabled_transitions()
    }

    /// Fire a specific transition by ID.
    /// Returns the firing event data if successful, None if the transition is not enabled.
    pub fn fire_transition(&mut self, transition_id: &str) -> Option<FiringEventData> {
        self.simulator.fire_transition(transition_id)
    }

    /// Get the current simulation time in milliseconds
    pub fn get_current_time(&self) -> i64 {
        self.simulator.get_current_time()
    }

    /// Set the current simulation time in milliseconds
    pub fn set_current_time(&mut self, time: i64) {
        self.simulator.set_current_time(time)
    }

    /// Get the simulation epoch (ISO 8601 string)
    pub fn get_simulation_epoch(&self) -> Option<&String> {
        self.simulator.get_simulation_epoch()
    }

    /// Set the simulation epoch (ISO 8601 string)
    pub fn set_simulation_epoch(&mut self, epoch: Option<String>) {
        self.simulator.set_simulation_epoch(epoch)
    }

    /// Advance simulation time by a given delta in milliseconds
    pub fn advance_time(&mut self, delta_ms: i64) {
        self.simulator.advance_time(delta_ms)
    }
}
