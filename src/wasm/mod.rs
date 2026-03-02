use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use crate::CoreSimulator;
use crate::monitor::MonitorConfig;
use js_sys::{Function, Array};
use serde_wasm_bindgen::{to_value, from_value};
use std::collections::HashMap;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub struct WasmSimulator {
    simulator: CoreSimulator,
    event_listener: Option<Function>,
}

#[wasm_bindgen]
impl WasmSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(ocpn_json_string: &str) -> Result<WasmSimulator, JsValue> {
        console_log!("[WASM] Creating simulator...");
        let simulator = CoreSimulator::new(ocpn_json_string)
            .map_err(|e| JsValue::from_str(&format!("Initialization error: {}", e)))?;
        
        console_log!("[WASM] Simulator created, current marking: {:?}", simulator.get_all_markings());
        Ok(WasmSimulator { simulator, event_listener: None })
    }

    #[wasm_bindgen(js_name = setEventListener)]
    pub fn set_event_listener(&mut self, listener: Function) {
        self.event_listener = Some(listener);
    }

    #[wasm_bindgen]
    pub fn run_step(&mut self) -> Result<JsValue, JsValue> {
        console_log!("[WASM] run_step called, current marking: {:?}", self.simulator.get_all_markings());
        match self.simulator.run_step() {
            Some(event_data) => {
                console_log!("[WASM] Transition fired: {:?}", event_data.transition_id);
                let event_data_js = to_value(&event_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;

                if let Some(ref listener) = self.event_listener {
                    let _ = listener.call1(&JsValue::NULL, &event_data_js);
                }

                Ok(event_data_js)
            }
            None => {
                console_log!("[WASM] No transition enabled");
                Ok(JsValue::NULL)
            }
        }
    }

    /// Run multiple steps without calling the event listener for each step.
    /// Returns an array of all event data from the executed steps.
    /// Useful for fast-forward simulation without intermediate UI updates.
    #[wasm_bindgen(js_name = runMultipleSteps)]
    pub fn run_multiple_steps(&mut self, steps: u32) -> Result<JsValue, JsValue> {
        console_log!("[WASM] run_multiple_steps called with {} steps, current marking: {:?}", steps, self.simulator.get_all_markings());
        
        let results = Array::new();
        let mut executed_count = 0u32;
        
        for _ in 0..steps {
            match self.simulator.run_step() {
                Some(event_data) => {
                    let event_data_js = to_value(&event_data)
                        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;
                    results.push(&event_data_js);
                    executed_count += 1;
                }
                None => {
                    console_log!("[WASM] No more transitions enabled after {} steps", executed_count);
                    break;
                }
            }
        }
        
        console_log!("[WASM] Executed {} steps", executed_count);
        Ok(results.into())
    }

    /// Get the list of currently enabled transitions with their possible bindings.
    /// Returns an array of objects with transitionId and transitionName.
    #[wasm_bindgen(js_name = getEnabledTransitions)]
    pub fn get_enabled_transitions(&self) -> Result<JsValue, JsValue> {
        console_log!("[WASM] get_enabled_transitions called");
        let enabled = self.simulator.get_enabled_transitions();
        let result = Array::new();
        
        for (transition_id, transition_name) in enabled {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("transitionId"), &JsValue::from_str(&transition_id))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("transitionName"), &JsValue::from_str(&transition_name))?;
            result.push(&obj);
        }
        
        Ok(result.into())
    }

    /// Fire a specific transition by its ID.
    /// Returns the event data if the transition was successfully fired, null otherwise.
    #[wasm_bindgen(js_name = fireTransition)]
    pub fn fire_transition(&mut self, transition_id: &str) -> Result<JsValue, JsValue> {
        console_log!("[WASM] fire_transition called for: {}", transition_id);
        
        match self.simulator.fire_transition(transition_id) {
            Some(event_data) => {
                console_log!("[WASM] Transition {} fired successfully", transition_id);
                let event_data_js = to_value(&event_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;

                if let Some(ref listener) = self.event_listener {
                    let _ = listener.call1(&JsValue::NULL, &event_data_js);
                }

                Ok(event_data_js)
            }
            None => {
                console_log!("[WASM] Transition {} is not enabled", transition_id);
                Ok(JsValue::NULL)
            }
        }
    }

    /// Get the current simulation time in milliseconds
    #[wasm_bindgen(js_name = getCurrentTime)]
    pub fn get_current_time(&self) -> i64 {
        self.simulator.get_current_time()
    }

    /// Set the current simulation time in milliseconds
    #[wasm_bindgen(js_name = setCurrentTime)]
    pub fn set_current_time(&mut self, time: i64) {
        self.simulator.set_current_time(time);
    }

    /// Get the simulation epoch as an ISO 8601 string (or null if not set)
    #[wasm_bindgen(js_name = getSimulationEpoch)]
    pub fn get_simulation_epoch(&self) -> JsValue {
        match self.simulator.get_simulation_epoch() {
            Some(epoch) => JsValue::from_str(epoch),
            None => JsValue::NULL,
        }
    }

    /// Set the simulation epoch (ISO 8601 string, or null to clear)
    #[wasm_bindgen(js_name = setSimulationEpoch)]
    pub fn set_simulation_epoch(&mut self, epoch: Option<String>) {
        self.simulator.set_simulation_epoch(epoch);
    }

    /// Advance simulation time by a given delta in milliseconds
    #[wasm_bindgen(js_name = advanceTime)]
    pub fn advance_time(&mut self, delta_ms: i64) {
        self.simulator.advance_time(delta_ms);
    }

    // ========================================================================
    // Monitor API
    // ========================================================================

    /// Register a monitor from a JSON config object.
    /// The config should match the MonitorConfig schema:
    /// { id, name, type, enabled, placeIds, transitionIds,
    ///   observationScript, predicateScript, stopCondition }
    #[wasm_bindgen(js_name = addMonitor)]
    pub fn add_monitor(&mut self, config_js: JsValue) -> Result<(), JsValue> {
        let config: MonitorConfig = serde_wasm_bindgen::from_value(config_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid monitor config: {}", e)))?;
        console_log!("[WASM] Adding monitor '{}' (type: {:?})", config.name, config.monitor_type);
        self.simulator
            .add_monitor(config)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Remove a monitor by its ID.
    #[wasm_bindgen(js_name = removeMonitor)]
    pub fn remove_monitor(&mut self, id: &str) {
        console_log!("[WASM] Removing monitor '{}'", id);
        self.simulator.remove_monitor(id);
    }

    /// Get all monitor results as a JSON array.
    /// Each result: { monitorId, monitorName, monitorType, observations, statistics, breakpointHit }
    #[wasm_bindgen(js_name = getMonitorResults)]
    pub fn get_monitor_results(&mut self) -> Result<JsValue, JsValue> {
        let results = self.simulator.get_monitor_results();
        to_value(&results).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Clear all monitor results (keeps definitions).
    #[wasm_bindgen(js_name = clearMonitorResults)]
    pub fn clear_monitor_results(&mut self) {
        console_log!("[WASM] Clearing monitor results");
        self.simulator.clear_monitor_results();
    }

    /// Check if any breakpoint monitor was triggered.
    #[wasm_bindgen(js_name = hasBreakpointHit)]
    pub fn has_breakpoint_hit(&self) -> bool {
        self.simulator.has_breakpoint_hit()
    }

    /// Reset step counter and monitor results.
    #[wasm_bindgen(js_name = resetMonitors)]
    pub fn reset_monitors(&mut self) {
        console_log!("[WASM] Resetting monitors");
        self.simulator.reset_monitors();
    }

    // ========================================================================
    // State Space Analysis
    // ========================================================================

    /// Calculate the state space (reachability graph).
    /// Returns a JSON object with `report` and `graph` fields.
    /// Options: maxStates (default 10000), maxArcs (default 50000), isTimed (default false)
    /// Optional `distOverrides` is a JS object like { "exponential": 5.0, "normal": 10.0 }
    /// Optional `intRangeOverrides` is a JS object like { "varName": 50 }
    #[wasm_bindgen(js_name = calculateStateSpace)]
    pub fn calculate_state_space(
        &mut self,
        max_states: Option<u32>,
        max_arcs: Option<u32>,
        is_timed: Option<bool>,
        dist_overrides: JsValue,
        int_range_overrides: JsValue,
    ) -> Result<JsValue, JsValue> {
        let max_s = max_states.unwrap_or(10_000);
        let max_a = max_arcs.unwrap_or(50_000);
        let timed = is_timed.unwrap_or(false);

        // Parse optional deterministic overrides from JS objects
        let dist_ov: Option<HashMap<String, f64>> = if dist_overrides.is_undefined() || dist_overrides.is_null() {
            None
        } else {
            from_value(dist_overrides).ok()
        };
        let int_range_ov: Option<HashMap<String, i64>> = if int_range_overrides.is_undefined() || int_range_overrides.is_null() {
            None
        } else {
            from_value(int_range_overrides).ok()
        };

        let has_overrides = dist_ov.is_some() || int_range_ov.is_some();
        console_log!(
            "[WASM] Calculating state space (maxStates={}, maxArcs={}, timed={}, deterministic={})",
            max_s, max_a, timed, has_overrides
        );
        let result = self.simulator.calculate_state_space(max_s, max_a, timed, dist_ov, int_range_ov);
        console_log!(
            "[WASM] State space: {} states, {} arcs, {} SCCs (full={})",
            result.report.num_states,
            result.report.num_arcs,
            result.report.num_scc,
            result.report.is_full
        );
        // Serialize via JSON to guarantee BTreeMap<String, …> becomes a plain
        // JS object (serde_wasm_bindgen::to_value turns maps into JS Map
        // instances which break Object.entries on the frontend).
        let json_str = serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization error: {}", e)))?;
        js_sys::JSON::parse(&json_str)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {:?}", e)))
    }
}
