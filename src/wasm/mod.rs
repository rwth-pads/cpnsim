use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;
use crate::CoreSimulator;
use js_sys::{Function, Array};
use serde_wasm_bindgen::to_value;

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
}
