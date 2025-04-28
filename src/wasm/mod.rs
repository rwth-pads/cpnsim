use wasm_bindgen::prelude::*;
use crate::CoreSimulator;
use js_sys::Function;
use serde_wasm_bindgen::to_value;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! _console_log {
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
        let simulator = CoreSimulator::new(ocpn_json_string)
            .map_err(|e| JsValue::from_str(&format!("Initialization error: {}", e)))?;
        
        Ok(WasmSimulator { simulator, event_listener: None })
    }

    #[wasm_bindgen(js_name = setEventListener)]
    pub fn set_event_listener(&mut self, listener: Function) {
        self.event_listener = Some(listener);
    }

    #[wasm_bindgen]
    pub fn run_step(&mut self) -> Result<JsValue, JsValue> {
        match self.simulator.run_step() {
            Some(event_data) => {
                let event_data_js = to_value(&event_data)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;

                if let Some(listener) = &self.event_listener {
                    listener.call1(&JsValue::NULL, &event_data_js).ok();
                }

                Ok(event_data_js)
            }
            None => Ok(JsValue::NULL),
        }
    }
}
