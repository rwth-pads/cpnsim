use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rhai::Dynamic; // Add missing import

// --- Core Petri Net Structure ---

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields as it's a data structure
pub struct PetriNetData {
    pub petri_nets: Vec<PetriNet>,
    pub color_sets: Vec<ColorSet>,
    pub variables: Vec<Variable>,
    pub priorities: Vec<Priority>,
    pub functions: Vec<FunctionDefinition>,
    pub uses: Vec<UseDefinition>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct PetriNet {
    pub id: String,
    pub name: String,
    pub places: Vec<Place>,
    pub transitions: Vec<Transition>,
    pub arcs: Vec<Arc>,
}

// --- Components ---

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct Place {
    pub id: String,
    pub name: String,
    pub color_set: String, // Name of the color set
    pub initial_marking: String, // Rhai expression string or empty
    #[serde(default)] // Handle cases where marking might be absent or null initially
    pub marking: String, // Placeholder type for marking
    pub position: Position,
    pub size: Size,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct Transition {
    pub id: String,
    pub name: String,
    #[serde(default)] // Handle optional fields
    pub guard: String, // Rhai expression string or empty
    #[serde(default)]
    pub time: String, // Placeholder for potential timed CPNs
    #[serde(default)]
    pub priority: String, // Name of the priority level or empty
    pub position: Position,
    pub size: Size,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct Arc {
    pub id: String,
    pub source: String, // ID of source node (Place or Transition)
    pub target: String, // ID of target node (Place or Transition)
    pub inscription: String, // Rhai expression string
}

// --- Declarations ---

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct ColorSet {
    pub id: String, // Consider using Uuid if parsing from string needed
    pub name: String,
    #[serde(rename = "type")] // Handle keyword conflict
    pub type_field: String, // e.g., "basic", "product", "record", "list", "union"
    pub definition: String, // e.g., "colset INT = int;"
    pub color: String, // UI color hex code
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct Variable {
    pub id: String, // Consider using Uuid
    pub name: String,
    pub color_set: String, // Name of the color set
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct Priority {
    pub id: String, // Consider using Uuid
    pub name: String,
    pub level: i64, // Using i64 for priority level
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct FunctionDefinition {
    pub id: String, // Consider using Uuid
    pub name: String,
    pub code: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)] // Allow unused fields
pub struct UseDefinition {
    pub id: String, // Consider using Uuid
    pub name: String, // Filename or identifier
    pub content: String, // SML or potentially Rhai code content
}

// --- Geometry ---

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Allow unused fields
pub struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)] // Allow unused fields
pub struct Size {
    pub width: f64,
    pub height: f64,
}

// --- Event Data Structure ---

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FiringEventData {
    pub transition_id: String,
    pub transition_name: String,
    // Change back to Dynamic
    pub consumed: HashMap<String, Vec<Dynamic>>,
    // Change back to Dynamic
    pub produced: HashMap<String, Vec<Dynamic>>,
    // TODO: Add binding information if needed (variable name -> bound value)
}

// --- Helper Functions (Example - Needs Implementation) ---

impl PetriNetData {
    // Potential helper to get a place by ID across all subnets
    #[allow(dead_code)] // Allow unused method
    pub fn find_place(&self, id: &str) -> Option<&Place> {
        self.petri_nets.iter().find_map(|net| net.places.iter().find(|p| p.id == id))
    }

    // Potential helper to get a transition by ID
    pub fn find_transition(&self, id: &str) -> Option<&Transition> {
         self.petri_nets.iter().find_map(|net| net.transitions.iter().find(|t| t.id == id))
    }

     // Potential helper to get priority level by name
    pub fn get_priority_level(&self, name: &str) -> Option<i64> {
        self.priorities.iter().find(|p| p.name == name).map(|p| p.level)
    }
}
