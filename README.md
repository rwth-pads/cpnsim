# CPNsim - Colored Petri Net Simulator

> ⚠️🚧 **Note:** This library is a work in progress. We are prioritizing tasks as we go. Feel free to leave an issue with your feature request or bug report — your feedback is greatly appreciated! 🚀

CPNsim is a Rust library and command-line tool for simulating Colored Petri Nets (CPNs). It parses CPN models defined in a specific JSON format (`.ocpn`) and executes the simulation step by step, handling token consumption, guard evaluation, and token production based on arc inscriptions and variable bindings.

## Features

*   Parses `.ocpn` JSON format for CPN models.
*   Supports basic CPN elements: Places, Transitions, Arcs.
*   Handles Color Sets (basic types like INT, STRING, etc.).
*   Supports Variables and binding during transition firing.
*   Evaluates transition guards using the Rhai scripting language.
*   Evaluates arc inscriptions (also using Rhai) for token production.
*   Provides a library interface for integration into other Rust projects.
*   Includes a debug executable for running simulations from the command line.
*   Compiles to WebAssembly (WASM) for use in web environments.

## Building the Project

You can build the project using standard Cargo commands:

```bash
# Build for development (debug mode)
cargo run
# Open a specific file in debug mode
cargo run -- examples/petrinet-2.ocpn

# Build for release (WebAssembly)
wasm-pack build --scope rwth-pads --target web --features wasm
