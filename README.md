# CPNsim

**CPNsim** is a Rust library and command-line tool for simulating **Colored Petri Nets (CPNs)**. It parses CPN models in the `.ocpn` JSON format and executes simulations step by step — handling token consumption, guard evaluation, and token production based on arc inscriptions and variable bindings.

> **Note:** This library is a work in progress. Feel free to leave an issue with your feature request or bug report — your feedback is greatly appreciated!

CPNsim powers the simulation engine behind [OCPN Tools](https://github.com/rwth-pads/ocpn-tools) and compiles to WebAssembly for in-browser execution.

## Features

### Core Simulation

- **CPN model parsing** — Reads `.ocpn` JSON format with support for places, transitions, and arcs
- **Color sets** — Basic types (`INT`, `STRING`, `BOOL`, `UNIT`), product types, record types, enumerated types, indexed types, subsets, lists, and `INT` with ranges
- **Variables & bindings** — Automatic variable binding during transition firing with support for multi-variable arcs
- **Guard evaluation** — Transition guards evaluated using the [Rhai](https://rhai.rs/) scripting engine
- **Arc inscriptions** — Input and output arc expressions evaluated via Rhai for flexible token manipulation
- **Timed simulation** — Time expressions, arc delays, and automatic time advancement for timed CPN models
- **Hierarchical nets** — Substitution transitions with port/socket bindings and fusion places
- **Priority levels** — Transition priority support for controlling firing order
- **Distribution functions** — 14 CPN Tools-compatible stochastic functions: `bernoulli`, `beta`, `binomial`, `chisq`, `discrete`, `erlang`, `exponential`, `gamma`, `normal`, `poisson`, `rayleigh`, `student`, `uniform`, `weibull`

### Monitors

- **Marking-size monitors** — Track token counts on watched places over simulation steps
- **Transition-count monitors** — Count firings of specific transitions
- **Breakpoint monitors** — Stop simulation when a place-based or transition-based predicate is met (e.g., place empty, transition fires)
- **Data collector monitors** — User-defined monitors with custom Rhai observation and predicate scripts
- **Statistics** — Each monitor computes running statistics: count, sum, average, min, max, and standard deviation

### State Space Analysis

- **BFS exploration** — Breadth-first state space construction with configurable limits (max states, max arcs)
- **Canonical markings** — Deterministic state deduplication; timed models include simulation time in the state key
- **SCC decomposition** — Tarjan's algorithm for strongly connected component detection
- **Report metrics** — Dead markings, home markings, terminal SCCs, place bounds (min/max tokens), dead/live transitions, and transition fairness (arc occurrence counts)
- **Timed state spaces** — Supports timed CPN models with configurable time bounds
- **Deterministic overrides** — Replace stochastic distribution functions and `IntRange` random bindings with constant values for finite, deterministic exploration

### Integration

- **Library interface** — Use as a Rust crate in other projects
- **WebAssembly** — Compiles to WASM via `wasm-pack` for browser-based simulation
- **Debug CLI** — Command-line executable for running simulations and inspecting model behavior

## Building

### Debug (CLI)

```bash
cargo run -- examples/petrinet-2.ocpn
```

### WebAssembly

```bash
wasm-pack build --scope rwth-pads --target web --release -- --features wasm
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
