use cpnsim::{CoreSimulator, MonitorConfig, MonitorType};

#[test]
fn interval_duration_monitor_records_elapsed_time_between_transitions() {
    let model = r##"
    {
      "petriNets": [
        {
          "id": "net",
          "name": "Monitor Test",
          "places": [
            {
              "id": "p-ready",
              "name": "Ready",
              "colorSet": "Aircraft",
              "initialMarking": "[{\"id\": 1}]",
              "marking": "",
              "position": { "x": 0, "y": 0 },
              "size": { "width": 60, "height": 40 }
            },
            {
              "id": "p-fueling",
              "name": "Fueling",
              "colorSet": "Aircraft",
              "initialMarking": "",
              "marking": "",
              "position": { "x": 100, "y": 0 },
              "size": { "width": 60, "height": 40 }
            },
            {
              "id": "p-done",
              "name": "Done",
              "colorSet": "Aircraft",
              "initialMarking": "",
              "marking": "",
              "position": { "x": 200, "y": 0 },
              "size": { "width": 60, "height": 40 }
            }
          ],
          "transitions": [
            {
              "id": "t-fuel-start",
              "name": "Fueling Start",
              "guard": "",
              "time": "",
              "priority": "",
              "codeSegment": "",
              "position": { "x": 50, "y": 0 },
              "size": { "width": 60, "height": 40 }
            },
            {
              "id": "t-fuel-end",
              "name": "Fueling End",
              "guard": "",
              "time": "delay_ms(1500)",
              "priority": "",
              "codeSegment": "",
              "position": { "x": 150, "y": 0 },
              "size": { "width": 60, "height": 40 }
            }
          ],
          "arcs": [
            { "id": "a1", "source": "p-ready", "target": "t-fuel-start", "inscription": "ac" },
            { "id": "a2", "source": "t-fuel-start", "target": "p-fueling", "inscription": "ac" },
            { "id": "a3", "source": "p-fueling", "target": "t-fuel-end", "inscription": "ac" },
            { "id": "a4", "source": "t-fuel-end", "target": "p-done", "inscription": "ac" }
          ]
        }
      ],
      "colorSets": [
        { "id": "cs-aircraft", "name": "Aircraft", "type": "record", "definition": "colset Aircraft = record id: INT timed;", "color": "#888888", "timed": true }
      ],
      "variables": [
        { "id": "v-ac", "name": "ac", "colorSet": "Aircraft" }
      ],
      "priorities": [],
      "functions": [],
      "uses": []
    }
    "##;

    let mut simulator = CoreSimulator::new(model).expect("model should initialize");
    simulator
        .add_monitor(MonitorConfig {
            id: "m-fueling-duration".to_string(),
            name: "Fueling Duration".to_string(),
            monitor_type: MonitorType::IntervalDuration,
            enabled: true,
            place_ids: vec![],
            transition_ids: vec![],
            observation_script: String::new(),
            predicate_script: String::new(),
            stop_condition: None,
            start_transition_id: Some("t-fuel-start".to_string()),
            end_transition_id: Some("t-fuel-end".to_string()),
            correlation_key: "ac.id".to_string(),
        })
        .expect("monitor should register");

    assert_eq!(simulator.run_step().unwrap().transition_id, "t-fuel-start");
    assert_eq!(simulator.run_step().unwrap().transition_id, "t-fuel-end");

    let results = simulator.get_monitor_results();
    let result = results
        .iter()
        .find(|result| result.monitor_id == "m-fueling-duration")
        .expect("monitor result should exist");

    assert_eq!(result.observations.len(), 1);
    assert_eq!(result.observations[0].value, 1500.0);
    assert_eq!(result.statistics.avg, 1500.0);
}
