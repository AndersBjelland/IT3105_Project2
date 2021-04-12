use std::fmt::Debug;

use log::LevelFilter;
use mcts::PolicyKind;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde;
use serde_json;
use simplelog::{Config, WriteLogger};
use tensorflow as tf;

use crate::{
    encoder::{self, Encoder},
    hex::Hex,
    mcts::{self, LeafEvaluator},
    runtime::{PolicyEvaluation, Runtime},
};

#[pyclass]
#[derive(Default, Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SampleSet {
    #[pyo3(get)]
    pub state_shape: Vec<u64>,
    #[pyo3(get)]
    pub samples: Vec<Sample>,
}

impl SampleSet {
    pub fn from<E: Encoder>(samples: &Vec<mcts::Sample>) -> Self {
        let state_shape = samples
            .first()
            .map(|sample| E::shape(sample.state.size))
            .unwrap_or_default();
        let samples = samples
            .into_iter()
            .map(|sample| {
                let encoded = E::encoded(&sample.state);
                let shape = encoded.shape;
                let state = encoded.data;

                assert!(shape == state_shape);

                let value = match sample.state.current == sample.winner {
                    true => 1.0,
                    false => -1.0,
                };

                let mut policy = vec![0.0; sample.state.size * sample.state.size];

                for &(position, probability) in &sample.distribution {
                    let index = PolicyEvaluation::get_index(
                        sample.state.size,
                        &position,
                        &sample.state.current,
                    );
                    policy[index] = probability as f32;
                }

                Sample {
                    bitstate: (sample.state.player1, sample.state.player2),
                    state,
                    policy,
                    value,
                }
            })
            .collect();

        SampleSet {
            state_shape,
            samples,
        }
    }
}

#[pyclass]
#[derive(Default, Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sample {
    #[pyo3(get)]
    pub bitstate: (u64, u64),
    #[pyo3(get)]
    pub state: Vec<f32>,
    #[pyo3(get)]
    pub policy: Vec<f32>,
    #[pyo3(get)]
    pub value: f64,
}

#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("hello_world".to_string())
}

#[pyfunction]
fn init_logging(level: &str) -> PyResult<()> {
    let log_level = match level {
        "debug" => LevelFilter::Debug,
        "error" => LevelFilter::Error,
        "info" => LevelFilter::Info,
        "off" => LevelFilter::Off,
        "trace" => LevelFilter::Trace,
        "warn" => LevelFilter::Warn,
        _ => panic!("log level must be one of 'debug', 'error', 'info', 'off', 'trace', 'warn'"),
    };

    WriteLogger::init(
        log_level,
        Config::default(),
        std::fs::File::create("self-play.log").unwrap(),
    )
    .unwrap();

    Ok(())
}

fn generate_samples<E: Encoder + Debug, L: LeafEvaluator<E>>(
    graph: tf::Graph,
    smb: tf::SavedModelBundle,
    size: usize,
    concurrents: usize,
    config: mcts::Config,
    samples: usize,
) -> PyResult<SampleSet> {
    let mut runtime = Runtime::<E, L>::new(
        graph,
        smb.meta_graph_def().clone(),
        smb.session,
        Hex::empty(size, ()),
        concurrents,
        config,
    )
    .expect("failed to init runtime");

    Ok(SampleSet::from::<E>(&runtime.generate(samples)))
}

#[pyfunction]
pub fn run(
    leaf_evaluation: &str,
    encoder: &str,
    path: &str,
    size: usize,
    concurrents: usize,
    samples: usize,
    simulations: usize,
    c: Option<f64>,
    epsilon: Option<f64>,
    policy_kind: Option<String>,
) -> PyResult<SampleSet> {
    let mut graph = tf::Graph::new();
    let smb = tf::SavedModelBundle::load(&tf::SessionOptions::new(), &["serve"], &mut graph, path)
        .unwrap();

    let policy_kind = match policy_kind.as_ref().map(|x| x.as_str()) {
        Some("proportionate") => PolicyKind::Proportionate,
        Some("greedy") => PolicyKind::Greedy,
        None => PolicyKind::Proportionate,
        _ => panic!("invalid policy_kind {}", policy_kind.unwrap()),
    };

    let config = mcts::Config {
        simulations,
        c: c.unwrap_or(3.0),
        epsilon: epsilon.unwrap_or(0.25),
        policy_kind,
    };

    match (encoder, leaf_evaluation) {
        ("normalized", "rollout") => generate_samples::<
            encoder::Normalized,
            mcts::Rollout<encoder::Normalized>,
        >(graph, smb, size, concurrents, config, samples),
        ("normalized", "value_fn") => generate_samples::<
            encoder::Normalized,
            mcts::ValueFunction<encoder::Normalized>,
        >(graph, smb, size, concurrents, config, samples),
        _ => Err(PyValueError::new_err(format!(
            "unknown (encoder, leaf_evaluation) combo ({}, {})",
            encoder, leaf_evaluation
        ))),
    }
}

#[pyfunction]
pub fn run_save(
    leaf_evaluation: &str,
    encoder: &str,
    model_path: &str,
    size: usize,
    concurrents: usize,
    samples: usize,
    simulations: usize,
    c: Option<f64>,
    epsilon: Option<f64>,
    policy_kind: Option<String>,
    out_path: &str,
) -> PyResult<()> {
    let mut graph = tf::Graph::new();
    let smb = tf::SavedModelBundle::load(
        &tf::SessionOptions::new(),
        &["serve"],
        &mut graph,
        model_path,
    )
    .unwrap();

    let policy_kind = match policy_kind.as_ref().map(|x| x.as_str()) {
        Some("proportionate") => PolicyKind::Proportionate,
        Some("greedy") => PolicyKind::Greedy,
        None => PolicyKind::Proportionate,
        _ => panic!("invalid policy_kind {}", policy_kind.unwrap()),
    };

    let config = mcts::Config {
        simulations,
        c: c.unwrap_or(3.0),
        epsilon: epsilon.unwrap_or(0.25),
        policy_kind,
    };

    let sample_set = match (encoder, leaf_evaluation) {
        ("normalized", "rollout") => generate_samples::<
            encoder::Normalized,
            mcts::Rollout<encoder::Normalized>,
        >(graph, smb, size, concurrents, config, samples),
        ("normalized", "value_fn") => generate_samples::<
            encoder::Normalized,
            mcts::ValueFunction<encoder::Normalized>,
        >(graph, smb, size, concurrents, config, samples),
        _ => Err(PyValueError::new_err(format!(
            "unknown (encoder, leaf_evaluation) combo ({}, {})",
            encoder, leaf_evaluation
        ))),
    }?;

    serde_json::to_writer(&std::fs::File::create(out_path).unwrap(), &sample_set).unwrap();
    Ok(())
}

#[pyfunction]
pub fn policy_distribution(
    leaf_evaluation: &str,
    encoder: &str,
    model_path: &str,
    size: usize,
    states: Vec<Vec<Vec<usize>>>,
    simulations: usize,
    c: Option<f64>,
    epsilon: Option<f64>,
    policy_kind: Option<String>,
) -> PyResult<Vec<Vec<((usize, usize), f64)>>> {
    let mut graph = tf::Graph::new();
    let smb = tf::SavedModelBundle::load(
        &tf::SessionOptions::new(),
        &["serve"],
        &mut graph,
        model_path,
    )
    .expect("could not load SavedModelBundle");

    let policy_kind = match policy_kind.as_ref().map(|x| x.as_str()) {
        Some("proportionate") => PolicyKind::Proportionate,
        Some("greedy") => PolicyKind::Greedy,
        None => PolicyKind::Proportionate,
        _ => panic!("invalid policy_kind {}", policy_kind.unwrap()),
    };

    let config = mcts::Config {
        simulations,
        c: c.unwrap_or(3.0),
        epsilon: epsilon.unwrap_or(0.25),
        policy_kind,
    };

    let states = states
        .into_iter()
        .map(|state| Hex::from_grid(state, size, ()).expect("invalid hex state"))
        .collect::<Vec<_>>();

    match (encoder, leaf_evaluation) {
        ("normalized", "rollout") => Ok(Runtime::<
            encoder::Normalized,
            mcts::Rollout<encoder::Normalized>,
        >::policy_distibution(
            size,
            &states,
            graph,
            smb.meta_graph_def().clone(),
            smb.session,
            config,
        )
        .unwrap()),
        ("normalized", "value_fn") => Ok(Runtime::<
            encoder::Normalized,
            mcts::ValueFunction<encoder::Normalized>,
        >::policy_distibution(
            size,
            &states,
            graph,
            smb.meta_graph_def().clone(),
            smb.session,
            config,
        )
        .unwrap()),
        _ => Err(PyValueError::new_err(format!(
            "unknown (encoder, leaf_evaluation) combo ({}, {})",
            encoder, leaf_evaluation
        ))),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn self_play(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_save, m)?)?;
    m.add_function(wrap_pyfunction!(policy_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}
