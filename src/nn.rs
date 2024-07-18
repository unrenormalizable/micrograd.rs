use crate::engine::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::iter::zip;

pub trait Module {
    fn zero_grad(&self) {
        self.parameters().iter().for_each(|v| v.reset_grad())
    }

    fn parameters(&self) -> Vec<Value>;
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(rng: &mut StdRng, nin: usize, nonlin: bool) -> Self {
        let w = (0..nin)
            .map(|_| rng.gen_range(-1.0..1.0))
            .map(Value::new)
            .collect();
        Self {
            w,
            b: Value::new(Default::default()),
            nonlin,
        }
    }

    pub fn run(&self, x: Vec<Value>) -> Value {
        assert!(x.len() == self.w.len(), "Mismatched input dimensions.");

        let sum = zip(&self.w, x)
            .map(|(wi, xi)| wi.clone() * xi)
            .reduce(|v1, v2| v1 + v2)
            .unwrap();

        let sum = sum + self.b.clone();

        if self.nonlin {
            sum.relu()
        } else {
            sum
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());

        params
    }
}

impl std::fmt::Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Neuron({})",
            if self.nonlin { "ReLU" } else { "Linear" },
            self.w.len()
        ))?;

        Ok(())
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(rng: &mut StdRng, nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(rng, nin, nonlin)).collect();

        Self { neurons }
    }

    pub fn run(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.run(x.clone())).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|x| x.parameters()).collect()
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = self
            .neurons
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        f.write_fmt(format_args!("Layer of [{}]", desc))?;

        Ok(())
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(rng: &mut StdRng, nin: usize, nouts: Vec<usize>) -> Self {
        let mut ls = vec![nin];
        ls.append(&mut nouts.clone());

        let layers = (0..nouts.len())
            .map(|n| Layer::new(rng, ls[n], ls[n + 1], n != nouts.len() - 1))
            .collect();

        Self { layers }
    }

    pub fn run(&self, x: Vec<Value>) -> Vec<Value> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.run(x);
        }

        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|x| x.parameters()).collect()
    }
}

impl std::fmt::Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = self
            .layers
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        f.write_fmt(format_args!("MLP of [{}]", desc))?;

        Ok(())
    }
}
