use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

// TODO: Implement operator overloading without needing to wrap with Value;
// TODO: Make Value generic, can we used an Inner Type?
// TODO: Ratatui visualization
// TODO: On destruction clear up stuff on the global caches.

// https://medium.com/sfu-cspmp/diy-deep-learning-crafting-your-own-autograd-engine-from-scratch-for-effortless-backpropagation-ddab167faaf5
// NOTE: Safe Singleton Globals in Rust: https://stackoverflow.com/a/27826181/6196679

fn next_id_store() -> &'static Mutex<usize> {
    static ID: OnceLock<Mutex<usize>> = OnceLock::new();
    ID.get_or_init(|| Mutex::new(0))
}

fn next_id() -> usize {
    let global_state = next_id_store();
    let mut state = global_state.lock().unwrap();
    *state += 1;
    *state
}

fn grad_store() -> &'static Mutex<HashMap<usize, f64>> {
    static MAP: OnceLock<Mutex<HashMap<usize, f64>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn grad(id: usize) -> f64 {
    *grad_store().lock().unwrap().get(&id).unwrap()
}

fn set_grad(id: usize, grad: f64) {
    grad_store().lock().unwrap().insert(id, grad);
}

fn val_store() -> &'static Mutex<HashMap<usize, f64>> {
    static MAP: OnceLock<Mutex<HashMap<usize, f64>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn val(id: usize) -> f64 {
    *val_store().lock().unwrap().get(&id).unwrap()
}

fn set_val(id: usize, val: f64) {
    val_store().lock().unwrap().insert(id, val);
}

type OpInfo = (String, Vec<usize>);

fn op_info_store() -> &'static Mutex<HashMap<usize, OpInfo>> {
    static MAP: OnceLock<Mutex<HashMap<usize, OpInfo>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn op_info(id: usize) -> (String, Vec<usize>) {
    (*op_info_store().lock().unwrap().get(&id).unwrap()).clone()
}

type Backward = Option<fn(Vec<usize>, usize)>;

fn backward_store() -> &'static Mutex<HashMap<usize, Backward>> {
    static MAP: OnceLock<Mutex<HashMap<usize, Backward>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn backward(id: usize) -> Backward {
    *backward_store().lock().unwrap().get(&id).unwrap()
}

fn set_backward(id: usize, backward: Backward) {
    backward_store().lock().unwrap().insert(id, backward);
}

#[derive(Copy, Clone, Debug)]
pub struct Value {
    id: usize,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self::new_impl(data)
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn data(&self) -> f64 {
        val(self.id())
    }

    pub fn grad(&self) -> f64 {
        grad(self.id())
    }

    pub fn backward(&self) {
        let topo = &mut Vec::<usize>::new();
        let visited = &mut HashSet::<usize>::new();
        Self::build_topology(self.id(), topo, visited);

        set_grad(self.id(), 1.0);

        for &node in topo.iter().rev() {
            if let Some(backward) = backward(node) {
                backward(op_info(node).1, node);
            }
        }
    }

    fn build_topology(id: usize, topology: &mut Vec<usize>, visited: &mut HashSet<usize>) {
        if !visited.contains(&id) {
            visited.insert(id);
            for child in op_info(id).1 {
                Self::build_topology(child, topology, visited);
            }
            topology.push(id);
        }
    }

    fn new_impl(data: f64) -> Self {
        let id = next_id();
        set_val(id, data);
        set_grad(id, 0.0);
        set_backward(id, None);
        op_info_store()
            .lock()
            .unwrap()
            .insert(id, ("".to_string(), vec![]));
        Self { id }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (op, args) = op_info(self.id());
        if !args.is_empty() {
            fmt.write_fmt(format_args!(
                "[{0}] [{1}] [{2}] = [{3}], ",
                args[0],
                op,
                args[1],
                self.data()
            ))?;
        }
        fmt.write_fmt(format_args!("grad = {0}", self.grad()))?;
        Ok(())
    }
}

fn add_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];
    let arg1 = args[1];

    let out_grad = grad(out);
    set_grad(arg0, grad(arg0) + 1.0 * out_grad);
    set_grad(arg1, grad(arg1) + 1.0 * out_grad);
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let val = Value::new(self.data() + other.data());
        set_backward(val.id(), Some(add_backward));
        op_info_store()
            .lock()
            .unwrap()
            .insert(val.id(), ("+".to_string(), vec![self.id(), other.id()]));
        val
    }
}

fn mul_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];
    let arg1 = args[1];

    let out_grad = grad(out);
    set_grad(arg0, grad(arg0) + val(arg1) * out_grad);
    set_grad(arg1, grad(arg1) + val(arg0) * out_grad);
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let val = Value::new(self.data() * other.data());
        set_backward(val.id(), Some(mul_backward));
        op_info_store()
            .lock()
            .unwrap()
            .insert(val.id(), ("*".to_string(), vec![self.id(), other.id()]));
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::*;

    #[test]
    fn init_tests_id() {
        let x = Value::new(1.0);
        let y = x + x;

        assert_ne!(x.id(), 0);
        assert!(x.id() < y.id());
    }

    #[test]
    fn init_tests_grad() {
        let x = Value::new(1.0);
        let y = x + x;

        assert_float_eq!(x.grad(), 0., abs <= 1e-10);
        assert_float_eq!(y.grad(), 0., abs <= 1e-10);
    }

    #[test]
    fn init_tests_display() {
        let x = Value::new(1.0);
        let y = x + x;

        assert_eq!(format!("{}", x), "grad = 0");
        assert_eq!(format!("{}", y), "[1] [+] [1] = [2], grad = 0");
    }

    #[test]
    fn test_add() {
        let x = Value::new(1.0);
        let y = Value::new(2.0);
        let z = x + y;

        assert_float_eq!(z.data(), 3.0, abs <= 1e-10);
    }

    #[test]
    fn test_add_grad() {
        let x = Value::new(1.0);
        let y = Value::new(2.0);
        let z = x + y;
        z.backward();

        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_self_add() {
        let x = Value::new(-2.0);
        let z = x + x;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 2.0, abs <= 1e-10);
        assert_float_eq!(z.data(), -4.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_mul() {
        let x = Value::new(-2.0);
        let y = Value::new(2.0);
        let z = x * y;

        assert_float_eq!(z.data(), -4.0, abs <= 1e-10);
    }

    #[test]
    fn test_mul_grad() {
        let x = Value::new(-2.0);
        let y = Value::new(2.0);
        let z = x * y;
        z.backward();

        assert_float_eq!(x.grad(), 2.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), -2.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_self_mul() {
        let x = Value::new(-2.0);
        let z = x * x;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -4.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 4.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }
}
