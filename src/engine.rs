use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

// TODO: Implement operator overloading without needing to wrap with Value;
// TODO: Make Value generic, can we used an Inner Type?
// TODO: Ratatui visualization
// TODO: On destruction clear up stuff on the global caches.

// TODO: RHS rename param
// TODO: move to namespaces
// TODO: Getter and setter within Value for cache items
// TODO: set_grad to update_grad
// TODO: Update readme.md, add links to karpathy, stackoverlfow safe globals
// TODO: display format as per karpathy's

///
/// Value backed by an autograd engine.
///
#[derive(Copy, Clone, Debug)]
pub struct Value {
    id: usize,
}

impl Value {
    pub fn new(data: f64) -> Self {
        let id = next_id();
        set_data(id, data);
        set_grad(id, 0.0);
        set_backward(id, None);
        set_op_info(id, ("".to_string(), vec![]));
        Self { id }
    }

    pub fn relu(&self) -> Value {
        let val = Value::new(if self.data() < 0. { 0. } else { self.data() });
        set_backward(val.id(), Some(relu_backward));
        set_op_info(val.id(), ("ReLU".to_string(), vec![self.id()]));
        val
    }

    pub fn pow(&self, exp: f64) -> Value {
        self.powv(Value::new(exp))
    }

    pub fn powv(&self, exp: Value) -> Value {
        let val = Value::new(self.data().powf(exp.data()));
        set_backward(val.id(), Some(pow_backward));
        set_op_info(val.id(), ("^".to_string(), vec![self.id(), exp.id()]));
        val
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn data(&self) -> f64 {
        data(self.id())
    }

    pub fn grad(&self) -> f64 {
        grad(self.id())
    }

    pub fn op_info(&self) -> OpInfo {
        op_info(self.id())
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

fn pow_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];
    let arg1 = args[1];

    let arg1_data = data(arg1);
    set_grad(
        arg0,
        grad(arg0) + (arg1_data * data(arg0).powf(arg1_data - 1.)) * grad(out),
    )
}

fn relu_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];

    let arg0_grad = grad(arg0);
    if data(out) > 0. {
        set_grad(arg0, arg0_grad + grad(out))
    } else {
        set_grad(arg0, arg0_grad + 0.)
    }
}

fn add_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];
    let arg1 = args[1];

    let out_grad = grad(out);
    set_grad(arg0, grad(arg0) + out_grad);
    set_grad(arg1, grad(arg1) + out_grad);
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let val = Value::new(self.data() + rhs.data());
        set_backward(val.id(), Some(add_backward));
        set_op_info(val.id(), ("+".to_string(), vec![self.id(), rhs.id()]));
        val
    }
}

impl std::ops::Add<Value> for f64 {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::new(self) + rhs
    }
}

impl std::ops::Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Self::Output {
        self + Value::new(rhs)
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<Value> for f64 {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<f64> for Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Self::Output {
        self + (-rhs)
    }
}

fn mul_backward(args: Vec<usize>, out: usize) {
    let arg0 = args[0];
    let arg1 = args[1];

    let out_grad = grad(out);
    set_grad(arg0, grad(arg0) + data(arg1) * out_grad);
    set_grad(arg1, grad(arg1) + data(arg0) * out_grad);
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let val = Value::new(self.data() * rhs.data());
        set_backward(val.id(), Some(mul_backward));
        set_op_info(val.id(), ("*".to_string(), vec![self.id(), rhs.id()]));
        val
    }
}

impl std::ops::Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value::new(self) * rhs
    }
}

impl std::ops::Mul<f64> for Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Value::new(rhs)
    }
}

impl std::ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::new(-1.0) * self
    }
}

impl std::ops::Div for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        self * rhs.pow(-1.)
    }
}

impl std::ops::Div<Value> for f64 {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        let lhs = Value::new(self);

        lhs / rhs
    }
}

impl std::ops::Div<f64> for Value {
    type Output = Value;

    fn div(self, rhs: f64) -> Self::Output {
        let rhs = Value::new(rhs);

        self / rhs
    }
}

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

fn data_store() -> &'static Mutex<HashMap<usize, f64>> {
    static MAP: OnceLock<Mutex<HashMap<usize, f64>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn data(id: usize) -> f64 {
    *data_store().lock().unwrap().get(&id).unwrap()
}

fn set_data(id: usize, data: f64) {
    data_store().lock().unwrap().insert(id, data);
}

type OpInfo = (String, Vec<usize>);

fn op_info_store() -> &'static Mutex<HashMap<usize, OpInfo>> {
    static MAP: OnceLock<Mutex<HashMap<usize, OpInfo>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn op_info(id: usize) -> (String, Vec<usize>) {
    (*op_info_store().lock().unwrap().get(&id).unwrap()).clone()
}

fn set_op_info(id: usize, op_info: OpInfo) {
    op_info_store().lock().unwrap().insert(id, op_info);
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

pub fn render_dot(root: usize) -> String {
    let (nodes, edges) = trace(root);

    let mut nodes_str = String::new();
    let mut edges_str = String::new();
    for node in nodes {
        let id_str = format!("{:08}", node);
        nodes_str += &format!(
            "    \"{}\" [label=\"{{ data {:.06} | grad {:.06} }}\" shape=record]\n",
            id_str,
            data(node),
            grad(node)
        );
        let op_info = op_info(node);
        if !op_info.0.is_empty() {
            nodes_str += &format!(
                "    \"{}{}\" [label=\"{}\"]\n",
                id_str, op_info.0, op_info.0
            );
            edges_str += &format!("    \"{}{}\" -> \"{}\"\n", id_str, op_info.0, id_str);
        }
    }

    for (n1, n2) in edges {
        let n1_str = format!("{:08}", n1);
        let n2_str = format!("{:08}", n2);
        let op_info2 = op_info(n2);
        edges_str += &format!("    \"{}\" -> \"{}{}\"\n", n1_str, n2_str, op_info2.0);
    }

    let x = format!(
        r#"strict digraph {{
    graph [rankdir=LR]
{}{}}}"#,
        nodes_str, edges_str
    );
    print!("{}", x);
    x
}

fn trace(root: usize) -> (HashSet<usize>, HashSet<(usize, usize)>) {
    let mut nodes = HashSet::new();
    let mut edges = HashSet::new();
    build(root, &mut nodes, &mut edges);
    (nodes, edges)
}

fn build(v: usize, nodes: &mut HashSet<usize>, edges: &mut HashSet<(usize, usize)>) {
    if !nodes.contains(&v) {
        nodes.insert(v);
        for child in op_info(v).1 {
            edges.insert((child, v));
            build(child, nodes, edges);
        }
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
    fn test_add_primitive() {
        let x = Value::new(-2.0);
        let z = 2. + x + 3.;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 3.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_sub_grad() {
        let x = Value::new(-2.0);
        let y = Value::new(5.0);
        let z = x - y;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(y.data(), 5.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), -1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), -7.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_self_sub() {
        let x = Value::new(-2.0);
        let z = x - x;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_sub_primitive() {
        let x = Value::new(-2.0);
        let z = 2. - x - 3.;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 1.0, abs <= 1e-10);
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

    #[test]
    fn test_mul_primitive() {
        let x = Value::new(-2.0);
        let z = 2. * x * 3.;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 6.0, abs <= 1e-10);
        assert_float_eq!(z.data(), -12.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_neg() {
        let x = Value::new(-2.0);
        let z = -x;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 2.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_relu() {
        let x = Value::new(-5.0);
        let z = x.relu();
        z.backward();

        assert_float_eq!(x.data(), -5.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);

        let x = Value::new(3.0);
        let z = x.relu();
        z.backward();

        assert_float_eq!(x.data(), 3.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 3.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_relu_complex() {
        let x = Value::new(-5.0);
        let z = x * x;
        let z = z.relu();
        z.backward();

        assert_float_eq!(x.data(), -5.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -10.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 25.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);

        let x = Value::new(3.0);
        let z = x * x;
        let z = z.relu();
        z.backward();

        assert_float_eq!(x.data(), 3.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 6.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 9.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_pow() {
        let x = Value::new(1.5);
        let y = -3.5;
        let z = x.pow(y);
        z.backward();

        assert_float_eq!(x.data(), 1.5, abs <= 1e-10);
        assert_float_eq!(x.grad(), -0.5644914633574403, abs <= 1e-10);
        assert_float_eq!(z.data(), 0.2419249128674744, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_div() {
        let x = Value::new(1.5);
        let y = Value::new(-3.5);
        let y = y.pow(-1.);
        let z = x * y;
        z.backward();

        let x = Value::new(1.51);
        let y = Value::new(-3.522);
        let z = x / y;
        z.backward();

        assert_float_eq!(x.data(), 1.51, abs <= 1e-10);
        assert_float_eq!(x.grad(), -0.2839295854628052, abs <= 1e-10);
        assert_float_eq!(z.data(), -0.4287336740488359, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }
}
