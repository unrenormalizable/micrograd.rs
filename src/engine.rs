use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type ValueId = usize;

pub type ValueType = f64;

pub type BackwardFn = Option<fn(&[Value], ValueType, ValueType)>;

/// Value backed by an autograd engine.
///
/// TODO: Move to an inner type.
#[derive(Clone, Debug)]
pub struct Value {
    id: ValueId,
    data: Rc<RefCell<ValueType>>,
    grad: Rc<RefCell<ValueType>>,
    backward: Rc<RefCell<BackwardFn>>,
    op_name: Rc<RefCell<Option<String>>>,
    op_args: Vec<Value>,
}

impl Value {
    pub fn new(data: ValueType) -> Self {
        Self {
            id: super::get_id(),
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.)),
            backward: Rc::new(RefCell::new(None)),
            op_name: Rc::new(RefCell::new(None)),
            op_args: vec![],
        }
    }

    pub fn id(&self) -> ValueId {
        self.id
    }

    pub fn data(&self) -> ValueType {
        *self.data.borrow()
    }

    pub fn grad(&self) -> ValueType {
        *self.grad.borrow()
    }

    pub fn reset_grad(&self) {
        *self.grad.borrow_mut() = Default::default()
    }

    pub fn relu(&self) -> Value {
        let mut val = Value::new(if self.data() < 0. { 0. } else { self.data() });
        *val.backward.borrow_mut() = Some(Self::relu_backward);
        *val.op_name.borrow_mut() = Some("ReLU".to_string());
        val.op_args.append(&mut vec![self.clone()]);
        val
    }

    pub fn pow(&self, exp: ValueType) -> Value {
        self.powv(Value::new(exp))
    }

    pub fn powv(&self, exp: Value) -> Value {
        let mut val = Value::new(self.data().powf(exp.data()));
        *val.backward.borrow_mut() = Some(Self::pow_backward);
        *val.op_name.borrow_mut() = Some("^".to_string());
        val.op_args.append(&mut vec![self.clone(), exp.clone()]);
        val
    }

    pub fn backward(&self) {
        *self.grad.borrow_mut() = 1.0;
        let topo = Self::build_topology(self.clone());
        for node in topo {
            let backward = node.backward.borrow();
            if backward.is_some() {
                backward.unwrap()(&node.op_args, node.data(), node.grad());
            }
        }
    }

    fn build_topology(root: Value) -> Vec<Value> {
        let mut topo = vec![];
        let mut visited = HashSet::<ValueId>::new();

        Self::build_topology_impl(root, &mut topo, &mut visited);
        topo.reverse();

        topo
    }

    fn build_topology_impl(root: Value, topology: &mut Vec<Value>, visited: &mut HashSet<ValueId>) {
        let id = root.id;
        if !visited.contains(&id) {
            visited.insert(id);
            for child in root.op_args.iter() {
                Self::build_topology_impl(child.clone(), topology, visited);
            }
            topology.push(root);
        }
    }

    fn add_backward(args: &[Value], _out_data: ValueType, out_grad: ValueType) {
        args[0].update_grad(out_grad);
        args[1].update_grad(out_grad);
    }

    fn mul_backward(args: &[Value], _out_data: ValueType, out_grad: ValueType) {
        args[0].update_grad(args[1].data() * out_grad);
        args[1].update_grad(args[0].data() * out_grad);
    }

    fn pow_backward(args: &[Value], _out_data: ValueType, out_grad: ValueType) {
        args[0].update_grad(args[1].data() * args[0].data().powf(args[1].data() - 1.) * out_grad)
    }

    fn relu_backward(args: &[Value], out_data: ValueType, out_grad: ValueType) {
        let delta = if out_data > 0. { out_grad } else { 0. };
        args[0].update_grad(delta);
    }

    fn update_grad(&self, delta: ValueType) {
        let args0_grad = *self.grad.borrow();
        *self.grad.borrow_mut() = args0_grad + delta;
    }
}

impl Drop for Value {
    fn drop(&mut self) {}
}

impl std::fmt::Display for Value {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.op_args.len() {
            0 => {
                fmt.write_fmt(format_args!(
                    "Value({:.06}, grad={:.06})",
                    self.data(),
                    self.grad()
                ))?;
            }
            1 => {
                fmt.write_fmt(format_args!(
                    "{} [Value({:.06}, grad={:.06})]",
                    self.op_name.borrow().clone().unwrap(),
                    self.op_args[0].data(),
                    self.op_args[0].grad(),
                ))?;
            }
            2 => {
                fmt.write_fmt(format_args!(
                    "[Value({:.06}, grad={:.06})] {} [Value({:.06}, grad={:.06})]",
                    self.op_args[0].data(),
                    self.op_args[0].grad(),
                    self.op_name.borrow().clone().unwrap(),
                    self.op_args[1].data(),
                    self.op_args[1].grad(),
                ))?;
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let mut val = Value::new(self.data() + rhs.data());
        *val.backward.borrow_mut() = Some(Self::add_backward);
        *val.op_name.borrow_mut() = Some("+".to_string());
        val.op_args.append(&mut vec![self.clone(), rhs.clone()]);
        val
    }
}

impl std::ops::Add<Value> for ValueType {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value::new(self) + rhs
    }
}

impl std::ops::Add<ValueType> for Value {
    type Output = Value;

    fn add(self, rhs: ValueType) -> Self::Output {
        self + Value::new(rhs)
    }
}

impl std::ops::Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<Value> for ValueType {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<ValueType> for Value {
    type Output = Value;

    fn sub(self, rhs: ValueType) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let mut val = Value::new(self.data() * rhs.data());
        *val.backward.borrow_mut() = Some(Self::mul_backward);
        *val.op_name.borrow_mut() = Some("*".to_string());
        val.op_args.append(&mut vec![self.clone(), rhs.clone()]);
        val
    }
}

impl std::ops::Mul<Value> for ValueType {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value::new(self) * rhs
    }
}

impl std::ops::Mul<ValueType> for Value {
    type Output = Value;

    fn mul(self, rhs: ValueType) -> Self::Output {
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

impl std::ops::Div<Value> for ValueType {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        let lhs = Value::new(self);

        lhs / rhs
    }
}

impl std::ops::Div<ValueType> for Value {
    type Output = Value;

    fn div(self, rhs: ValueType) -> Self::Output {
        let rhs = Value::new(rhs);

        self / rhs
    }
}

pub mod viz {
    use super::*;
    use std::collections::HashMap;

    pub fn render_dot(root: Value) -> String {
        let (nodes_ids, edges, id_node_map) = trace(root);

        let mut nodes_str = String::new();
        let mut edges_str = String::new();
        for node_id in nodes_ids {
            let node = id_node_map.get(&node_id).unwrap();
            let id_str = format!("{:08}", node_id);
            nodes_str += &format!(
                "    \"{}\" [label=\"{{ data {:.06} | grad {:.06} }}\" shape=record]\n",
                id_str,
                node.data(),
                node.grad(),
            );
            let op_name = node.op_name.borrow().clone();
            if op_name.is_some() {
                let op_name = op_name.unwrap();
                nodes_str += &format!("    \"{}{}\" [label=\"{}\"]\n", id_str, op_name, op_name);
                edges_str += &format!("    \"{}{}\" -> \"{}\"\n", id_str, op_name, id_str);
            }
        }

        for (n1, n2) in edges {
            let n1_str = format!("{:08}", n1);
            let n2_str = format!("{:08}", n2);
            let node2 = id_node_map.get(&n2).unwrap();
            let op_name = node2.op_name.borrow().clone().unwrap();

            edges_str += &format!("    \"{}\" -> \"{}{}\"\n", n1_str, n2_str, op_name);
        }

        format!(
            "strict digraph {{\n    graph [rankdir=LR]\n\n{}{}}}",
            nodes_str, edges_str
        )
    }

    type NodeId = ValueId;
    type Edge = (NodeId, NodeId);

    fn trace(root: Value) -> (HashSet<NodeId>, HashSet<Edge>, HashMap<NodeId, Value>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        let mut id_node_map = HashMap::new();
        build(root, &mut nodes, &mut edges, &mut id_node_map);
        (nodes, edges, id_node_map)
    }

    fn build(
        node: Value,
        nodes: &mut HashSet<NodeId>,
        edges: &mut HashSet<Edge>,
        id_node_map: &mut HashMap<NodeId, Value>,
    ) {
        let node_id = node.id();
        if !nodes.contains(&node_id) {
            nodes.insert(node_id);
            id_node_map.insert(node_id, node.clone());
            for child in node.op_args.clone() {
                edges.insert((child.id(), node_id));
                build(child, nodes, edges, id_node_map);
            }
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
        let y = x.clone() + x.clone();

        assert_ne!(x.id(), 0);
        assert!(x.id() < y.id());
    }

    #[test]
    fn init_tests_grad() {
        let x = Value::new(1.0);
        let y = x.clone() + x.clone();

        assert_float_eq!(x.grad(), 0., abs <= 1e-10);
        assert_float_eq!(y.grad(), 0., abs <= 1e-10);
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
        let z = x.clone() + y.clone();
        z.backward();

        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_self_add() {
        let x = Value::new(-2.0);
        let z = x.clone() + x.clone();
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 2.0, abs <= 1e-10);
        assert_float_eq!(z.data(), -4.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_add_primitive() {
        let x = Value::new(-2.0);
        let z = 2. + x.clone() + 3.;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 1.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 3.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn init_tests_display() {
        let x = Value::new(1.0);
        let y = x.relu();
        let z = x.clone() + y.clone();
        z.backward();

        assert_float_eq!(x.data(), 1.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 2.0, abs <= 1e-10);
        assert_eq!(format!("{}", x), "Value(1.000000, grad=2.000000)");

        assert_float_eq!(y.data(), 1.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), 1.0, abs <= 1e-10);
        assert_eq!(format!("{}", y), "ReLU [Value(1.000000, grad=2.000000)]");

        assert_float_eq!(z.data(), 2.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
        assert_eq!(
            format!("{}", z),
            "[Value(1.000000, grad=2.000000)] + [Value(1.000000, grad=1.000000)]"
        );
    }

    #[test]
    fn test_sub_grad() {
        let x = Value::new(-2.0);
        let y = Value::new(5.0);
        let z = x.clone() - y.clone();
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
        let z = x.clone() - x.clone();
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 0.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_sub_primitive() {
        let x = Value::new(-2.0);
        let z = 2. - x.clone() - 3.;
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
        let z = x.clone() * y.clone();
        z.backward();

        assert_float_eq!(x.grad(), 2.0, abs <= 1e-10);
        assert_float_eq!(y.grad(), -2.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_self_mul() {
        let x = Value::new(-2.0);
        let z = x.clone() * x.clone();
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -4.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 4.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_mul_primitive() {
        let x = Value::new(-2.0);
        let z = 2. * x.clone() * 3.;
        z.backward();

        assert_float_eq!(x.data(), -2.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), 6.0, abs <= 1e-10);
        assert_float_eq!(z.data(), -12.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }

    #[test]
    fn test_neg() {
        let x = Value::new(-2.0);
        let z = -x.clone();
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
        let z = x.clone() * x.clone();
        let z = z.relu();
        z.backward();

        assert_float_eq!(x.data(), -5.0, abs <= 1e-10);
        assert_float_eq!(x.grad(), -10.0, abs <= 1e-10);
        assert_float_eq!(z.data(), 25.0, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);

        let x = Value::new(3.0);
        let z = x.clone() * x.clone();
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
        let z = x.clone() / y;
        z.backward();

        assert_float_eq!(x.data(), 1.51, abs <= 1e-10);
        assert_float_eq!(x.grad(), -0.283_929_585_462_805_2, abs <= 1e-10);
        assert_float_eq!(z.data(), -0.428_733_674_048_835_9, abs <= 1e-10);
        assert_float_eq!(z.grad(), 1.0, abs <= 1e-10);
    }
}
