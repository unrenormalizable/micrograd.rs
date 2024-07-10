extern crate micrograd;

use micrograd::engine::*;

fn main() {
    let a = Value::new(5.0);
    let b = 35.0 + a.pow(-3.) / -1.5;
    let c = b.relu();
    c.backward();

    let dot = viz::render_dot(c);
    println!("{dot}");
}
