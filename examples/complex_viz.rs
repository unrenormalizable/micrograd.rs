extern crate micrograd;

use micrograd::engine::*;

fn main() {
    let a = Value::new(-4.0);
    let b = Value::new(2.0);
    let mut c = a + b;
    let mut d = a * b + b.pow(3.);
    c = c + (c + 1.);
    c = c + (1. + c + (-a));
    d = d + (d * 2. + (b + a).relu());
    d = d + (3. * d + (b - a).relu());
    let e = c - d;
    let f = e.pow(2.);
    let mut g = f / 2.0;
    g = g + (10.0 / f);
    g.backward();

    let dot = viz::render_dot(g.id());
    println!("{dot}");
}
