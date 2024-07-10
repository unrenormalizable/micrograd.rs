extern crate micrograd;

use micrograd::engine::*;

fn main() {
    let x = Value::new(-4.0);
    let z = 2. * x + 2. + x;
    let q = z.relu() + z * x;
    let h = (z * z).relu();
    let y = h + q + q * x;
    y.backward();

    let dot = render_dot(y.id());
    println!("{dot}");
}
