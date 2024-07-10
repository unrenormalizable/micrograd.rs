extern crate micrograd;

use float_eq::*;
use micrograd::engine::*;

#[test]
fn karpathy_test_sanity_check() {
    let x = Value::new(-4.0);
    let z = 2. * x + 2. + x;
    let q = z.relu() + z * x;
    let h = (z * z).relu();
    let y = h + q + q * x;
    y.backward();
    let (xmg, ymg) = (x, y);

    // forward pass went well
    assert_float_eq!(ymg.data(), -20.0, abs <= 1e-10);
    // backward pass went well
    assert_float_eq!(xmg.grad(), 46.0, abs <= 1e-10);
}

#[test]
fn karpathy_test_more_ops() {
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
    let (amg, bmg, gmg) = (a, b, g);

    let tol = 1e-6;
    // forward pass went well
    assert_float_eq!(gmg.data(), 24.70408163265306, abs <= tol);
    // backward pass went well
    assert_float_eq!(amg.grad(), 138.83381924198252, abs <= tol);
    assert_float_eq!(bmg.grad(), 645.5772594752186, abs <= tol);
}
