extern crate micrograd;

use float_eq::*;
use micrograd::engine::*;

#[test]
fn karpathy_test_sanity_check2() {
    let x = Value::new(-4.0);
    let z = 2. * x.clone() + 2. + x.clone();
    let q = z.clone().relu() + z.clone() * x.clone();
    let h = (z.clone() * z).relu();
    let y = h + q.clone() + q * x.clone();
    y.backward();
    let (xmg, ymg) = (x, y.clone());

    // forward pass went well
    assert_float_eq!(ymg.data(), -20.0, abs <= 1e-10);
    // backward pass went well
    assert_float_eq!(xmg.grad(), 46.0, abs <= 1e-10);
}

#[test]
fn karpathy_test_more_ops2() {
    let a = Value::new(-4.0);
    let b = Value::new(2.0);
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().pow(3.);
    c = c.clone() + c.clone() + 1.;
    c = c.clone() + 1. + c + -a.clone();
    d = d.clone() + d.clone() * 2. + (b.clone() + a.clone()).relu();
    d = d.clone() + 3. * d + (b.clone() - a.clone()).relu();
    let e = c - d;
    let f = e.pow(2.);
    let mut g = f.clone() / 2.0;
    g = g + 10.0 / f;
    g.backward();
    let (amg, bmg, gmg) = (a, b, g);

    let tol = 1e-6;
    // forward pass went well
    assert_float_eq!(gmg.data(), 24.70408163265306, abs <= tol);
    // backward pass went well
    assert_float_eq!(amg.grad(), 138.83381924198252, abs <= tol);
    assert_float_eq!(bmg.grad(), 645.5772594752186, abs <= tol);
}
