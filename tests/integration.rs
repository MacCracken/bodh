//! Integration tests for bodh.

use bodh::decision;
use bodh::learning;
use bodh::perception;
use bodh::psychophysics;

#[test]
fn test_fitts_law_d256_w4() {
    let id = psychophysics::fitts_law(256.0, 4.0).unwrap();
    assert!((id - 7.0).abs() < 1e-10);
}

#[test]
fn test_hicks_law_8_choices_gives_3b() {
    let rt = psychophysics::hicks_law(8, 1.0).unwrap();
    assert!((rt - 3.0).abs() < 1e-10);
}

#[test]
fn test_ebbinghaus_at_zero() {
    let r = learning::ebbinghaus_forgetting(0.0, 1.0).unwrap();
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn test_prospect_theory_loss_aversion_integration() {
    let gain = decision::prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    let loss = decision::prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    assert!(gain > 0.0);
    assert!(loss < 0.0);
    assert!(loss.abs() > gain.abs());
}

#[test]
fn test_d_prime_discriminability() {
    let d = perception::d_prime(0.9, 0.1).unwrap();
    assert!(d > 2.0); // good discrimination
}

#[test]
fn test_weber_fechner_no_change() {
    let p = psychophysics::weber_fechner(100.0, 100.0, 1.0).unwrap();
    assert!((p - 0.0).abs() < 1e-10);
}

#[test]
fn test_stevens_power_law_zero_stimulus() {
    let s = psychophysics::stevens_power_law(0.0, 1.0, 0.33).unwrap();
    assert!((s - 0.0).abs() < 1e-10);
}

#[test]
fn test_spaced_repetition_increasing_intervals() {
    let i1 = learning::spaced_repetition_interval(1, 2.5).unwrap();
    let i2 = learning::spaced_repetition_interval(2, 2.5).unwrap();
    let i3 = learning::spaced_repetition_interval(3, 2.5).unwrap();
    assert!(i1 < i2);
    assert!(i2 < i3);
}
