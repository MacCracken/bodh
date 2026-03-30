//! Basic examples of bodh psychology formulas.

fn main() {
    // Psychophysics: Fitts' law
    let id = bodh::psychophysics::fitts_law(256.0, 4.0).unwrap();
    println!("Fitts' law ID (D=256, W=4): {id:.1} bits");

    // Psychophysics: Hick's law
    let rt = bodh::psychophysics::hicks_law_full(8, 0.2, 0.1).unwrap();
    println!("Hick's law RT (8 choices): {rt:.3} s");

    // Learning: Ebbinghaus forgetting curve
    let r = bodh::learning::ebbinghaus_forgetting(1.0, 2.0).unwrap();
    println!("Ebbinghaus retention at t=1, S=2: {r:.3}");

    // Decision: Prospect theory
    let gain = bodh::decision::prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    let loss = bodh::decision::prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    println!("Prospect theory: gain of 100 = {gain:.2}, loss of 100 = {loss:.2}");

    // Perception: d-prime
    let d = bodh::perception::d_prime(0.9, 0.1).unwrap();
    println!("d' (hit=0.9, FA=0.1): {d:.2}");

    // Development: Piaget stage
    let stage = bodh::development::PiagetStage::from_age(5.0);
    println!("Piaget stage at age 5: {stage:?}");
}
