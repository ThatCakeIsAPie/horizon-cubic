//! The arithmetic of the Cubic Chunks Customize screen: CubicWorldGen's slider
//! mappings (`CwgGuiFactory`: `slider`, `intSlider`, `positiveExponentialSlider`,
//! `symmetricExponentialSlider`, `invertedPositiveExponentialSlider`), its Lock
//! of the expected heights, and the ore height units. No drawing here.
use horizon_world::cubic::custom::CustomGeneratorSettings;

/// How a slider's position (0 to 1 along its track) stands for a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mapping {
    /// `slider(min, max)`: linear.
    Linear { min: f64, max: f64 },
    /// `intSlider(min, max)`: linear, whole numbers.
    Int { min: i32, max: i32 },
    /// `positiveExponentialSlider(lo, hi)`: `2^lo` to `2^hi`, evenly in the
    /// exponent.
    PosExp { lo: f64, hi: f64 },
    /// `symmetricExponentialSlider(lo, hi)`: the middle is 0, the two halves
    /// are the negative and positive sides of `PosExp`, so both ends are
    /// `2^hi - 2^lo` in size.
    SymExp { lo: f64, hi: f64 },
    /// A noise period, `2^lo` to `2^hi` blocks (CWG's sliders store the
    /// period and write the frequency `1 / period`).
    Period { lo: f64, hi: f64 },
    /// Linear in the cube of the position: finer steps near 0, for
    /// probabilities. `0` to `max`.
    Cubic { max: f64 },
}

impl Mapping {
    /// The value at `pos` (0 to 1).
    pub fn value(self, pos: f32) -> f64 {
        let p = f64::from(pos.clamp(0.0, 1.0));
        match self {
            Self::Linear { min, max } => min + (max - min) * p,
            Self::Int { min, max } => {
                (f64::from(min) + (f64::from(max) - f64::from(min)) * p).round()
            }
            Self::PosExp { lo, hi } | Self::Period { lo, hi } => (lo + (hi - lo) * p).exp2(),
            Self::SymExp { lo, hi } => {
                let t = p * 2.0 - 1.0;
                let magnitude = (lo + (hi - lo) * t.abs()).exp2() - lo.exp2();
                // The centre cell is exactly 0.
                if t.abs() < 1e-9 {
                    0.0
                } else {
                    magnitude.copysign(t)
                }
            }
            Self::Cubic { max } => max * p * p * p,
        }
    }
    /// The position (0 to 1) of `value`: the inverse of [`Mapping::value`],
    /// clamped to the track.
    pub fn pos(self, value: f64) -> f32 {
        let p = match self {
            Self::Linear { min, max } => (value - min) / (max - min),
            Self::Int { min, max } => (value - f64::from(min)) / (f64::from(max) - f64::from(min)),
            Self::PosExp { lo, hi } | Self::Period { lo, hi } => {
                (value.max(f64::MIN_POSITIVE).log2() - lo) / (hi - lo)
            }
            Self::SymExp { lo, hi } => {
                let t = ((value.abs() + lo.exp2()).log2() - lo) / (hi - lo);
                (t.copysign(value) + 1.0) / 2.0
            }
            Self::Cubic { max } => (value / max).max(0.0).cbrt(),
        };
        if p.is_finite() {
            p.clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    }
}

/// A number as a slider's label shows it: enough digits to tell neighbours
/// apart, no trailing zeros.
pub fn show(value: f64) -> String {
    if !value.is_finite() {
        return if value < 0.0 { "-inf" } else { "inf" }.into();
    }
    let a = value.abs();
    let text = if a >= 1000.0 || a == 0.0 {
        format!("{value:.0}")
    } else if a >= 100.0 {
        format!("{value:.1}")
    } else if a >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.4}")
    };
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

/// `1 / period`, the frequency a period slider writes (0 for a period of 0).
pub fn frequency(period: f64) -> f64 {
    if period == 0.0 { 0.0 } else { 1.0 / period }
}
/// The period a stored frequency shows as.
pub fn period(frequency: f64) -> f64 {
    if frequency == 0.0 {
        f64::INFINITY
    } else {
        1.0 / frequency
    }
}

/// The Lock of the Advanced page (`updateExpectedHeightsIfNeeded`): while it is
/// on, the expected base height is the height offset, the expected variation is
/// the height factor and the actual height is
/// `heightOffset + heightVariationOffset + max(2 hf + hvf, hf + 2 hvf)`.
pub fn lock_heights(s: &mut CustomGeneratorSettings) {
    s.expected_base_height = s.height_offset;
    s.expected_height_variation = s.height_factor;
    s.actual_height = actual_height(s);
}
/// The actual height the Lock computes.
pub fn actual_height(s: &CustomGeneratorSettings) -> f64 {
    let hf = s.height_factor;
    let hvf = s.height_variation_factor;
    s.height_offset + s.height_variation_offset + (hf * 2.0 + hvf).max(hf + hvf * 2.0)
}
/// Whether the settings' expected heights are the ones the Lock would set (so
/// the screen opens with the Lock on, as CWG's does for its preset).
pub fn heights_locked(s: &CustomGeneratorSettings) -> bool {
    let mut locked = s.clone();
    lock_heights(&mut locked);
    (
        locked.expected_base_height,
        locked.expected_height_variation,
        locked.actual_height,
    ) == (
        s.expected_base_height,
        s.expected_height_variation,
        s.actual_height,
    )
}

/// An ore height in blocks: `h * expectedHeightVariation + expectedBaseHeight`.
pub fn ore_block_y(h: f64, s: &CustomGeneratorSettings) -> f64 {
    h * s.expected_height_variation + s.expected_base_height
}

/// The ends of an ore height range slider: a height at or past them is the
/// range's open end (CWG's `-inf` / `+inf`).
pub const ORE_RANGE: f64 = 2.0;

/// The height range slider's value for `h` (an open end is the slider's end).
pub fn ore_slider_value(h: f64) -> f64 {
    h.clamp(-ORE_RANGE, ORE_RANGE)
}
/// The height a range slider value stands for: its ends are unbounded.
pub fn ore_from_slider(value: f64) -> f64 {
    if value <= -ORE_RANGE {
        f64::NEG_INFINITY
    } else if value >= ORE_RANGE {
        f64::INFINITY
    } else {
        // Hundredths, as the slider steps.
        (value * 100.0).round() / 100.0
    }
}

/// The cube Y a cave height slider (in cubes, `-CAVE_RANGE..=CAVE_RANGE`)
/// stands for: its ends are unbounded, as CWG's defaults are.
pub const CAVE_RANGE: i32 = 64;
pub fn cave_from_slider(value: i32, lower: bool) -> i32 {
    if value <= -CAVE_RANGE && lower {
        i32::MIN / 16
    } else if value >= CAVE_RANGE && !lower {
        i32::MAX / 16
    } else {
        value
    }
}
pub fn cave_slider_value(cube_y: i32) -> i32 {
    cube_y.clamp(-CAVE_RANGE, CAVE_RANGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} is not {b}");
    }

    #[test]
    fn every_mapping_returns_the_value_it_was_given() {
        let cases: [(Mapping, &[f64]); 6] = [
            (
                Mapping::Linear {
                    min: -1.0,
                    max: 1.0,
                },
                &[-0.7, 0.0, 0.3, 1.0],
            ),
            (Mapping::Int { min: 1, max: 50 }, &[1.0, 17.0, 50.0]),
            (
                Mapping::PosExp { lo: 0.0, hi: 20.0 },
                &[1.0, 64.0, 1048576.0],
            ),
            (
                Mapping::SymExp { lo: 1.0, hi: 20.0 },
                &[-1000.0, 64.0, 5000.0],
            ),
            (
                Mapping::Period { lo: 2.0, hi: 13.0 },
                &[16.0, 655.36, 4096.0],
            ),
            (Mapping::Cubic { max: 1.0 }, &[0.0, 0.0032, 0.25]),
        ];
        for (mapping, values) in cases {
            for &value in values {
                near(mapping.value(mapping.pos(value)), value);
            }
        }
    }

    #[test]
    fn the_symmetric_slider_is_zero_in_the_middle_and_odd() {
        let m = Mapping::SymExp { lo: 0.0, hi: 20.0 };
        assert_eq!(m.value(0.5), 0.0);
        near(m.value(1.0), 2f64.powi(20) - 1.0);
        near(m.value(0.0), -(2f64.powi(20) - 1.0));
        near(m.value(0.75), -m.value(0.25));
        // A zero offset (the preset's height variation offset) sits in the middle.
        assert_eq!(m.pos(0.0), 0.5);
    }

    #[test]
    fn a_noise_period_writes_the_inverse_as_its_frequency() {
        // The preset's depth noise: period 655.36 (frequency 0.00152588).
        let frequency_x = 1.0 / 655.36;
        let shown = period(frequency_x);
        near(shown, 655.360_0);
        let m = Mapping::Period { lo: 2.0, hi: 13.0 };
        let stored = frequency(m.value(m.pos(shown)));
        near(stored, frequency_x);
        assert_eq!(frequency(0.0), 0.0);
        // The ends of the track: periods 4 and 8192 blocks.
        near(m.value(0.0), 4.0);
        near(m.value(1.0), 8192.0);
    }

    #[test]
    fn int_sliders_step_in_whole_numbers_and_stay_in_range() {
        let m = Mapping::Int { min: 1, max: 8 };
        assert_eq!(m.value(0.0), 1.0);
        assert_eq!(m.value(1.0), 8.0);
        assert_eq!(m.value(0.43), 4.0);
        assert_eq!(m.pos(100.0), 1.0, "out of range stays on the track");
    }

    #[test]
    fn the_lock_follows_height_offset_and_factor_and_computes_the_actual_height() {
        let mut s = CustomGeneratorSettings::defaults();
        s.height_factor = 100.0;
        s.height_offset = 30.0;
        s.height_variation_factor = 40.0;
        s.height_variation_offset = 5.0;
        lock_heights(&mut s);
        assert_eq!(s.expected_base_height, 30.0);
        assert_eq!(s.expected_height_variation, 100.0);
        // 30 + 5 + max(2*100 + 40, 100 + 2*40) = 35 + 240.
        assert_eq!(s.actual_height, 275.0);
        // With a large variation factor the other term wins:
        // 30 + 5 + max(2*10 + 400, 10 + 800) = 35 + 810.
        s.height_factor = 10.0;
        s.height_variation_factor = 400.0;
        lock_heights(&mut s);
        assert_eq!(s.actual_height, 845.0);
        assert!(heights_locked(&s));
        s.expected_base_height = 1.0;
        assert!(!heights_locked(&s));
    }

    #[test]
    fn the_preset_opens_with_the_lock_on() {
        assert!(heights_locked(&CustomGeneratorSettings::defaults()));
    }

    #[test]
    fn ore_heights_convert_to_blocks_and_back_and_the_slider_ends_are_open() {
        let s = CustomGeneratorSettings::defaults();
        // blockY = h * 64 + 64.
        near(ore_block_y(0.5, &s), 96.0);
        near(ore_block_y(-1.0, &s), 0.0);
        assert_eq!(ore_from_slider(-2.0), f64::NEG_INFINITY);
        assert_eq!(ore_from_slider(2.0), f64::INFINITY);
        assert_eq!(ore_from_slider(0.504), 0.5);
        assert_eq!(ore_slider_value(f64::INFINITY), 2.0);
        assert_eq!(cave_from_slider(-CAVE_RANGE, true), i32::MIN / 16);
        assert_eq!(cave_from_slider(-CAVE_RANGE, false), -CAVE_RANGE);
        assert_eq!(cave_from_slider(CAVE_RANGE, false), i32::MAX / 16);
        assert_eq!(cave_slider_value(i32::MAX / 16), CAVE_RANGE);
    }

    #[test]
    fn numbers_show_without_trailing_zeros() {
        assert_eq!(show(64.0), "64");
        assert_eq!(show(0.5), "0.5");
        assert_eq!(show(655.36), "655.4");
        assert_eq!(show(0.003_2), "0.0032");
        assert_eq!(show(f64::INFINITY), "inf");
    }
}
