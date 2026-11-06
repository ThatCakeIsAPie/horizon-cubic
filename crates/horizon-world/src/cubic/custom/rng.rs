//! Small random helpers shared by the custom generator: a `java.util.Random`
//! with infallible draws, CWG's per-cube seed, and position hashes.

use horizon_core::random::{LegacyRandom, RandomSource, mix_stafford13};

/// `java.util.Random` (the generator CWG uses for caves, ores and lakes) with
/// draws that cannot fail.
#[derive(Clone, Debug)]
pub struct JRandom(LegacyRandom);

impl JRandom {
    pub fn new(seed: i64) -> Self {
        Self(LegacyRandom::new(seed))
    }
    pub fn set_seed(&mut self, seed: i64) {
        self.0 = LegacyRandom::new(seed);
    }
    /// `nextInt(bound)`; a bound below 1 gives 0 (Java would throw).
    pub fn int(&mut self, bound: i32) -> i32 {
        if bound <= 0 {
            return 0;
        }
        self.0.next_int_bound(bound).unwrap_or(0)
    }
    pub fn float(&mut self) -> f32 {
        self.0.next_float()
    }
    pub fn double(&mut self) -> f64 {
        self.0.next_double()
    }
    pub fn long(&mut self) -> i64 {
        self.0.next_long()
    }
    pub fn boolean(&mut self) -> bool {
        self.0.next_boolean()
    }
}

/// `Coords.coordsSeedRandom(worldSeed, x, y, z)` of CubicChunks: three odd
/// multipliers drawn from the world seed, mixed with the cube coordinates.
pub fn coords_seed_random(world_seed: i64, x: i32, y: i32, z: i32) -> JRandom {
    let mut rnd = JRandom::new(world_seed);
    let x_seed = (rnd.long() / 2 * 2).wrapping_add(1);
    let y_seed = (rnd.long() / 2 * 2).wrapping_add(1);
    let z_seed = (rnd.long() / 2 * 2).wrapping_add(1);
    let seed = x_seed
        .wrapping_mul(i64::from(x))
        .wrapping_add(y_seed.wrapping_mul(i64::from(y)))
        .wrapping_add(z_seed.wrapping_mul(i64::from(z)))
        ^ world_seed;
    JRandom::new(seed)
}

/// A 64-bit hash of a seed, a salt and a position (Stafford mix 13 rounds; CWG
/// uses xxHash64 / Murmur here, so values differ but are as uniform).
pub fn hash_pos(seed: i64, salt: i32, x: i32, y: i32, z: i32) -> u64 {
    let mut h = mix_stafford13(seed as u64 ^ 0x9e37_79b9_7f4a_7c15);
    h = mix_stafford13(h ^ (salt as u32 as u64));
    h = mix_stafford13(h ^ (x as u32 as u64) ^ ((z as u32 as u64) << 32));
    mix_stafford13(h ^ (y as u32 as u64).wrapping_mul(0xff51_afd7_ed55_8ccd))
}

/// `hash_pos` as a number in `[0, 1)`.
pub fn hash_unit(seed: i64, salt: i32, x: i32, y: i32, z: i32) -> f64 {
    (hash_pos(seed, salt, x, y, z) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// `sin` of an `f32` with the same result on every platform.
#[inline]
pub fn sin_f32(v: f32) -> f32 {
    libm::sinf(v)
}
/// `cos` of an `f32` with the same result on every platform.
#[inline]
pub fn cos_f32(v: f32) -> f32 {
    libm::cosf(v)
}
