//! Ore placement with ordered replacement rules and air-exposure checks.
use super::{
    DecodeError, Object,
    block_feature::{BlockReplacements, FeatureBlockBindings, decode_targets},
    density::identifier,
    feature::{FeatureError, FeatureWorld},
    feature_graph::FeatureExecutionError,
    int_range, number_text,
    placement::PlacementContext,
    required,
};
use crate::heightmap::HeightmapType;
use crate::registry::BlockStateId;
use horizon_core::{geometry::Direction, random::RandomSource};

#[derive(Clone, Debug)]
pub struct ScatteredOreFeature {
    targets: BlockReplacements,
    size: i32,
    discard: f32,
    air: [BlockStateId; 3],
}
impl ScatteredOreFeature {
    pub fn from_json(json: &str, bindings: &FeatureBlockBindings<'_>) -> Result<Self, DecodeError> {
        Self::decode(json, bindings, 0, &mut 65536)
    }
    pub(crate) fn decode(
        json: &str,
        bindings: &FeatureBlockBindings<'_>,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Self, DecodeError> {
        let object: Object = serde_json::from_str(json)?;
        let kind = identifier(&serde_json::from_str::<String>(
            required(&object, "type")?.get(),
        )?)?;
        if kind != "minecraft:scattered_ore" {
            return Err(DecodeError::Field("scattered ore type"));
        }
        let targets = decode_targets(&object, bindings, depth, remaining)?;
        let size = int_range(required(&object, "size")?, "ore size", 0, 64)?;
        let discard: f32 = number_text(required(&object, "discard_chance_on_air_exposure")?)?
            .parse()
            .map_err(|_| DecodeError::Field("ore air exposure"))?;
        if !(0.0..=1.0).contains(&discard) || discard.is_sign_negative() {
            return Err(DecodeError::Field("ore air exposure"));
        }
        Ok(Self {
            targets,
            size,
            discard,
            air: bindings.air,
        })
    }
    pub fn place<W: FeatureWorld>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        origin: [i32; 3],
    ) -> Result<bool, FeatureError<W::Error>> {
        let count = random
            .next_int_bound(self.size + 1)
            .map_err(FeatureError::Random)?;
        for attempt in 0..count {
            let spread = attempt.min(7) as f32;
            let position = origin.map(|component| {
                let offset = (random.next_float() - random.next_float()) * spread;
                // Widen before adding 0.5 so a float just below a tie cannot
                // round up during addition. Values are bounded by +/-7.
                let offset = (f64::from(offset) + 0.5).floor() as i32;
                component.wrapping_add(offset)
            });
            let current = world.block(position).map_err(FeatureError::World)?;
            for (rule, state) in &self.targets {
                if !rule.test(current, position, random) {
                    continue;
                }
                let skip_air = self.discard <= 0.0
                    || (self.discard < 1.0 && random.next_float() >= self.discard);
                if !skip_air {
                    let mut exposed = false;
                    for direction in Direction::ALL {
                        let delta = direction.normal();
                        let neighbor = std::array::from_fn(|i| position[i].wrapping_add(delta[i]));
                        if self
                            .air
                            .contains(&world.block(neighbor).map_err(FeatureError::World)?)
                        {
                            exposed = true;
                            break;
                        }
                    }
                    if exposed {
                        continue;
                    }
                }
                world
                    .set_block(position, *state, 2)
                    .map_err(FeatureError::World)?;
                break;
            }
        }
        Ok(true)
    }
}

/// Vanilla `minecraft:ore`: an interpolated chain of spheres written straight
/// into chunk sections (BulkSectionAccess), bypassing block update effects.
#[derive(Clone, Debug)]
pub struct OreFeature {
    targets: BlockReplacements,
    size: i32,
    discard: f32,
    air: [BlockStateId; 3],
}
impl OreFeature {
    pub fn from_json(json: &str, bindings: &FeatureBlockBindings<'_>) -> Result<Self, DecodeError> {
        Self::decode(json, bindings, 0, &mut 65536)
    }
    pub(crate) fn decode(
        json: &str,
        bindings: &FeatureBlockBindings<'_>,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<Self, DecodeError> {
        let object: Object = serde_json::from_str(json)?;
        let kind = identifier(&serde_json::from_str::<String>(
            required(&object, "type")?.get(),
        )?)?;
        if kind != "minecraft:ore" {
            return Err(DecodeError::Field("ore type"));
        }
        let targets = decode_targets(&object, bindings, depth, remaining)?;
        let size = int_range(required(&object, "size")?, "ore size", 0, 64)?;
        let discard: f32 = number_text(required(&object, "discard_chance_on_air_exposure")?)?
            .parse()
            .map_err(|_| DecodeError::Field("ore air exposure"))?;
        if !(0.0..=1.0).contains(&discard) || discard.is_sign_negative() {
            return Err(DecodeError::Field("ore air exposure"));
        }
        Ok(Self {
            targets,
            size,
            discard,
            air: bindings.air,
        })
    }
    pub fn place<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        origin: [i32; 3],
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        let dir = random.next_float() * std::f32::consts::PI;
        let spread_xy = self.size as f32 / 8.0;
        let max_radius = ((self.size as f32 / 16.0 * 2.0 + 1.0) / 2.0).ceil() as i32;
        let (ox, oz) = (f64::from(origin[0]), f64::from(origin[2]));
        let (sin, cos) = (
            horizon_core::jmath::sin(f64::from(dir)),
            horizon_core::jmath::cos(f64::from(dir)),
        );
        let spread = f64::from(spread_xy);
        let x0 = ox + sin * spread;
        let x1 = ox - sin * spread;
        let z0 = oz + cos * spread;
        let z1 = oz - cos * spread;
        let y0 = f64::from(
            origin[1]
                .wrapping_add(random.next_int_bound(3)?)
                .wrapping_sub(2),
        );
        let y1 = f64::from(
            origin[1]
                .wrapping_add(random.next_int_bound(3)?)
                .wrapping_sub(2),
        );
        let spread_ceil = spread.ceil() as i32;
        let x_start = origin[0].wrapping_sub(spread_ceil).wrapping_sub(max_radius);
        let y_start = origin[1].wrapping_sub(2).wrapping_sub(max_radius);
        let z_start = origin[2].wrapping_sub(spread_ceil).wrapping_sub(max_radius);
        let size_xz = 2 * (spread_ceil + max_radius);
        let size_y = 2 * (2 + max_radius);
        if let Some(found) = world
            .any_height_at_least(
                HeightmapType::OceanFloorWg,
                [x_start, z_start],
                size_xz,
                y_start,
            )
            .map_err(FeatureExecutionError::Query)?
        {
            return match found {
                true => self.do_place(
                    world,
                    random,
                    [x0, x1, z0, z1, y0, y1],
                    [x_start, y_start, z_start],
                    size_xz,
                    size_y,
                ),
                false => Ok(false),
            };
        }
        for x in 0..=size_xz {
            for z in 0..=size_xz {
                let (px, pz) = (x_start.wrapping_add(x), z_start.wrapping_add(z));
                let floor = world
                    .height(HeightmapType::OceanFloorWg, px, pz)
                    .map_err(FeatureExecutionError::Query)?;
                if y_start <= floor {
                    return self.do_place(
                        world,
                        random,
                        [x0, x1, z0, z1, y0, y1],
                        [x_start, y_start, z_start],
                        size_xz,
                        size_y,
                    );
                }
            }
        }
        Ok(false)
    }
    fn do_place<W: FeatureWorld + PlacementContext>(
        &self,
        world: &mut W,
        random: &mut impl RandomSource,
        [x0, x1, z0, z1, y0, y1]: [f64; 6],
        [x_start, y_start, z_start]: [i32; 3],
        size_xz: i32,
        size_y: i32,
    ) -> Result<bool, FeatureExecutionError<W::Error>> {
        let size = self.size as usize;
        let mut data = vec![[0f64; 4]; size];
        for (i, entry) in data.iter_mut().enumerate() {
            let step = i as f32 / self.size as f32;
            let step64 = f64::from(step);
            let xx = x0 + step64 * (x1 - x0);
            let yy = y0 + step64 * (y1 - y0);
            let zz = z0 + step64 * (z1 - z0);
            let ss = random.next_double() * f64::from(self.size) / 16.0;
            let sine = horizon_core::geometry::trig::sin(f64::from(std::f32::consts::PI * step));
            let r = (f64::from(sine + 1.0) * ss + 1.0) / 2.0;
            *entry = [xx, yy, zz, r];
        }
        for i1 in 0..size.saturating_sub(1) {
            if data[i1][3] <= 0.0 {
                continue;
            }
            for i2 in i1 + 1..size {
                if data[i2][3] <= 0.0 {
                    continue;
                }
                let dx = data[i1][0] - data[i2][0];
                let dy = data[i1][1] - data[i2][1];
                let dz = data[i1][2] - data[i2][2];
                let dr = data[i1][3] - data[i2][3];
                if dr * dr > dx * dx + dy * dy + dz * dz {
                    if dr > 0.0 {
                        data[i2][3] = -1.0;
                    } else {
                        data[i1][3] = -1.0;
                    }
                }
            }
        }
        let mut placed = 0;
        sphere_blocks(
            &mut (&mut *world, &mut *random, &mut placed),
            &data,
            [x_start, y_start, z_start],
            [size_xz, size_y],
            || FeatureExecutionError::WorkLimit,
            |(world, _, _), y| {
                world
                    .outside_build_height(y)
                    .map_err(FeatureExecutionError::Query)
            },
            |(world, random, placed), p| {
                if !world
                    .ensure_can_write(p)
                    .map_err(FeatureExecutionError::Query)?
                {
                    return Ok(());
                }
                let Some(state) = world
                    .section_block(p)
                    .map_err(FeatureExecutionError::Query)?
                else {
                    return Ok(());
                };
                for (rule, target) in &self.targets {
                    if !rule.test(state, p, *random) {
                        continue;
                    }
                    if !skip_air_check(*random, self.discard) {
                        let mut exposed = false;
                        for direction in Direction::ALL {
                            let n = direction.normal();
                            let q = std::array::from_fn(|i| p[i].wrapping_add(n[i]));
                            let neighbor = world
                                .section_block(q)
                                .map_err(FeatureExecutionError::Query)?
                                .unwrap_or(self.air[0]);
                            if self.air.contains(&neighbor) {
                                exposed = true;
                                break;
                            }
                        }
                        if exposed {
                            continue;
                        }
                    }
                    world
                        .set_section_block(p, *target)
                        .map_err(FeatureExecutionError::Query)?;
                    **placed += 1;
                    break;
                }
                Ok(())
            },
        )?;
        Ok(placed > 0)
    }
}
/// The blocks OreFeature.doPlace tests, in its order: sphere by sphere, X, Y
/// then Z ascending over each sphere's box, the ones inside the sphere and
/// within build height (`outside` is asked only for those), each the first
/// time any sphere reaches it (vanilla's BitSet). The squared distances are
/// the same expressions as vanilla's, computed once per sphere and axis value
/// instead of per block, and each Z row's tested blocks are one word.
#[allow(clippy::too_many_arguments)]
fn sphere_blocks<C, E>(
    state: &mut C,
    spheres: &[[f64; 4]],
    [x_start, y_start, z_start]: [i32; 3],
    [size_xz, size_y]: [i32; 2],
    limit: impl Fn() -> E,
    mut outside: impl FnMut(&mut C, i32) -> Result<bool, E>,
    mut visit: impl FnMut(&mut C, [i32; 3]) -> Result<(), E>,
) -> Result<(), E> {
    // Vanilla indexes its BitSet by x + y * size_xz + z * size_xz * size_y;
    // blocks inside a sphere never leave the box, so a row per X and Y with a
    // bit per Z is the same set.
    if !(0..64).contains(&size_xz) || size_y < 0 {
        return Err(limit());
    }
    let mut tested = vec![0u64; size_xz as usize * size_y as usize];
    let (mut ysq, mut zsq) = (Vec::new(), Vec::new());
    for &[xx, yy, zz, r] in spheres {
        if r < 0.0 {
            continue;
        }
        let x_min = ((xx - r).floor() as i32).max(x_start);
        let y_min = ((yy - r).floor() as i32).max(y_start);
        let z_min = ((zz - r).floor() as i32).max(z_start);
        let x_max = ((xx + r).floor() as i32).max(x_min);
        let y_max = ((yy + r).floor() as i32).max(y_min);
        let z_max = ((zz + r).floor() as i32).max(z_min);
        if i64::from(z_max) - i64::from(z_start) >= 64 {
            return Err(limit());
        }
        ysq.clear();
        ysq.extend((y_min..=y_max).map(|y| {
            let yd = (f64::from(y) + 0.5 - yy) / r;
            yd * yd
        }));
        zsq.clear();
        zsq.extend((z_min..=z_max).map(|z| {
            let zd = (f64::from(z) + 0.5 - zz) / r;
            zd * zd
        }));
        let z_shift = (z_min - z_start) as u32;
        for x in x_min..=x_max {
            let xd = (f64::from(x) + 0.5 - xx) / r;
            let xsq = xd * xd;
            if xsq >= 1.0 {
                continue;
            }
            for (y, &ys) in (y_min..=y_max).zip(&ysq) {
                let xy = xsq + ys;
                if xy >= 1.0 {
                    continue;
                }
                let mut inside = 0u64;
                for (i, &zs) in zsq.iter().enumerate() {
                    inside |= u64::from(xy + zs < 1.0) << i;
                }
                if inside == 0 || outside(state, y)? {
                    continue;
                }
                let inside = inside << z_shift;
                let (dx, dy) = (x - x_start, y - y_start);
                if dx >= size_xz || dy >= size_y || inside >> size_xz != 0 {
                    return Err(limit());
                }
                let row = &mut tested[dx as usize * size_y as usize + dy as usize];
                let mut fresh = inside & !*row;
                *row |= inside;
                while fresh != 0 {
                    let dz = fresh.trailing_zeros() as i32;
                    fresh &= fresh - 1;
                    visit(state, [x, y, z_start + dz])?;
                }
            }
        }
    }
    Ok(())
}
fn skip_air_check(random: &mut impl RandomSource, discard: f32) -> bool {
    if discard <= 0.0 {
        true
    } else if discard >= 1.0 {
        false
    } else {
        random.next_float() >= discard
    }
}

#[cfg(test)]
mod tests {
    use super::sphere_blocks;

    /// OreFeature.doPlace's loop as vanilla writes it.
    fn reference(
        spheres: &[[f64; 4]],
        [x_start, y_start, z_start]: [i32; 3],
        [size_xz, size_y]: [i32; 2],
        bottom: i32,
    ) -> Vec<[i32; 3]> {
        let mut tested = vec![false; (size_xz * size_y * size_xz) as usize];
        let mut out = Vec::new();
        for &[xx, yy, zz, r] in spheres {
            if r < 0.0 {
                continue;
            }
            let x_min = ((xx - r).floor() as i32).max(x_start);
            let y_min = ((yy - r).floor() as i32).max(y_start);
            let z_min = ((zz - r).floor() as i32).max(z_start);
            let x_max = ((xx + r).floor() as i32).max(x_min);
            let y_max = ((yy + r).floor() as i32).max(y_min);
            let z_max = ((zz + r).floor() as i32).max(z_min);
            for x in x_min..=x_max {
                let xd = (f64::from(x) + 0.5 - xx) / r;
                if xd * xd >= 1.0 {
                    continue;
                }
                for y in y_min..=y_max {
                    let yd = (f64::from(y) + 0.5 - yy) / r;
                    if xd * xd + yd * yd >= 1.0 {
                        continue;
                    }
                    for z in z_min..=z_max {
                        let zd = (f64::from(z) + 0.5 - zz) / r;
                        if xd * xd + yd * yd + zd * zd >= 1.0 || y < bottom {
                            continue;
                        }
                        let index = (x - x_start)
                            + (y - y_start) * size_xz
                            + (z - z_start) * size_xz * size_y;
                        if !std::mem::replace(&mut tested[index as usize], true) {
                            out.push([x, y, z]);
                        }
                    }
                }
            }
        }
        out
    }

    /// The blobs OreFeature.place builds, from a small LCG.
    #[test]
    fn sphere_blocks_match_vanilla_order() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..400 {
            let size = 1 + (next() * 64.0) as i32;
            let origin = [(next() * 40.0) as i32 - 20, (next() * 40.0) as i32 - 20, 7];
            let dir = next() as f32 * std::f32::consts::PI;
            let spread = f64::from(size as f32 / 8.0);
            let max_radius = ((size as f32 / 16.0 * 2.0 + 1.0) / 2.0).ceil() as i32;
            let (sin, cos) = (f64::from(dir).sin(), f64::from(dir).cos());
            let [x0, x1] = [1.0, -1.0].map(|s| f64::from(origin[0]) + s * sin * spread);
            let [z0, z1] = [1.0, -1.0].map(|s| f64::from(origin[2]) + s * cos * spread);
            let [y0, y1] = [0, 0].map(|_| f64::from(origin[1] + (next() * 3.0) as i32 - 2));
            let spheres: Vec<[f64; 4]> = (0..size)
                .map(|i| {
                    let step = f64::from(i as f32 / size as f32);
                    let ss = next() * f64::from(size) / 16.0;
                    let sine = (std::f64::consts::PI * step).sin();
                    let r = if next() < 0.1 {
                        -1.0
                    } else {
                        ((sine + 1.0) * ss + 1.0) / 2.0
                    };
                    [
                        x0 + step * (x1 - x0),
                        y0 + step * (y1 - y0),
                        z0 + step * (z1 - z0),
                        r,
                    ]
                })
                .collect();
            let spread_ceil = spread.ceil() as i32;
            let start = [
                origin[0] - spread_ceil - max_radius,
                origin[1] - 2 - max_radius,
                origin[2] - spread_ceil - max_radius,
            ];
            let sizes = [2 * (spread_ceil + max_radius), 2 * (2 + max_radius)];
            let bottom = if case % 3 == 0 { origin[1] } else { i32::MIN };
            let mut got = Vec::new();
            sphere_blocks(
                &mut got,
                &spheres,
                start,
                sizes,
                || (),
                |_, y| Ok(y < bottom),
                |got, p| {
                    got.push(p);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(
                got,
                reference(&spheres, start, sizes, bottom),
                "case {case}"
            );
        }
    }
}
