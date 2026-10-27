//! Compiled material rule composition, sharing the condition worker's context.
use super::{
    Compiled, Context, DensityHeights, MaterialCondition, MaterialConditions,
    MaterialSurfaceQueries, RUN_GUARDS, RunTrace,
};
use crate::{registry::BlockStateId, worldgen::DecodeError};
use horizon_core::{
    density::DensitySampler,
    position::BlockPos,
    random::{PositionalRandom, RandomSource},
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum MaterialRule {
    Block(BlockStateId),
    Reference(Arc<MaterialRule>),
    /// Density roots must be compiled for the same world and sampler context.
    OreVein {
        ore: BlockStateId,
        raw_ore: BlockStateId,
        filler: BlockStateId,
        raw_ore_chance: f32,
        density: Arc<dyn DensitySampler>,
        richness: Arc<dyn DensitySampler>,
        filler_gap: Arc<dyn DensitySampler>,
        /// What the density's definition tells about where it can be above zero.
        gate: Option<DensityGate>,
    },
    Sequence(Vec<MaterialRule>),
    Condition {
        if_true: MaterialCondition,
        then_run: Box<MaterialRule>,
    },
    /// Vanilla calls this rule "bandlands"; it queries terracotta bands.
    Bandlands,
}
/// A density that is a constant `<= 0` wherever `input`, a function of Y
/// alone, lies outside `min_inclusive..max_exclusive` (vanilla's ore veins
/// are a `range_choice` on Y).
#[derive(Clone, Debug)]
pub struct DensityGate {
    pub input: Arc<dyn DensitySampler>,
    pub min_inclusive: f32,
    pub max_exclusive: f32,
    /// For the density and the richness: a Y period of their volume samples
    /// (moving a volume's minimum Y by a multiple of it keeps every shared
    /// block's value), when known.
    pub lattices: [Option<i32>; 2],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialRuleError {
    UnknownRule,
    UnknownCondition,
    MissingBiomeProvider,
    BiomeQueryFailed,
    MissingBiome,
    CachedConditionUnavailable,
    MissingBandProvider,
    MissingDensityProvider,
    Density(horizon_core::density::SampleError),
    InvalidBandIndex,
}
impl std::fmt::Display for MaterialRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Self::Density(error) = self {
            return error.fmt(f);
        }
        f.write_str(match self {
            Self::Density(_) => unreachable!(),
            Self::MissingDensityProvider => "missing material density provider",
            Self::UnknownRule => "unknown material rule",
            Self::UnknownCondition => "unknown material condition",
            Self::MissingBiomeProvider => "missing material biome provider",
            Self::BiomeQueryFailed => "material biome lookup failed",
            Self::MissingBiome => "material position has no biome",
            Self::CachedConditionUnavailable => "condition was updated without a cached result",
            Self::MissingBandProvider => "missing material band provider",
            Self::InvalidBandIndex => "negative material band index",
        })
    }
}
impl std::error::Error for MaterialRuleError {}
impl From<horizon_core::density::SampleError> for MaterialRuleError {
    fn from(error: horizon_core::density::SampleError) -> Self {
        Self::Density(error)
    }
}

enum Node<R: RandomSource> {
    Empty,
    Block(BlockStateId),
    OreVein {
        blocks: [BlockStateId; 3],
        raw_ore_chance: f32,
        densities: [usize; 3],
        /// Where the eagerly sampled vein density is known to be `<= 0`.
        heights: Option<DensityHeights>,
        random: R::Positional,
    },
    Sequence(Vec<Self>),
    Condition {
        condition: Compiled<R>,
        rule: Box<Self>,
    },
    Bandlands,
}
impl<R: RandomSource> Node<R> {
    fn compile<S: MaterialSurfaceQueries>(
        rule: MaterialRule,
        root: &R::Positional,
        context: &mut Context<S>,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<Self, DecodeError> {
        *nodes += 1;
        if depth >= 256 || *nodes > 65536 {
            return Err(DecodeError::Field("material rule size"));
        }
        Ok(match rule {
            MaterialRule::Block(block) => Self::Block(block),
            MaterialRule::Reference(value) => {
                return Self::compile((*value).clone(), root, context, depth + 1, nodes);
            }
            MaterialRule::OreVein {
                ore,
                raw_ore,
                filler,
                raw_ore_chance,
                density,
                richness,
                filler_gap,
                gate,
            } => {
                let (density, richness) = match &gate {
                    // The richness is only read where the density is above zero.
                    Some(gate) => (
                        context.surface.bind_gated_density(
                            density,
                            gate,
                            true,
                            gate.lattices[0],
                        )?,
                        context.surface.bind_gated_density(
                            richness,
                            gate,
                            false,
                            gate.lattices[1],
                        )?,
                    ),
                    None => (
                        context.surface.bind_density(density, true)?,
                        context.surface.bind_density(richness, true)?,
                    ),
                };
                let gap = context.surface.bind_density(filler_gap, false)?;
                let random = root.from_hash_of("minecraft:ore").fork_positional();
                Self::OreVein {
                    blocks: [ore, raw_ore, filler],
                    raw_ore_chance,
                    densities: [density, richness, gap],
                    heights: context.surface.density_heights(density),
                    random,
                }
            }
            MaterialRule::Bandlands => Self::Bandlands,
            MaterialRule::Condition { if_true, then_run } => {
                // Compile the predicate before the nested rule, regardless of
                // whether the predicate will match any subsequent position.
                let condition = Compiled::compile(if_true, root, context, depth + 1)?;
                let rule = Box::new(Self::compile(*then_run, root, context, depth + 1, nodes)?);
                // Compile both sides first: binding and errors must retain their order.
                match condition.constant_result() {
                    Some(false) => Self::Empty,
                    Some(true) => *rule,
                    None => Self::Condition { condition, rule },
                }
            }
            MaterialRule::Sequence(mut children) => {
                if children.len() == 1 {
                    return Self::compile(children.pop().unwrap(), root, context, depth + 1, nodes);
                }
                let mut result = Vec::with_capacity(children.len());
                for child in children {
                    let child = Self::compile(child, root, context, depth + 1, nodes)?;
                    if !matches!(child, Self::Empty) {
                        result.push(child);
                    }
                }
                match result.len() {
                    0 => Self::Empty,
                    1 => result.pop().unwrap(),
                    _ => Self::Sequence(result),
                }
            }
        })
    }
    /// Lay the tree out in evaluation order: a condition that fails skips
    /// its subtree, a subtree that gives nothing falls through to the next
    /// sibling, and the first block-producing step ends the walk.
    fn flatten(self, rule: &mut CompiledRule<R>) {
        match self {
            Self::Empty => {}
            Self::Block(block) => rule.ops.push(Op::Block(block)),
            Self::OreVein {
                blocks,
                raw_ore_chance,
                densities,
                heights,
                random,
            } => rule.ops.push(Op::OreVein {
                blocks,
                raw_ore_chance,
                densities,
                heights,
                random,
            }),
            Self::Bandlands => rule.ops.push(Op::Bandlands),
            Self::Condition {
                condition,
                rule: then,
            } => {
                let at = rule.ops.len();
                rule.ops.push(Op::Condition {
                    condition: rule.conditions.len(),
                    skip: 0,
                });
                rule.conditions.push(condition);
                then.flatten(rule);
                let end = rule.ops.len();
                if let Op::Condition { skip, .. } = &mut rule.ops[at] {
                    *skip = end;
                }
            }
            Self::Sequence(children) => {
                for child in children {
                    child.flatten(rule);
                }
            }
        }
    }
}

/// A compiled rule as a flat program (see `Node::flatten`): no recursion or
/// per-node call for each block.
pub(super) struct CompiledRule<R: RandomSource> {
    ops: Vec<Op<R>>,
    conditions: Vec<Compiled<R>>,
}
enum Op<R: RandomSource> {
    Block(BlockStateId),
    OreVein {
        blocks: [BlockStateId; 3],
        raw_ore_chance: f32,
        densities: [usize; 3],
        heights: Option<DensityHeights>,
        random: R::Positional,
    },
    Bandlands,
    /// Test `conditions[condition]`; when false, continue at `skip`.
    Condition {
        condition: usize,
        skip: usize,
    },
}
impl<R: RandomSource> CompiledRule<R> {
    fn compile<S: MaterialSurfaceQueries>(
        rule: MaterialRule,
        root: &R::Positional,
        context: &mut Context<S>,
    ) -> Result<Self, DecodeError> {
        let node = Node::compile(rule, root, context, 0, &mut 0)?;
        let mut compiled = Self {
            ops: Vec::new(),
            conditions: Vec::new(),
        };
        node.flatten(&mut compiled);
        Ok(compiled)
    }
    fn apply<S: MaterialSurfaceQueries>(
        &mut self,
        position: BlockPos,
        context: &mut Context<S>,
    ) -> Result<Option<BlockStateId>, MaterialRuleError> {
        let mut at = 0;
        while let Some(op) = self.ops.get(at) {
            match op {
                Op::Block(block) => return Ok(Some(*block)),
                Op::Bandlands => return context.surface.band(position).map(Some),
                Op::Condition { condition, skip } => {
                    at = if self.conditions[*condition].test_once(context)? {
                        at + 1
                    } else {
                        *skip
                    };
                }
                Op::OreVein {
                    blocks,
                    raw_ore_chance,
                    densities,
                    heights,
                    random,
                } => {
                    if let Some(block) = ore_vein(
                        blocks,
                        *raw_ore_chance,
                        densities,
                        heights,
                        random,
                        position,
                        context,
                    )? {
                        return Ok(Some(block));
                    }
                    at += 1;
                }
            }
        }
        Ok(None)
    }
    /// `apply`, also telling down to which height of the current run of
    /// solid blocks (and for which biome) the result holds: see `RunMemo`.
    fn apply_traced<S: MaterialSurfaceQueries>(
        &mut self,
        position: BlockPos,
        context: &mut Context<S>,
        run: &mut RunTrace,
    ) -> Result<Option<BlockStateId>, MaterialRuleError> {
        let mut at = 0;
        while let Some(op) = self.ops.get(at) {
            match op {
                Op::Block(block) => return Ok(Some(*block)),
                Op::Bandlands => {
                    run.varies = true;
                    return context.surface.band(position).map(Some);
                }
                Op::Condition { condition, skip } => {
                    let index = *condition;
                    let condition = &self.conditions[index];
                    let value = condition.test_once(context)?;
                    if !run.varies && condition.stable_below(value, context, run) {
                        if run.guard_count < RUN_GUARDS {
                            run.guards[run.guard_count] = (index as u32, value);
                            run.guard_count += 1;
                        } else {
                            run.varies = true;
                        }
                    }
                    at = if value { at + 1 } else { *skip };
                }
                Op::OreVein {
                    blocks,
                    raw_ore_chance,
                    densities,
                    heights,
                    random,
                } => {
                    match heights {
                        Some(heights) if heights.never_positive(context.position) => {
                            let y = context.position.y;
                            let low = if y > heights.y[1] {
                                heights.y[1].wrapping_add(1).max(heights.min[1])
                            } else {
                                heights.min[1]
                            };
                            run.low = run.low.max(i64::from(low));
                        }
                        _ => run.varies = true,
                    }
                    if let Some(block) = ore_vein(
                        blocks,
                        *raw_ore_chance,
                        densities,
                        heights,
                        random,
                        position,
                        context,
                    )? {
                        return Ok(Some(block));
                    }
                    at += 1;
                }
            }
        }
        Ok(None)
    }
}
/// Results of a rule program down one run of solid blocks of a column (no
/// air or fluid in between: the stone above grows by one a block, while the
/// water height, the stone below's bottom and the column stay). Each entry
/// holds while every condition its program run tested keeps its value, which
/// for the tests that are height bounds is down to a known height; programs
/// that tested something varying block by block (3D noise, gradients inside
/// their band, ore veins where they may be dense, bands, temperatures) give no
/// entry. Clear it whenever the run ends.
#[derive(Default)]
pub struct RunMemo {
    entries: Vec<RunEntry>,
    /// The biome of the run's last biome lookup, and the height down to
    /// which the column keeps it.
    biome: Option<(u32, i32)>,
}
#[derive(Clone, Copy)]
struct RunEntry {
    /// The biome the program tested, `None` when it tested none.
    biome: Option<u32>,
    low: i64,
    /// Tests to give these values again at the block (`RunTrace::guards`).
    guards: [(u32, bool); RUN_GUARDS],
    guard_count: usize,
    result: Option<BlockStateId>,
}
impl RunMemo {
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
        self.biome = None;
    }
}
fn ore_vein<P: PositionalRandom, S: MaterialSurfaceQueries>(
    blocks: &[BlockStateId; 3],
    raw_ore_chance: f32,
    densities: &[usize; 3],
    heights: &Option<DensityHeights>,
    random: &P,
    position: BlockPos,
    context: &mut Context<S>,
) -> Result<Option<BlockStateId>, MaterialRuleError> {
    // Most blocks lie outside a vein's heights: the density read
    // would give `<= 0`, and nothing happens before it.
    if heights.is_some_and(|heights| heights.never_positive(context.position)) {
        return Ok(None);
    }
    // Densities use the context; randomness uses application coordinates.
    let density = context.surface.density(densities[0], context.position)?;
    if density <= 0.0 {
        return Ok(None);
    }
    let mut random = random.at(position);
    if random.next_float() > density {
        return Ok(None);
    }
    let richness = context.surface.density(densities[1], context.position)?;
    if random.next_float() < richness
        && context.surface.density(densities[2], context.position)? < 0.0
    {
        Ok(Some(if random.next_float() < raw_ore_chance {
            blocks[1]
        } else {
            blocks[0]
        }))
    } else {
        Ok(Some(blocks[2]))
    }
}
impl<R: RandomSource, S: MaterialSurfaceQueries> MaterialConditions<R, S> {
    /// Compile all children in order, even after an unconditional block rule.
    /// Failed compilation never adds a partial rule to this worker.
    pub fn add_rule(&mut self, rule: MaterialRule) -> Result<usize, DecodeError> {
        if self.rules.len() >= 8192 {
            return Err(DecodeError::Field("material rule count"));
        }
        let compiled = CompiledRule::compile(rule, &self.root, &mut self.context)?;
        let index = self.rules.len();
        self.rules.push(compiled);
        Ok(index)
    }
    /// `update_y(stone_above, stone_below, water_height, position.y)` then
    /// `try_apply(index, position)`, for the blocks of a column's runs of
    /// solid blocks walked down in y, reusing `memo`'s results where they
    /// provably hold (the caller clears `memo` when a run ends). Also gives
    /// a height down to which the result is the same for every block of the
    /// run (`position.y` when not known; `floor`, the run's lowest block,
    /// bounds how far down it looks).
    #[allow(clippy::too_many_arguments)]
    pub fn try_apply_in_run(
        &mut self,
        index: usize,
        position: BlockPos,
        memo: &mut RunMemo,
        stone_above: i32,
        stone_below: i32,
        water_height: i32,
        floor: i32,
    ) -> Result<(Option<BlockStateId>, i32), MaterialRuleError> {
        let y = i64::from(position.y);
        // The block's biome and its span, looked up when an entry asks.
        let mut biome: Option<Option<(u32, i32)>> = None;
        let mut found = None;
        let mut guard_y = false;
        let rules = &self.rules;
        let context = &mut self.context;
        'entries: for (at, entry) in memo.entries.iter().enumerate() {
            if y < entry.low {
                continue;
            }
            let span = match entry.biome {
                None => floor,
                Some(want) => {
                    let span = *biome.get_or_insert_with(|| match memo.biome {
                        Some((id, low)) if position.y >= low => Some((id, low)),
                        // A failed lookup is left for the program to report, if it asks.
                        _ => context
                            .surface
                            .biome_id_down(position, floor)
                            .ok()
                            .flatten(),
                    });
                    match span {
                        Some((id, low)) if id == want => low,
                        _ => continue,
                    }
                }
            };
            if entry.guard_count == 0 {
                found = Some((at, entry.result, entry.low.max(i64::from(span)) as i32));
                break;
            }
            // Position-pure tests, read at this block.
            if !guard_y {
                context.position.y = position.y;
                context.noise_floor = floor;
                guard_y = true;
            }
            let rule = rules.get(index).ok_or(MaterialRuleError::UnknownRule)?;
            for &(condition, value) in &entry.guards[..entry.guard_count] {
                if rule.conditions[condition as usize].test_once(context)? != value {
                    continue 'entries;
                }
            }
            found = Some((at, entry.result, position.y));
            break;
        }
        if let Some(span) = biome {
            memo.biome = span;
        }
        if let Some((at, result, low)) = found {
            // The next block most likely takes the same entry.
            memo.entries.swap(0, at);
            return Ok((result, low));
        }
        let biome = biome.flatten().map(|(id, _)| id);
        self.update_y(stone_above, stone_below, water_height, position.y);
        self.context.biome_id = biome;
        self.context.noise_floor = floor;
        let rule = self
            .rules
            .get_mut(index)
            .ok_or(MaterialRuleError::UnknownRule)?;
        let mut run = RunTrace {
            low: i64::MIN,
            biome: false,
            guards: [(0, false); RUN_GUARDS],
            guard_count: 0,
            varies: false,
        };
        let result = rule.apply_traced(position, &mut self.context, &mut run)?;
        let key = match (run.biome, self.context.biome_id) {
            (false, _) => Some(None),
            (true, Some(id)) => Some(Some(id)),
            (true, None) => None,
        };
        if let Some(biome) = key.filter(|_| !run.varies && run.low <= y) {
            // Entries ending above this block no longer serve.
            memo.entries.retain(|entry| entry.low <= y);
            if memo.entries.len() < 16 {
                memo.entries.push(RunEntry {
                    biome,
                    low: run.low,
                    guards: run.guards,
                    guard_count: run.guard_count,
                    result,
                });
            }
        }
        Ok((result, position.y))
    }
    /// Predicates read the worker's updated context. Position arguments are
    /// forwarded to block-producing rules independently, as in vanilla.
    pub fn try_apply(
        &mut self,
        index: usize,
        position: BlockPos,
    ) -> Result<Option<BlockStateId>, MaterialRuleError> {
        self.rules
            .get_mut(index)
            .ok_or(MaterialRuleError::UnknownRule)?
            .apply(position, &mut self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldgen::anchor::{VerticalAnchor, WorldGenerationContext};
    use horizon_core::random::{RandomSource, XoroshiroRandom};

    #[derive(Debug)]
    struct Wavy;
    impl horizon_core::noise::Noise for Wavy {
        fn range(&self) -> Result<[f32; 2], horizon_core::noise::NanRange> {
            Ok([-1.0, 1.0])
        }
        fn get2(&self, x: f64, z: f64) -> f32 {
            self.get3(x, 0.0, z)
        }
        fn get3(&self, x: f64, y: f64, z: f64) -> f32 {
            ((x * 0.7 + y * 1.3 + z * 0.3).sin() * 0.9) as f32
        }
    }
    struct Surface;
    impl MaterialSurfaceQueries for Surface {
        fn bind_noise(
            &mut self,
            _: &str,
        ) -> Result<Arc<dyn horizon_core::noise::Noise>, DecodeError> {
            Ok(Arc::new(Wavy))
        }
        fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
            (x ^ z) & 3
        }
        fn surface_secondary(&mut self, x: i32, _: i32) -> f64 {
            f64::from(x % 5) / 5.0
        }
        fn preliminary_surface(&mut self, p: BlockPos) -> Result<f32, MaterialRuleError> {
            Ok(14.5 + p.x as f32)
        }
        fn biome_index(&self, name: &str) -> Option<Option<u32>> {
            Some(name.strip_prefix("minecraft:b").map(|n| n.parse().unwrap()))
        }
        fn biome_id(&mut self, p: BlockPos) -> Result<Option<u32>, MaterialRuleError> {
            Ok(Some(((p.y.div_euclid(5) + p.x) % 3) as u32))
        }
        fn biome_id_down(
            &mut self,
            p: BlockPos,
            _: i32,
        ) -> Result<Option<(u32, i32)>, MaterialRuleError> {
            Ok(self.biome_id(p)?.map(|id| (id, p.y.div_euclid(5) * 5)))
        }
        fn band(&mut self, p: BlockPos) -> Result<BlockStateId, MaterialRuleError> {
            Ok(BlockStateId(100 + p.y.rem_euclid(3) as u32))
        }
    }
    fn when(condition: MaterialCondition, rule: MaterialRule) -> MaterialRule {
        MaterialRule::Condition {
            if_true: condition,
            then_run: Box::new(rule),
        }
    }
    fn block(id: u32) -> MaterialRule {
        MaterialRule::Block(BlockStateId(id))
    }
    fn stone_depth(
        offset: i32,
        add_surface_depth: bool,
        range: i32,
        ceiling: bool,
    ) -> MaterialCondition {
        MaterialCondition::StoneDepth {
            offset,
            add_surface_depth,
            secondary_depth_range: range,
            ceiling,
        }
    }

    fn noise(min: f64, max: f64) -> MaterialCondition {
        MaterialCondition::NoiseThreshold {
            noise: "minecraft:wavy".into(),
            min,
            max,
            is_3d: true,
        }
    }
    /// Runs memoized down a column give what the program gives block by block.
    #[test]
    fn run_memo_matches_each_block() {
        let rule = MaterialRule::Sequence(vec![
            when(
                MaterialCondition::VerticalGradient {
                    random_name: "minecraft:floor".into(),
                    true_at_and_below: VerticalAnchor::Absolute(-30),
                    false_at_and_above: VerticalAnchor::Absolute(-25),
                },
                block(1),
            ),
            when(stone_depth(0, false, 0, false), block(2)),
            when(
                MaterialCondition::Biome(vec!["minecraft:b0".into()]),
                MaterialRule::Sequence(vec![
                    when(noise(-0.4, -0.1), block(9)),
                    when(noise(0.0, 0.4), block(10)),
                    when(noise(0.4, f64::MAX), block(9)),
                ]),
            ),
            when(
                MaterialCondition::Biome(vec!["minecraft:b1".into()]),
                MaterialRule::Sequence(vec![
                    when(
                        MaterialCondition::YAbove {
                            anchor: VerticalAnchor::Absolute(10),
                            surface_depth_multiplier: 1,
                            add_stone_depth: false,
                        },
                        when(stone_depth(1, true, 4, true), block(3)),
                    ),
                    when(
                        MaterialCondition::Water {
                            offset: -2,
                            surface_depth_multiplier: 0,
                            add_stone_depth: false,
                        },
                        block(4),
                    ),
                    when(
                        MaterialCondition::Not(Box::new(MaterialCondition::YAbove {
                            anchor: VerticalAnchor::Absolute(30),
                            surface_depth_multiplier: -1,
                            add_stone_depth: true,
                        })),
                        block(5),
                    ),
                ]),
            ),
            when(
                MaterialCondition::Biome(vec!["minecraft:b2".into()]),
                when(
                    MaterialCondition::AbovePreliminarySurface,
                    MaterialRule::Bandlands,
                ),
            ),
            when(
                MaterialCondition::Water {
                    offset: 0,
                    surface_depth_multiplier: 1,
                    add_stone_depth: true,
                },
                when(stone_depth(3, true, 0, false), block(6)),
            ),
            when(
                MaterialCondition::Not(Box::new(stone_depth(2, false, 3, true))),
                when(MaterialCondition::Hole, block(7)),
            ),
            when(MaterialCondition::Steep, block(8)),
        ]);
        let new = || {
            let mut worker = MaterialConditions::<XoroshiroRandom, _>::new(
                WorldGenerationContext::new(-32, 96, 20),
                XoroshiroRandom::new(5).fork_positional(),
                Surface,
            );
            let index = worker.add_rule(rule.clone()).unwrap();
            (worker, index)
        };
        let (mut memoized, rule_m) = new();
        let (mut plain, rule_p) = new();
        let mut memo = RunMemo::default();
        let mut seed = 12345u64;
        let mut hits = 0;
        let mut long_spans = 0;
        for column in 0..40 {
            let (x, z) = (column % 7 - 3, column / 7);
            // 0 air, 1 fluid, 2 solid; from y 63 down to -32.
            let mut kinds = Vec::new();
            while kinds.len() < 96 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let kind = [2, 2, 2, 0, 1][(seed >> 33) as usize % 5];
                let len = 1 + (seed >> 40) as usize % 30;
                kinds.extend(std::iter::repeat_n(kind, len));
            }
            kinds.truncate(96);
            let kind = |y: i32| if y < -32 { 0 } else { kinds[(63 - y) as usize] };
            let gradient = [(column % 9) - 4, (column % 11) - 5];
            memoized.update_xz(x, z, gradient[0], gradient[1]);
            plain.update_xz(x, z, gradient[0], gradient[1]);
            memo.clear();
            let (mut above, mut water, mut bottom) = (0, i32::MIN, i32::MAX);
            let mut spans = Vec::new();
            let mut expected_at = std::collections::HashMap::new();
            for y in (-32..=63).rev() {
                match kind(y) {
                    0 => {
                        memo.clear();
                        above = 0;
                        water = i32::MIN;
                    }
                    1 => {
                        memo.clear();
                        if water == i32::MIN {
                            water = y + 1;
                        }
                    }
                    _ => {
                        if bottom >= y {
                            bottom = -33;
                            for below in (-33..y).rev() {
                                if kind(below) != 2 {
                                    bottom = below + 1;
                                    break;
                                }
                            }
                        }
                        above += 1;
                        let below = y - bottom + 1;
                        let position = BlockPos::new(x, y, z);
                        let before = memo.entries.len();
                        let (got, low) = memoized
                            .try_apply_in_run(
                                rule_m, position, &mut memo, above, below, water, bottom,
                            )
                            .unwrap();
                        assert!(low <= y);
                        spans.push((y, low, got));
                        hits += usize::from(memo.entries.len() == before && before > 0);
                        plain.update_y(above, below, water, y);
                        let expected = plain.try_apply(rule_p, position).unwrap();
                        assert_eq!(got, expected, "{x} {y} {z}");
                        expected_at.insert(y, (bottom, expected));
                    }
                }
            }
            // A span holds for every block of its run down to its low end.
            for (y, low, got) in spans {
                let run = expected_at[&y].0;
                for below in low.max(run)..=y {
                    assert_eq!(expected_at[&below], (run, got), "{x} {y} {below} {z}");
                }
                long_spans += usize::from(y - low.max(run) >= 3);
            }
        }
        assert!(hits > 100, "{hits}");
        assert!(long_spans > 20, "{long_spans}");
    }
}
