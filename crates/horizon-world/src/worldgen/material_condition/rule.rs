//! Compiled material rule composition, sharing the condition worker's context.
use super::{
    Compiled, Context, Facts, MaterialCondition, MaterialConditions, MaterialSurfaceQueries,
    RunTrace, density::MaterialDensitySources,
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
                random,
            } => rule.ops.push(Op::OreVein {
                blocks,
                raw_ore_chance,
                densities,
                random,
            }),
            Self::Bandlands => rule.ops.push(Op::Bandlands),
            Self::Condition {
                condition,
                rule: then,
            } => {
                let at = rule.open_condition(condition);
                then.flatten(rule);
                rule.close_condition(at);
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
    /// Append a condition op, whose subtree is what is appended until
    /// `close_condition` with the op's index.
    fn open_condition(&mut self, condition: Compiled<R>) -> usize {
        let at = self.ops.len();
        self.ops.push(Op::Condition {
            condition: self.conditions.len(),
            skip: 0,
        });
        self.conditions.push(condition);
        at
    }
    fn close_condition(&mut self, at: usize) {
        let end = self.ops.len();
        if let Op::Condition { skip, .. } = &mut self.ops[at] {
            *skip = end;
        }
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
                    at = if self.conditions[*condition].test(context)? {
                        at + 1
                    } else {
                        *skip
                    };
                }
                Op::OreVein {
                    blocks,
                    raw_ore_chance,
                    densities,
                    random,
                } => {
                    if let Some(block) = ore_vein(
                        blocks,
                        *raw_ore_chance,
                        densities,
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
                    let condition = &mut self.conditions[*condition];
                    let value = condition.test_once(context)?;
                    condition.stable_below(value, context, run);
                    at = if value { at + 1 } else { *skip };
                }
                Op::OreVein {
                    blocks,
                    raw_ore_chance,
                    densities,
                    random,
                } => {
                    match context.surface.density_heights(densities[0]) {
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
}
#[derive(Clone, Copy)]
struct RunEntry {
    /// The biome the program tested, `None` when it tested none.
    biome: Option<u32>,
    low: i64,
    result: Option<BlockStateId>,
}
impl RunMemo {
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
fn ore_vein<P: PositionalRandom, S: MaterialSurfaceQueries>(
    blocks: &[BlockStateId; 3],
    raw_ore_chance: f32,
    densities: &[usize; 3],
    random: &P,
    position: BlockPos,
    context: &mut Context<S>,
) -> Result<Option<BlockStateId>, MaterialRuleError> {
    // Most blocks lie outside a vein's heights: the density read
    // would give `<= 0`, and nothing happens before it. The
    // heights are the buffer's as sampled now (`retarget` moves it).
    if context
        .surface
        .density_heights(densities[0])
        .is_some_and(|heights| heights.never_positive(context.position))
    {
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
impl<R: RandomSource> CompiledRule<R>
where
    R::Positional: Clone,
{
    /// This rule where `facts` hold: conditions they decide are folded away,
    /// with what can no longer produce a block, and a sequence stops at its
    /// first child that always produces one. For every position the facts
    /// cover it gives the block the whole rule gives.
    fn specialize(&self, facts: Facts) -> Self {
        let mut rule = Self {
            ops: Vec::new(),
            conditions: Vec::new(),
        };
        self.specialize_into(0, self.ops.len(), facts, &mut rule);
        rule
    }
    /// Append the subtrees of ops `start..end` (a sequence) specialised to
    /// `facts` to `rule`. Returns whether what it appended always produces
    /// a block (only an error otherwise); nothing follows it then.
    fn specialize_into(&self, start: usize, end: usize, facts: Facts, rule: &mut Self) -> bool {
        let mut at = start;
        while at < end {
            match &self.ops[at] {
                Op::Condition { condition, skip } => {
                    let condition = &self.conditions[*condition];
                    match condition.decided(facts) {
                        Some(false) => {}
                        Some(true) => {
                            if self.specialize_into(at + 1, *skip, facts, rule) {
                                return true;
                            }
                        }
                        None => {
                            let open = rule.open_condition(condition.clone());
                            self.specialize_into(at + 1, *skip, facts, rule);
                            if rule.ops.len() == open + 1 {
                                // Nothing under it can produce a block.
                                rule.ops.pop();
                                rule.conditions.pop();
                            } else {
                                rule.close_condition(open);
                            }
                        }
                    }
                    at = *skip;
                }
                Op::Block(block) => {
                    rule.ops.push(Op::Block(*block));
                    return true;
                }
                Op::Bandlands => {
                    rule.ops.push(Op::Bandlands);
                    return true;
                }
                Op::OreVein {
                    blocks,
                    raw_ore_chance,
                    densities,
                    random,
                } => {
                    rule.ops.push(Op::OreVein {
                        blocks: *blocks,
                        raw_ore_chance: *raw_ore_chance,
                        densities: *densities,
                        random: random.clone(),
                    });
                    at += 1;
                }
            }
        }
        false
    }
    fn inputs(&self, gradients: &mut Vec<(i32, i32)>, surface: &mut bool) {
        for condition in &self.conditions {
            condition.inputs(gradients, surface);
        }
    }
}
/// A rule specialised (`CompiledRule::specialize`) for each band of y between
/// its vertical gradients' bounds, and below and at or above the preliminary
/// surface when the rule asks for it: most blocks of a column, deep stone,
/// then evaluate only the few rules that can apply there.
pub struct SurfaceVariants {
    /// The first y of each band after the first, ascending.
    starts: Vec<i32>,
    /// The rule index per band (two per band, below then above the surface,
    /// when `surface`).
    rules: Vec<usize>,
    pub surface: bool,
}
impl SurfaceVariants {
    /// The rule for a block at `y`, `above` the preliminary surface (ignored
    /// unless `surface`).
    #[inline]
    pub fn rule(&self, y: i32, above: bool) -> usize {
        let band = self.starts.partition_point(|&start| start <= y);
        if self.surface {
            self.rules[band * 2 + usize::from(above)]
        } else {
            self.rules[band]
        }
    }
    /// `rule(y, y >= minimum)`, with the lowest height down to which the
    /// blocks below get the same rule (`minimum`, the preliminary surface
    /// level, is only read when `surface`).
    #[inline]
    pub fn rule_down(
        &self,
        y: i32,
        minimum: impl FnOnce() -> Result<i32, MaterialRuleError>,
    ) -> Result<(usize, i32), MaterialRuleError> {
        let band = self.starts.partition_point(|&start| start <= y);
        let low = if band == 0 {
            i32::MIN
        } else {
            self.starts[band - 1]
        };
        if !self.surface {
            return Ok((self.rules[band], low));
        }
        let minimum = minimum()?;
        Ok(if y >= minimum {
            (self.rules[band * 2 + 1], low.max(minimum))
        } else {
            (self.rules[band * 2], low)
        })
    }
}
impl<R: RandomSource, S: MaterialSurfaceQueries> MaterialConditions<R, S>
where
    R::Positional: Clone,
{
    /// Add the specialised variants of rule `index` (`SurfaceVariants`).
    pub fn surface_variants(&mut self, index: usize) -> Result<SurfaceVariants, MaterialRuleError> {
        let rule = self
            .rules
            .get(index)
            .ok_or(MaterialRuleError::UnknownRule)?;
        let (mut gradients, mut surface) = (Vec::new(), false);
        rule.inputs(&mut gradients, &mut surface);
        // A gradient is true at and below `low`, random between, false at
        // and above `high`: it is decided within each band these bound.
        let mut starts: Vec<i32> = gradients
            .iter()
            .flat_map(|&(low, high)| [low.saturating_add(1), high])
            .collect();
        starts.sort_unstable();
        starts.dedup();
        let mut variants = Vec::new();
        for band in 0..=starts.len() {
            let low = if band == 0 {
                i32::MIN
            } else {
                starts[band - 1]
            };
            let high = starts.get(band).map_or(i32::MAX, |&next| next - 1);
            let sides: &[Option<bool>] = if surface {
                &[Some(false), Some(true)]
            } else {
                &[None]
            };
            for &above_surface in sides {
                let facts = Facts {
                    low,
                    high,
                    above_surface,
                };
                variants.push(self.rules[index].specialize(facts));
            }
        }
        let first = self.rules.len();
        self.rules.extend(variants);
        Ok(SurfaceVariants {
            starts,
            rules: (first..self.rules.len()).collect(),
            surface,
        })
    }
}
impl<R: RandomSource, T: MaterialSurfaceQueries> MaterialConditions<R, MaterialDensitySources<T>> {
    /// Re-target the per-cell density bindings for a reused worker.
    pub fn retarget_surface_densities(&mut self, position: [i32; 3]) -> Result<(), DecodeError> {
        self.context.surface.retarget(position)
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
    /// run (`position.y` when not known). `floor` bounds how far down: the
    /// run's lowest block, or above it the lowest height rule `index` is
    /// applied to in this run (`SurfaceVariants::rule_down`).
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
        let mut biome = None;
        if !memo.entries.is_empty() {
            for entry in &memo.entries {
                if entry.biome.is_none() && y >= entry.low {
                    return Ok((entry.result, entry.low.max(i64::from(floor)) as i32));
                }
            }
            // A failed lookup is left for the program to report, if it asks.
            if let Ok(Some((id, low))) = self.context.surface.biome_id_down(position, floor) {
                biome = Some(id);
                for entry in &memo.entries {
                    if entry.biome == Some(id) && y >= entry.low {
                        return Ok((entry.result, entry.low.max(i64::from(low)) as i32));
                    }
                }
            }
        }
        self.update_y(stone_above, stone_below, water_height, position.y);
        self.context.biome_id = biome;
        let rule = self
            .rules
            .get_mut(index)
            .ok_or(MaterialRuleError::UnknownRule)?;
        let mut run = RunTrace {
            low: i64::from(floor),
            biome: false,
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
            if memo.entries.len() < 8 {
                memo.entries.push(RunEntry {
                    biome,
                    low: run.low,
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

    struct Surface;
    impl MaterialSurfaceQueries for Surface {
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

    /// Runs memoized down a column, through the rule's surface variants,
    /// give what the whole program gives block by block.
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
        let variants = memoized.surface_variants(rule_m).unwrap();
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
                        let (rule, rule_low) = variants
                            .rule_down(y, || memoized.minimum_surface_level())
                            .unwrap();
                        let floor = bottom.max(rule_low);
                        let (got, low) = memoized
                            .try_apply_in_run(rule, position, &mut memo, above, below, water, floor)
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
