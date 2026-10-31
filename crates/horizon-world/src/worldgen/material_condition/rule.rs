//! Compiled material rule composition, sharing the condition worker's context.
use super::{
    Compiled, Context, Facts, MaterialCondition, MaterialConditions, MaterialSurfaceQueries,
    density::MaterialDensitySources,
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
