//! Compiled material rule composition, sharing the condition worker's context.
use super::{
    Compiled, Context, DensityHeights, MaterialCondition, MaterialConditions,
    MaterialSurfaceQueries,
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
                        context.surface.bind_gated_density(density, gate, true)?,
                        context.surface.bind_gated_density(richness, gate, false)?,
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
