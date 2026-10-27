//! Global fluid selection and the 26.3 aquifer terrain paths.
mod config;
mod noise_based;
use crate::registry::BlockStateId;
pub use config::{AquiferConfig, AquiferField, AquiferSamplers};
pub use noise_based::{
    AquiferBlockStates, AquiferError, NoiseBasedAquifer, RegistryAquiferBlockStates,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidStatus {
    pub level: i32,
    pub state: BlockStateId,
}
impl FluidStatus {
    pub const fn at(self, y: i32, air: BlockStateId) -> BlockStateId {
        if y < self.level { self.state } else { air }
    }
}
pub trait FluidPicker: std::fmt::Debug + Send + Sync {
    fn compute(&self, position: [i32; 3]) -> FluidStatus;
    /// A height from which up `compute(p).at(p.y, air)` is air at every
    /// position, if the picker knows one. None is the conservative answer.
    fn air_from(&self) -> Option<i32> {
        None
    }
}
impl<P: FluidPicker + ?Sized> FluidPicker for &P {
    fn compute(&self, position: [i32; 3]) -> FluidStatus {
        P::compute(self, position)
    }
    fn air_from(&self) -> Option<i32> {
        P::air_from(self)
    }
}
/// State IDs must come from one immutable block registry. Default fluid can be
/// any registered block state; vanilla does not validate its material here.
#[derive(Clone, Copy, Debug)]
pub struct GlobalFluidPicker {
    sea_level: i32,
    default_fluid: BlockStateId,
    lava: BlockStateId,
    air: BlockStateId,
    disabled: bool,
}
impl GlobalFluidPicker {
    pub const fn new(
        sea_level: i32,
        default_fluid: BlockStateId,
        lava: BlockStateId,
        air: BlockStateId,
    ) -> Self {
        Self {
            sea_level,
            default_fluid,
            lava,
            air,
            disabled: false,
        }
    }
    /// Matches the vanilla developer flag. Regular gameplay keeps this false.
    pub const fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}
impl FluidPicker for GlobalFluidPicker {
    fn compute(&self, [_, y, _]: [i32; 3]) -> FluidStatus {
        if self.disabled {
            FluidStatus {
                level: -4064,
                state: self.air,
            }
        } else if y < (-54).min(self.sea_level) {
            FluidStatus {
                level: -54,
                state: self.lava,
            }
        } else {
            FluidStatus {
                level: self.sea_level,
                state: self.default_fluid,
            }
        }
    }
    /// Lava stops at -54 below the lower of -54 and the sea level, the
    /// default fluid at the sea level.
    fn air_from(&self) -> Option<i32> {
        Some(if self.disabled {
            i32::MIN
        } else {
            self.sea_level
        })
    }
}
#[derive(Debug)]
pub struct DisabledAquifer<P> {
    picker: P,
    air: BlockStateId,
}
impl<P: FluidPicker> DisabledAquifer<P> {
    pub const fn new(picker: P, air: BlockStateId) -> Self {
        Self { picker, air }
    }
    /// None selects the generator's default solid block. Java DCMPL treats NaN
    /// as -1, so NaN and density <= zero call the picker; density > zero does not.
    pub fn compute_substance(&self, position: [i32; 3], density: f64) -> Option<BlockStateId> {
        if density > 0. {
            None
        } else {
            Some(self.picker.compute(position).at(position[1], self.air))
        }
    }
    pub const fn should_schedule_fluid_update(&self) -> bool {
        false
    }
    /// Above this height a density not above zero gives air.
    pub fn open_above(&self) -> Option<(i32, BlockStateId)> {
        let from = self.picker.air_from()?;
        Some((from.saturating_sub(1), self.air))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_picker_is_air_from_its_air_height() {
        let (fluid, lava, air) = (BlockStateId(1), BlockStateId(2), BlockStateId(0));
        for sea in [-100, -54, -10, 63, 300] {
            for disabled in [false, true] {
                let picker = GlobalFluidPicker::new(sea, fluid, lava, air).with_disabled(disabled);
                let from = picker.air_from().unwrap();
                for y in (-200..400).filter(|&y| y >= from) {
                    assert_eq!(picker.compute([0, y, 0]).at(y, air), air, "{sea} {y}");
                }
                if !disabled {
                    let below = from - 1;
                    assert_ne!(picker.compute([0, below, 0]).at(below, air), air);
                }
            }
        }
    }
}
