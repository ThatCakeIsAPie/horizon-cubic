use super::*;

/// Interpolation over a coarse density grid. Scalar queries and volume queries
/// retain vanilla's distinct floating-point paths and child bulk calls.
#[derive(Clone, Debug)]
pub struct InterpolatedSampler {
    input: Arc<dyn DensitySampler>,
    cell_size: [i32; 3],
    inverse: [f32; 3],
    /// The input is pointwise ([`is_pointwise`]), so its lattice samples can
    /// come from another volume over the same cells.
    pointwise_input: bool,
    /// Equal for samplers compiled from the same function by one compiler.
    identity: Option<(u64, usize)>,
}
impl InterpolatedSampler {
    pub fn new(
        input: Arc<dyn DensitySampler>,
        cell_size_xz: i32,
        cell_size_y: i32,
    ) -> Result<Self, SampleError> {
        if cell_size_xz <= 0 || cell_size_y <= 0 {
            return Err(DensityError::NonpositiveSteps.into());
        }
        let cell_size = [cell_size_xz, cell_size_y, cell_size_xz];
        Ok(Self {
            pointwise_input: is_pointwise(&input),
            identity: None,
            input,
            cell_size,
            inverse: cell_size.map(|size| 1.0 / size as f32),
        })
    }

    fn sample_block_volume(
        &self,
        context: &mut SamplerContext,
        output: &mut [f32],
        volume: DensityVolume,
    ) -> Result<(), SampleError> {
        let minimum = volume.minimum();
        let maximum = volume.maximum();
        let first: [i32; 3] = std::array::from_fn(|i| minimum[i].div_euclid(self.cell_size[i]));
        let counts: [i32; 3] = std::array::from_fn(|i| {
            maximum[i]
                .div_euclid(self.cell_size[i])
                .wrapping_sub(first[i])
                .wrapping_add(1)
        });
        let sizes = std::array::from_fn(|i| {
            counts[i].wrapping_add(i32::from(maximum[i].rem_euclid(self.cell_size[i]) != 0))
        });
        let coarse = DensityVolume::new(
            sizes,
            std::array::from_fn(|i| first[i].wrapping_mul(self.cell_size[i])),
            self.cell_size,
        )?;
        context.with_buffer(coarse.size(), |context, samples| {
            if !self.samples_from_lattice(context, samples, coarse)? {
                self.input.sample_volume(context, samples, coarse)?;
            }
            let [size_x, size_y, size_z] = coarse.sizes();
            let [count_x, count_y, count_z] = counts.map(|count| count.max(0) as usize);
            // Corner samples beyond the coarse grid clamp to its last layer.
            let next = |index: usize, size: usize| (index + 1).min(size - 1);
            for z in 0..count_z {
                let zs = [z, next(z, size_z)];
                for x in 0..count_x {
                    let xs = [x, next(x, size_x)];
                    // Corner bit 0 selects X, bit 2 Z (bit 1, Y, is added per cell).
                    let columns: [usize; 4] =
                        std::array::from_fn(|i| (xs[i & 1] + zs[i >> 1] * size_x) * size_y);
                    for y in 0..count_y {
                        let ys = [y, next(y, size_y)];
                        let corners = std::array::from_fn(|corner| {
                            samples[columns[(corner & 1) | ((corner >> 1) & 2)]
                                + ys[(corner >> 1) & 1]]
                        });
                        self.fill_cell(output, volume, coarse, [x, y, z], corners)?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Samplers with the same identity compute the same function (the graph
    /// compiler gives its function id), so they share lattice samples.
    pub fn with_identity(mut self, compiler: u64, function: usize) -> Self {
        self.identity = Some((compiler, function));
        self
    }

    /// Fill `samples` (the input over `coarse`, a volume on the cell lattice)
    /// from the context's known lattice samples of this sampler, sampling
    /// only the points none of them holds. The input is pointwise, so each
    /// value is the one sampling `coarse` whole would give. False when there
    /// is nothing to reuse.
    fn samples_from_lattice(
        &self,
        context: &mut SamplerContext,
        samples: &mut [f32],
        coarse: DensityVolume,
    ) -> Result<bool, SampleError> {
        let Some(key) = self.identity else {
            return Ok(false);
        };
        if !self.pointwise_input || !context.lattice.iter().any(|known| known.sampler == key) {
            return Ok(false);
        }
        let lattice = std::mem::take(&mut context.lattice);
        let result = self.fill_from_known(context, samples, coarse, &lattice);
        context.lattice = lattice;
        result.map(|()| true)
    }

    fn fill_from_known(
        &self,
        context: &mut SamplerContext,
        samples: &mut [f32],
        coarse: DensityVolume,
        lattice: &[LatticeSamples],
    ) -> Result<(), SampleError> {
        let key = self.identity;
        let [sx, sy, sz] = coarse.sizes();
        // Missing Y runs of each column, as (column, first Y, length).
        let mut missing: Vec<(usize, usize, usize)> = Vec::new();
        let mut known = vec![false; sy];
        for z in 0..sz {
            for x in 0..sx {
                let column = x + z * sx;
                let target = &mut samples[column * sy..(column + 1) * sy];
                known.fill(false);
                let position = [
                    coarse.coordinate(0, x),
                    coarse.minimum()[1],
                    coarse.coordinate(2, z),
                ];
                for piece in lattice.iter().filter(|known| Some(known.sampler) == key) {
                    let volume = piece.volume;
                    let relative: [i64; 3] = std::array::from_fn(|i| {
                        i64::from(position[i]) - i64::from(volume.minimum()[i])
                    });
                    let step = |axis: usize| i64::from(self.cell_size[axis]);
                    if relative[0] < 0
                        || relative[2] < 0
                        || relative[0] % step(0) != 0
                        || relative[1] % step(1) != 0
                        || relative[2] % step(2) != 0
                    {
                        continue;
                    }
                    let (px, pz) = (
                        (relative[0] / step(0)) as usize,
                        (relative[2] / step(2)) as usize,
                    );
                    if px >= volume.sizes()[0] || pz >= volume.sizes()[2] {
                        continue;
                    }
                    let source_column = (px + pz * volume.sizes()[0]) * volume.sizes()[1];
                    // Coarse Y index y is piece Y index y + offset.
                    let offset = relative[1] / step(1);
                    for (y, value) in target.iter_mut().enumerate() {
                        let source = y as i64 + offset;
                        if !known[y] && source >= 0 && (source as usize) < volume.sizes()[1] {
                            *value = piece.values[source_column + source as usize];
                            known[y] = true;
                        }
                    }
                }
                let mut y = 0;
                while y < sy {
                    if known[y] {
                        y += 1;
                        continue;
                    }
                    let start = y;
                    while y < sy && !known[y] {
                        y += 1;
                    }
                    missing.push((column, start, y - start));
                }
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let columns = sx * sz;
        let runs = missing.iter().filter(|run| run.0 == 0).count();
        let uniform = missing.len() == runs * columns
            && missing.chunks(runs).enumerate().all(|(column, chunk)| {
                chunk
                    .iter()
                    .zip(&missing[..runs])
                    .all(|(a, b)| a.0 == column && (a.1, a.2) == (b.1, b.2))
            });
        if uniform {
            // The same Y runs in every column (the top layer): one volume each.
            for &(_, start, length) in &missing[..runs] {
                let volume = DensityVolume::new(
                    [sx as i32, length as i32, sz as i32],
                    [
                        coarse.minimum()[0],
                        coarse.coordinate(1, start),
                        coarse.minimum()[2],
                    ],
                    coarse.steps(),
                )?;
                context.with_buffer(volume.size(), |context, values| {
                    self.input.sample_volume(context, values, volume)?;
                    for column in 0..columns {
                        let base = column * sy + start;
                        samples[base..base + length]
                            .copy_from_slice(&values[column * length..(column + 1) * length]);
                    }
                    Ok(())
                })?;
            }
        } else {
            for &(column, start, length) in &missing {
                let (x, z) = (column % sx, column / sx);
                let volume = DensityVolume::new(
                    [1, length as i32, 1],
                    [
                        coarse.coordinate(0, x),
                        coarse.coordinate(1, start),
                        coarse.coordinate(2, z),
                    ],
                    coarse.steps(),
                )?;
                context.with_buffer(volume.size(), |context, values| {
                    self.input.sample_volume(context, values, volume)?;
                    let base = column * sy + start;
                    samples[base..base + length].copy_from_slice(values);
                    Ok(())
                })?;
            }
        }
        Ok(())
    }

    fn fill_cell(
        &self,
        output: &mut [f32],
        volume: DensityVolume,
        coarse: DensityVolume,
        cell: [usize; 3],
        corners: [f32; 8],
    ) -> Result<(), SampleError> {
        let offset: [i32; 3] = std::array::from_fn(|i| {
            coarse.minimum()[i]
                .wrapping_add((cell[i] as i32).wrapping_mul(self.cell_size[i]))
                .wrapping_sub(volume.minimum()[i])
        });
        let start = offset.map(|offset| offset.wrapping_neg().max(0));
        let end: [i32; 3] = std::array::from_fn(|i| {
            self.cell_size[i]
                .min((volume.sizes()[i] as i32).wrapping_sub(offset[i]))
                .wrapping_sub(1)
        });
        if end[1] < start[1] {
            return Ok(());
        }
        let length = (end[1] - start[1] + 1) as usize;
        let output_len = output.len();
        for z in start[2]..=end[2] {
            let alpha_z = z as f32 * self.inverse[2];
            let z00 = lerp(alpha_z, corners[0], corners[4]);
            let z01 = lerp(alpha_z, corners[2], corners[6]);
            let z10 = lerp(alpha_z, corners[1], corners[5]);
            let z11 = lerp(alpha_z, corners[3], corners[7]);
            let column = |x: i32| {
                let alpha_x = x as f32 * self.inverse[0];
                let low = lerp(alpha_x, z00, z10);
                let high = lerp(alpha_x, z01, z11);
                let increment = (high - low) * self.inverse[1];
                let value = low + increment * start[1] as f32;
                let index = volume
                    .index([
                        offset[0].wrapping_add(x) as usize,
                        offset[1].wrapping_add(start[1]) as usize,
                        offset[2].wrapping_add(z) as usize,
                    ])
                    .filter(|index| {
                        index
                            .checked_add(length)
                            .is_some_and(|end| end <= output_len)
                    });
                (index, value, increment)
            };
            // A row four columns wide (a whole cell) runs its columns' independent
            // repeated-addition chains side by side, so their additions overlap.
            if end[0] - start[0] == 3 {
                let columns: [_; 4] = std::array::from_fn(|i| column(start[0] + i as i32));
                if let Some(rows) = four_columns(output, columns.map(|(index, ..)| index), length) {
                    let mut values = columns.map(|(_, value, _)| value);
                    let increments = columns.map(|(.., increment)| increment);
                    let [a, b, c, d] = rows;
                    for y in 0..length {
                        [a[y], b[y], c[y], d[y]] = values;
                        for (value, increment) in values.iter_mut().zip(increments) {
                            *value += increment;
                        }
                    }
                    continue;
                }
            }
            for x in start[0]..=end[0] {
                let (index, mut value, increment) = column(x);
                let index = index.ok_or(DensityError::VolumeTooLarge)?;
                for output in &mut output[index..index + length] {
                    *output = value;
                    // Repeated addition is observable: multiplication by the
                    // current Y offset does not have the same rounding.
                    value += increment;
                }
            }
        }
        Ok(())
    }
}

/// Four disjoint `length`-long column slices of `output` at ascending `starts`, if
/// every start was valid and the columns do not overlap.
fn four_columns(
    output: &mut [f32],
    starts: [Option<usize>; 4],
    length: usize,
) -> Option<[&mut [f32]; 4]> {
    let [a, b, c, d] = starts;
    let [a, b, c, d] = [a?, b?, c?, d?];
    if b.checked_sub(a)? < length || c.checked_sub(b)? < length || d.checked_sub(c)? < length {
        return None;
    }
    let (first, rest) = output.get_mut(a..)?.split_at_mut(b - a);
    let (second, rest) = rest.split_at_mut(c - b);
    let (third, fourth) = rest.split_at_mut(d - c);
    Some([
        &mut first[..length],
        &mut second[..length],
        &mut third[..length],
        fourth.get_mut(..length)?,
    ])
}

fn lerp(alpha: f32, first: f32, second: f32) -> f32 {
    first + alpha * (second - first)
}
impl DensitySampler for InterpolatedSampler {
    fn bulk_differs(&self) -> bool {
        true
    }
    fn scalar_inputs(&self) -> Option<Vec<&Arc<dyn DensitySampler>>> {
        Some(vec![&self.input])
    }
    fn sample_value(
        &self,
        context: &mut SamplerContext,
        position: [i32; 3],
    ) -> Result<f32, SampleError> {
        let remainder: [i32; 3] =
            std::array::from_fn(|i| position[i].rem_euclid(self.cell_size[i]));
        if remainder == [0; 3] {
            return self.input.sample_value(context, position);
        }
        let volume = DensityVolume::new(
            [2; 3],
            std::array::from_fn(|i| position[i].wrapping_sub(remainder[i])),
            self.cell_size,
        )?;
        context.with_buffer(8, |context, corners| {
            self.input.sample_volume(context, corners, volume)?;
            let alpha: [f32; 3] =
                std::array::from_fn(|i| remainder[i] as f32 / self.cell_size[i] as f32);
            // Samples use Y-innermost storage; interpolation blends X, Y, Z.
            let xy0 = lerp(
                alpha[1],
                lerp(alpha[0], corners[0], corners[2]),
                lerp(alpha[0], corners[1], corners[3]),
            );
            let xy1 = lerp(
                alpha[1],
                lerp(alpha[0], corners[4], corners[6]),
                lerp(alpha[0], corners[5], corners[7]),
            );
            Ok(lerp(alpha[2], xy0, xy1))
        })
    }
    fn sample_volume(
        &self,
        context: &mut SamplerContext,
        output: &mut [f32],
        volume: DensityVolume,
    ) -> Result<(), SampleError> {
        check_buffer(output, volume)?;
        if context.sparse {
            if (0..3).all(|i| {
                volume.steps()[i] % self.cell_size[i] == 0
                    && volume.minimum()[i].rem_euclid(self.cell_size[i]) == 0
            }) {
                return self.input.sample_volume(context, output, volume);
            }
            return visit(volume, |index, position| {
                output[index] = self.sample_value(context, position)?;
                Ok(())
            });
        }
        if (0..3).all(|i| {
            (volume.steps()[i] == self.cell_size[i] || volume.sizes()[i] == 1)
                && volume.minimum()[i].rem_euclid(self.cell_size[i]) == 0
        }) {
            self.input.sample_volume(context, output, volume)?;
            if let (true, Some(key)) = (
                self.pointwise_input && volume.steps() == self.cell_size,
                self.identity,
            ) {
                if let Some(capture) = &mut context.capture {
                    if !capture
                        .iter()
                        .any(|known| known.sampler == key && known.volume == volume)
                    {
                        capture.push(LatticeSamples {
                            sampler: key,
                            volume,
                            values: output[..volume.size()].into(),
                        });
                    }
                }
            }
            return Ok(());
        }
        if volume.steps() == [1; 3] {
            return self.sample_block_volume(context, output, volume);
        }
        // The reference evaluates the complete block-resolution bounding volume
        // before selecting strided samples, including trailing unsampled blocks.
        let dense = DensityVolume::new(
            std::array::from_fn(|i| (volume.sizes()[i] as i32).wrapping_mul(volume.steps()[i])),
            volume.minimum(),
            [1; 3],
        )?;
        context.with_buffer(dense.size(), |context, samples| {
            self.sample_block_volume(context, samples, dense)?;
            let [sx, sy, sz] = volume.sizes();
            for z in 0..sz {
                for x in 0..sx {
                    for y in 0..sy {
                        let source = dense.index(std::array::from_fn(|i| {
                            [x, y, z][i] * volume.steps()[i] as usize
                        }));
                        output[(x + z * sx) * sy + y] = samples[source.unwrap()];
                    }
                }
            }
            Ok(())
        })
    }
}
