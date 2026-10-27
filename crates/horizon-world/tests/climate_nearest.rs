//! `ClimateSearch::find_nearest` with fresh answers found ahead of time gives
//! the same answers, ties included, as `find` with its previous result.
use horizon_world::worldgen::climate::{ClimateIndex, Parameter, ParameterPoint, TargetPoint};

#[test]
fn nearest_answers_ahead_of_time_match_the_stateful_search() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move |n: i64| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as i64 % n
    };
    // Few coarse values: many points and targets tie.
    let mut entries = Vec::new();
    for id in 0..300 {
        let coordinates = std::array::from_fn(|_| {
            let min = next(5) * 1000;
            Parameter {
                min,
                max: min + next(2) * 1000,
            }
        });
        entries.push((
            ParameterPoint {
                coordinates,
                offset: next(2) * 1000,
            },
            id,
        ));
    }
    for branching in [2, 6, 19] {
        let index = ClimateIndex::with_branching(entries.clone(), branching).unwrap();
        let (mut stateful, mut ahead) = (index.searcher(), index.searcher());
        for _ in 0..20_000 {
            let target = TargetPoint {
                temperature: next(7) * 500,
                humidity: next(7) * 500,
                continentalness: next(7) * 500,
                erosion: next(7) * 500,
                depth: next(7) * 500,
                weirdness: next(7) * 500,
            };
            let nearest = index.nearest(target).unwrap();
            assert_eq!(
                stateful.find(target).unwrap(),
                ahead.find_nearest(target, nearest).unwrap(),
                "branching {branching}, {target:?}"
            );
        }
    }
}
