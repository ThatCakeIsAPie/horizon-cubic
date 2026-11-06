//! Vertical shards: the grouping maths (docs/cubic-v2-plan.md, "Residency").
//!
//! A cubic dimension's server holds a list of shards. Each is a `World` with
//! a window of cubes of its own, backed by the same [`crate::cubic::CubicLevel`].
//! Players whose wanted ranges (`cubic::wanted_cubes`) overlap share a shard,
//! whose window is the union of their ranges; no cube is in two shards.
//!
//! This module decides who is in which shard. It knows nothing of worlds:
//! [`plan`] takes the players' ranges, the shard each is in now and the
//! windows the shards hold, and says which players share a shard, which
//! shards merge into which, which groups need a new shard, and the window each
//! shard should move to. `game/cubic.rs` carries the plan out.
//!
//! The rules, in cubes between two ranges ([`gap`], negative when they
//! overlap):
//!
//! - **Merge**: ranges that touch or overlap (gap 0 or less) are one shard.
//! - **Split**: players of one shard whose ranges are two or more cubes apart
//!   (gap 2 or more, [`SPLIT_GAP`]) are not. A gap of one cube is the
//!   hysteresis: players who were together stay together, players who were
//!   apart stay apart, so a player at the edge does not flip each time it
//!   moves a cube.
//! - **Cap**: a shard's window is at most `MAX_WINDOW_CUBES` tall. A chain of
//!   overlapping ranges that is taller is cut between players, by cube.
//! - **Disjoint**: the windows the plan names never overlap
//!   ([`separate_windows`]).
use crate::cubic::{MAX_WINDOW_CUBES, union_cubes};

/// A range of cubes (first, last), both included.
pub type Range = (i32, i32);

/// The shard that always exists once the dimension is open: the saved window
/// (`cubic::save_window`) is the one of the shard with this id, and the
/// level's per-level files (forced chunks, attachments, raids) are its.
pub const PRIMARY: u16 = 0;

/// The gap from which players of one shard split.
pub const SPLIT_GAP: i32 = 2;

/// The empty cubes between ranges `a` and `b`: 0 when they touch, negative
/// (minus the cubes both hold) when they overlap.
pub fn gap(a: Range, b: Range) -> i32 {
    a.0.max(b.0) - a.1.min(b.1) - 1
}

/// Whether two windows hold a cube in common.
pub fn overlaps(a: Range, b: Range) -> bool {
    gap(a, b) < 0
}

/// A player to place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    /// Whatever names the player to the caller (the connection).
    pub key: u32,
    /// The player's cube.
    pub cube: i32,
    /// The cubes it wants loaded (`cubic::wanted_cubes`).
    pub range: Range,
    /// The shard it is in now.
    pub shard: u16,
}

/// A shard that exists, with the window its world holds now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Existing {
    pub id: u16,
    pub window: Range,
}

/// Players that share a shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// The shard they share: one that exists, or `None` for a new one.
    pub shard: Option<u16>,
    /// Shards that merge into `shard` (their players join it and they close).
    pub absorbed: Vec<u16>,
    /// The players' keys, lowest cube first.
    pub members: Vec<u32>,
    /// The window the shard should hold: the union of its players' ranges
    /// (at most `MAX_WINDOW_CUBES`), cut so it overlaps no other group's.
    pub window: Range,
}

/// What [`plan`] decided.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub groups: Vec<Group>,
}

struct Union(Vec<usize>);

impl Union {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.0[a] != a {
            self.0[a] = self.0[self.0[a]];
            a = self.0[a];
        }
        a
    }
    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b)] = a.min(b);
        }
    }
}

/// Splits `members` (sorted by cube) into runs whose ranges' union is at
/// most `MAX_WINDOW_CUBES` tall, never between players in the same cube.
fn cut_to_cap(members: &[Member]) -> Vec<Vec<Member>> {
    let mut runs: Vec<Vec<Member>> = Vec::new();
    for m in members {
        if let Some(run) = runs.last_mut() {
            let lo = run.iter().map(|r| r.range.0).min().unwrap_or(m.range.0);
            let hi = run.iter().map(|r| r.range.1).max().unwrap_or(m.range.1);
            let same_cube = run.last().is_some_and(|r| r.cube == m.cube);
            if same_cube || hi.max(m.range.1) - lo.min(m.range.0) + 1 <= MAX_WINDOW_CUBES {
                run.push(*m);
                continue;
            }
        }
        runs.push(vec![*m]);
    }
    runs
}

/// Makes `windows` (one per group, in any order) overlap no other, keeping
/// each around its players' cubes (`cubes`: first and last cube of each
/// group's players, same order). Where two overlap, the boundary is put
/// between the two groups' nearest players. Windows that cannot be separated
/// (players in the same cube) are left as they are.
pub fn separate_windows(windows: &mut [Range], cubes: &[Range]) {
    let mut order: Vec<usize> = (0..windows.len()).collect();
    order.sort_by_key(|&i| (cubes[i].0, windows[i].0));
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if !overlaps(windows[a], windows[b]) {
            continue;
        }
        let (top_a, bottom_b) = (cubes[a].1, cubes[b].0);
        if top_a >= bottom_b {
            continue;
        }
        let boundary = (top_a + bottom_b).div_euclid(2);
        windows[a].1 = windows[a].1.min(boundary).max(top_a);
        windows[b].0 = windows[b].0.max(boundary + 1).min(bottom_b);
    }
}

/// Which players share a shard, and what becomes of the shards. `members`
/// are the players placed in the dimension, `shards` the shards that exist.
///
/// A shard without players (`idle`) takes part by its window: players whose
/// ranges meet it join it, so no cube is loaded twice; one that nobody meets
/// is left out of the plan (the caller closes it, unless it is the
/// [`PRIMARY`], which stays). A group no existing shard fits reuses the idle
/// primary if there is one, else it gets a new shard (`shard: None`).
pub fn plan(members: &[Member], shards: &[Existing]) -> Plan {
    let n = members.len();
    let occupied = |id: u16| members.iter().any(|m| m.shard == id);
    let idle: Vec<Existing> = shards.iter().copied().filter(|s| !occupied(s.id)).collect();
    let mut sets = Union::new(n + idle.len());
    for a in 0..n {
        for b in a + 1..n {
            let g = gap(members[a].range, members[b].range);
            let together = members[a].shard == members[b].shard;
            if g < 1 || (g < SPLIT_GAP && together) {
                sets.join(a, b);
            }
        }
        for (k, s) in idle.iter().enumerate() {
            if gap(members[a].range, s.window) < 1 {
                sets.join(a, n + k);
            }
        }
    }
    // Components, in the order of their lowest member.
    let mut roots: Vec<usize> = Vec::new();
    for a in 0..n {
        let r = sets.find(a);
        if !roots.contains(&r) {
            roots.push(r);
        }
    }
    struct Part {
        members: Vec<Member>,
        idle: Vec<u16>,
    }
    let mut parts: Vec<Part> = Vec::new();
    for &root in &roots {
        let mut all: Vec<Member> = (0..n)
            .filter(|&a| sets.find(a) == root)
            .map(|a| members[a])
            .collect();
        all.sort_by_key(|m| (m.cube, m.key));
        let idles: Vec<u16> = idle
            .iter()
            .enumerate()
            .filter(|(k, _)| sets.find(n + k) == root)
            .map(|(_, s)| s.id)
            .collect();
        let mut runs = cut_to_cap(&all).into_iter();
        if let Some(first) = runs.next() {
            parts.push(Part {
                members: first,
                idle: idles,
            });
        }
        for run in runs {
            parts.push(Part {
                members: run,
                idle: Vec::new(),
            });
        }
    }
    // The shard each part keeps: the one most of its players are in, then
    // the lowest id; a shard serves one part only.
    let mut taken: Vec<u16> = Vec::new();
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(parts[i].members.len()));
    let mut chosen: Vec<Option<u16>> = vec![None; parts.len()];
    for &i in &order {
        let mut candidates: Vec<(usize, u16)> = Vec::new();
        for m in &parts[i].members {
            if shards.iter().any(|s| s.id == m.shard) && !taken.contains(&m.shard) {
                match candidates.iter_mut().find(|c| c.1 == m.shard) {
                    Some(c) => c.0 += 1,
                    None => candidates.push((1, m.shard)),
                }
            }
        }
        for &id in &parts[i].idle {
            if !taken.contains(&id) && !candidates.iter().any(|c| c.1 == id) {
                candidates.push((0, id));
            }
        }
        // The primary never closes, so it is the survivor whenever it is a candidate.
        candidates.sort_by_key(|&(count, id)| (id != PRIMARY, std::cmp::Reverse(count), id));
        if let Some(&(_, id)) = candidates.first() {
            chosen[i] = Some(id);
            taken.push(id);
        }
    }
    // A part no shard fits: the idle primary nobody met, if there is one.
    for &i in &order {
        if chosen[i].is_none()
            && let Some(s) = idle
                .iter()
                .find(|s| s.id == PRIMARY && !taken.contains(&s.id))
        {
            chosen[i] = Some(s.id);
            taken.push(s.id);
        }
    }
    let mut groups: Vec<Group> = parts
        .iter()
        .zip(&chosen)
        .map(|(part, &shard)| {
            // Shards that merge into the survivor: the idle ones the part met
            // and the shards of its players that serve no other part.
            let mut absorbed: Vec<u16> = part
                .idle
                .iter()
                .copied()
                .chain(part.members.iter().map(|m| m.shard))
                .filter(|id| Some(*id) != shard && shards.iter().any(|s| s.id == *id))
                .collect();
            absorbed.sort_unstable();
            absorbed.dedup();
            absorbed.retain(|id| !chosen.contains(&Some(*id)));
            Group {
                shard,
                absorbed,
                members: part.members.iter().map(|m| m.key).collect(),
                window: union_cubes(part.members.iter().map(|m| m.range)).unwrap_or((0, 0)),
            }
        })
        .collect();
    let cubes: Vec<Range> = parts
        .iter()
        .map(|p| {
            (
                p.members.first().map_or(0, |m| m.cube),
                p.members.last().map_or(0, |m| m.cube),
            )
        })
        .collect();
    let mut windows: Vec<Range> = groups.iter().map(|g| g.window).collect();
    separate_windows(&mut windows, &cubes);
    for (group, window) in groups.iter_mut().zip(windows) {
        group.window = window;
    }
    groups.sort_by_key(|g| g.window);
    Plan { groups }
}

fn within(inner: Range, outer: Range) -> bool {
    inner.0 >= outer.0 && inner.1 <= outer.1
}

fn hull(a: Range, b: Range) -> Range {
    (a.0.min(b.0), a.1.max(b.1))
}

/// What a group's players do when its move cannot be made now: each stays
/// in the shard it is in, and the shard's window is the union of their ranges.
fn stay(group: &Group, members: &[Member]) -> Vec<Group> {
    let mut by_shard: Vec<(u16, Vec<Member>)> = Vec::new();
    for key in &group.members {
        let Some(m) = members.iter().find(|m| m.key == *key) else {
            continue;
        };
        match by_shard.iter_mut().find(|(s, _)| *s == m.shard) {
            Some((_, list)) => list.push(*m),
            None => by_shard.push((m.shard, vec![*m])),
        }
    }
    by_shard
        .into_iter()
        .map(|(shard, list)| Group {
            shard: Some(shard),
            absorbed: Vec::new(),
            members: list.iter().map(|m| m.key).collect(),
            window: union_cubes(list.iter().map(|m| m.range)).unwrap_or((0, 0)),
        })
        .collect()
}

/// Groups that name the same shard become one.
fn combine(groups: Vec<Group>, members: &[Member]) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    for g in groups {
        match g
            .shard
            .and_then(|s| out.iter_mut().find(|o| o.shard == Some(s)))
        {
            Some(o) => {
                o.members.extend(g.members);
                o.absorbed.extend(g.absorbed);
                o.absorbed.sort_unstable();
                o.absorbed.dedup();
                // Players too far apart for one window leave the shard's
                // window as it was: the ones that stay wait without terrain.
                let ranges: Vec<Range> = o
                    .members
                    .iter()
                    .filter_map(|k| members.iter().find(|m| m.key == *k))
                    .map(|m| m.range)
                    .collect();
                let (lo, hi) = ranges
                    .iter()
                    .fold((i32::MAX, i32::MIN), |(l, h), r| (l.min(r.0), h.max(r.1)));
                if hi - lo + 1 <= MAX_WINDOW_CUBES {
                    o.window = union_cubes(ranges).unwrap_or(o.window);
                }
            }
            None => out.push(g),
        }
    }
    out
}

/// [`plan`]'s groups, kept to the moves that can be made now without a cube
/// being in two shards:
///
/// - A merge needs the survivor's window to meet no shard that stays apart.
/// - A new shard needs its window to meet no window the others will hold once
///   they have moved this step (a shard grows by at most the ring a step; it
///   shrinks at once).
///
/// A group that fails its test keeps its players where they are (`stay`),
/// which can fail another group's test in turn, so it runs until nothing
/// changes. The shard a player stays in keeps its terrain: its window is
/// the union of the ranges of the players that stay.
pub fn settle(groups: Vec<Group>, members: &[Member], shards: &[Existing]) -> Vec<Group> {
    let mut groups = groups;
    for _ in 0..groups.len() + 2 {
        let merged: Vec<u16> = groups.iter().flat_map(|g| g.absorbed.clone()).collect();
        let current = |id: u16| shards.iter().find(|s| s.id == id).map(|s| s.window);
        // The most each shard's window can be after this step.
        let after = |id: u16| -> Option<Range> {
            if merged.contains(&id) {
                return None;
            }
            let have = current(id)?;
            Some(match groups.iter().find(|g| g.shard == Some(id)) {
                // A shrink is at once, and a window that does not overlap
                // the one it moves to jumps there; otherwise it grows by the
                // ring at most.
                Some(g) if within(g.window, have) || !overlaps(have, g.window) => g.window,
                Some(g) => hull(have, g.window),
                None => have,
            })
        };
        let blocked = groups
            .iter()
            .position(|g| match (g.shard, g.absorbed.is_empty()) {
                (None, _) => shards
                    .iter()
                    .filter_map(|s| after(s.id))
                    .any(|w| overlaps(w, g.window)),
                (Some(id), false) => shards
                    .iter()
                    .filter(|s| s.id != id && !merged.contains(&s.id))
                    .any(|s| overlaps(s.window, g.window)),
                (Some(_), true) => false,
            });
        let Some(i) = blocked else {
            break;
        };
        let g = groups.remove(i);
        groups.extend(stay(&g, members));
        groups = combine(groups, members);
    }
    groups.sort_by_key(|g| g.window);
    groups
}

/// `next` (a window a shard moves to) cut so it overlaps none of `others`
/// (the other shards' windows now) while it keeps `keep` (its players' cubes,
/// first and last). `None` when it cannot.
pub fn clip_window(next: Range, others: &[Range], keep: Range) -> Option<Range> {
    let mut next = next;
    for &other in others {
        if !overlaps(next, other) {
            continue;
        }
        // Cut on the side away from the players.
        if other.0 <= keep.0 {
            next.0 = next.0.max(other.1 + 1);
        } else {
            next.1 = next.1.min(other.0 - 1);
        }
    }
    (next.0 <= keep.0 && keep.1 <= next.1 && next.0 <= next.1).then_some(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubic::wanted_cubes;

    fn member(key: u32, cube: i32, shard: u16) -> Member {
        Member {
            key,
            cube,
            range: wanted_cubes(cube, 8),
            shard,
        }
    }

    fn shard(id: u16, window: Range) -> Existing {
        Existing { id, window }
    }

    #[test]
    fn gap_counts_the_cubes_between_ranges() {
        assert_eq!(gap((0, 10), (11, 20)), 0);
        assert_eq!(gap((0, 10), (12, 20)), 1);
        assert_eq!(gap((12, 20), (0, 10)), 1);
        assert_eq!(gap((0, 10), (10, 20)), -1);
        assert!(overlaps((0, 10), (10, 20)));
        assert!(!overlaps((0, 10), (11, 20)));
    }

    #[test]
    fn players_with_overlapping_ranges_share_one_shard_with_the_union_as_its_window() {
        let members = [member(1, 0, 0), member(2, 10, 0)];
        let p = plan(&members, &[shard(0, (-9, 19))]);
        assert_eq!(p.groups.len(), 1);
        assert_eq!(p.groups[0].shard, Some(0));
        assert_eq!(p.groups[0].members, vec![1, 2]);
        assert_eq!(p.groups[0].window, (-9, 19));
    }

    #[test]
    fn a_player_far_above_gets_a_shard_of_its_own() {
        // y 0 and y 20,000: cubes 0 and 1250.
        let members = [member(1, 0, 0), member(2, 1250, 0)];
        let p = plan(&members, &[shard(0, (-9, 9))]);
        assert_eq!(p.groups.len(), 2);
        let low = &p.groups[0];
        let high = &p.groups[1];
        assert_eq!((low.shard, low.members.clone()), (Some(0), vec![1]));
        assert_eq!(low.window, wanted_cubes(0, 8));
        assert_eq!((high.shard, high.members.clone()), (None, vec![2]));
        assert_eq!(high.window, wanted_cubes(1250, 8));
        assert!(!overlaps(low.window, high.window));
    }

    #[test]
    fn a_new_group_reuses_the_idle_primary_before_asking_for_a_new_shard() {
        // Shard 1 holds the player; the primary is empty far below.
        let members = [member(1, 500, 1), member(2, 2000, 1)];
        let p = plan(&members, &[shard(0, (-9, 9)), shard(1, (491, 2009))]);
        let shards: Vec<Option<u16>> = p.groups.iter().map(|g| g.shard).collect();
        assert!(
            shards.contains(&Some(1)) && shards.contains(&Some(0)),
            "{shards:?}"
        );
    }

    #[test]
    fn splitting_waits_for_a_gap_of_two_cubes_and_merging_for_a_touch() {
        // V = 8: ranges are 19 cubes tall, so players at cubes c and c + 19 + g
        // have a gap of g.
        let apart = |gap_cubes: i32, together: bool| {
            let (a, b) = (
                member(1, 0, 0),
                member(2, 19 + gap_cubes, u16::from(!together)),
            );
            let shards = [shard(0, (-9, 9)), shard(1, (10, 40))];
            plan(&[a, b], if together { &shards[..1] } else { &shards[..] })
        };
        // Together with a gap of one cube: still one shard (hysteresis).
        assert_eq!(apart(1, true).groups.len(), 1);
        // Together with a gap of two: split.
        assert_eq!(apart(2, true).groups.len(), 2);
        // Apart with a gap of one: still two.
        assert_eq!(apart(1, false).groups.len(), 2);
        // Apart with a gap of zero (touching): merge.
        let merged = apart(0, false);
        assert_eq!(merged.groups.len(), 1);
        assert_eq!(merged.groups[0].absorbed.len(), 1);
        // Apart and overlapping: merge too.
        assert_eq!(apart(-3, false).groups.len(), 1);
    }

    #[test]
    fn a_player_at_the_edge_does_not_flip_between_shards() {
        // The second player walks up and down across the split distance: the
        // gap to the first is its cube minus 19.
        let mut shard_of = 0u16;
        let mut seen = Vec::new();
        for cube in [18, 19, 20, 21, 20, 19, 20, 21, 22, 21, 20, 21] {
            let members = [member(1, 0, 0), member(2, cube, shard_of)];
            let existing = [shard(0, (-9, 9)), shard(1, (10, 40))];
            let p = plan(
                &members,
                if shard_of == 0 {
                    &existing[..1]
                } else {
                    &existing[..]
                },
            );
            let group = p.groups.iter().find(|g| g.members.contains(&2)).unwrap();
            shard_of = u16::from(group.members.len() == 1);
            seen.push(shard_of);
        }
        // Split at a gap of two (21), merged again at a touch (19), and at a gap of
        // one (20) it stays where it was.
        assert_eq!(seen, [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn a_shard_with_nobody_is_met_by_a_player_whose_range_reaches_it() {
        let members = [member(1, 40, 1)];
        let existing = [shard(0, (-9, 9)), shard(1, (31, 49))];
        // Nobody in the primary and it is far: it is left out.
        let p = plan(&members, &existing);
        assert_eq!(p.groups.len(), 1);
        assert_eq!(p.groups[0].shard, Some(1));
        assert!(p.groups[0].absorbed.is_empty());
        // The player climbs down to the idle primary's window: they merge.
        let members = [member(1, 15, 1)];
        let p = plan(&members, &existing);
        assert_eq!(p.groups.len(), 1);
        // The primary never closes: the player's shard merges into it.
        assert_eq!(p.groups[0].shard, Some(PRIMARY));
        assert_eq!(p.groups[0].absorbed, vec![1]);
    }

    #[test]
    fn a_tall_chain_is_cut_between_players_into_disjoint_windows() {
        // Each range is 51 cubes tall (V = 24); the chain spans 130.
        let wide = |key, cube| Member {
            key,
            cube,
            range: wanted_cubes(cube, 24),
            shard: 0,
        };
        let members = [wide(1, 0), wide(2, 40), wide(3, 80)];
        let p = plan(&members, &[shard(0, (-25, 25))]);
        assert!(p.groups.len() >= 2);
        for g in &p.groups {
            assert!(g.window.1 - g.window.0 < MAX_WINDOW_CUBES);
        }
        for pair in p.groups.windows(2) {
            assert!(!overlaps(pair[0].window, pair[1].window), "{pair:?}");
        }
        // Every player is inside its own group's window.
        for g in &p.groups {
            for key in &g.members {
                let m = members.iter().find(|m| m.key == *key).unwrap();
                assert!((g.window.0..=g.window.1).contains(&m.cube));
            }
        }
    }

    #[test]
    fn a_window_is_cut_to_stay_off_its_neighbours() {
        assert_eq!(clip_window((0, 40), &[(30, 60)], (5, 10)), Some((0, 29)));
        assert_eq!(clip_window((20, 70), &[(0, 29)], (40, 50)), Some((30, 70)));
        // The neighbour holds the player's cube: no window.
        assert_eq!(clip_window((0, 40), &[(5, 60)], (8, 10)), None);
        assert_eq!(clip_window((0, 10), &[], (3, 4)), Some((0, 10)));
    }

    #[test]
    fn a_new_shard_opens_once_the_shard_it_leaves_has_shrunk_or_jumped_away() {
        // Shard 0 holds a player at cube 30 but its window lags at [-9, 9]; a
        // player of shard 1 comes down to cube 5. Shard 0 jumps to the first
        // (no overlap), so the second can have the freed cubes as a shard.
        let members = [member(1, 30, 0), member(2, 70, 1), member(3, 5, 1)];
        let shards = [shard(0, (-9, 9)), shard(1, (60, 80))];
        let planned = plan(&members, &shards);
        let settled = settle(planned.groups, &members, &shards);
        assert_eq!(settled.len(), 3, "{settled:?}");
        let opened = settled.iter().find(|g| g.shard.is_none()).unwrap();
        assert_eq!(opened.members, vec![3]);
    }

    #[test]
    fn a_new_shard_waits_while_the_shard_next_to_it_grows_into_its_window() {
        // Shard 0 [0, 40] holds the players at 30 and 42 and grows to [21, 51],
        // which still covers cubes 0 to 9 of the window the player of shard 1
        // at cube 0 would open: it stays in shard 1 for now.
        let members = [
            member(1, 30, 0),
            member(2, 42, 0),
            member(3, 110, 1),
            member(4, 0, 1),
        ];
        let shards = [shard(0, (0, 40)), shard(1, (100, 120))];
        let planned = plan(&members, &shards);
        assert!(planned.groups.iter().any(|g| g.shard.is_none()));
        let settled = settle(planned.groups, &members, &shards);
        assert!(settled.iter().all(|g| g.shard.is_some()), "{settled:?}");
        let one = settled.iter().find(|g| g.shard == Some(1)).unwrap();
        assert!(one.members.contains(&3) && one.members.contains(&4));
    }

    #[test]
    fn a_merge_that_would_meet_a_third_shard_stays_apart() {
        // Shards 0 and 1 touch, but the survivor's window [-9, 28] would cover
        // cubes 20 to 28 of shard 2's window [20, 49], which holds a player of its own.
        let members = [member(1, 0, 0), member(2, 19, 1), member(3, 40, 2)];
        let shards = [shard(0, (-9, 9)), shard(1, (10, 28)), shard(2, (20, 49))];
        let planned = plan(&members, &shards);
        assert_eq!(planned.groups.len(), 2);
        let settled = settle(planned.groups, &members, &shards);
        let total: usize = settled.iter().map(|g| g.members.len()).sum();
        assert_eq!(total, 3);
        assert_eq!(settled.len(), 3, "{settled:?}");
        assert!(settled.iter().all(|g| g.absorbed.is_empty()));
    }
}
