//! The wide block position wire (cubic levels): one process-wide switch, so these
//! checks share one test and run in order.
use horizon_protocol::buf::{Codec, Pos, Reader, Writer, set_wide_positions, wide_positions};
use horizon_protocol::chunk::BlockEntityInfo;
use horizon_world::nbt::Tag;

fn pos_bytes(p: [i32; 3]) -> Vec<u8> {
    let mut w = Writer::new();
    Pos::write(&p, &mut w);
    w.data
}
fn info_bytes(y: i32) -> Vec<u8> {
    let mut w = Writer::new();
    let info = BlockEntityInfo {
        packed_xz: 0x21,
        y,
        kind: 5,
        tag: Tag::End,
    };
    BlockEntityInfo::write(&info, &mut w);
    w.data
}

#[test]
fn positions_are_vanilla_until_wide_and_exact_after() {
    assert!(!wide_positions());
    // BlockPos.asLong(1, 2, 3), as the vanilla wire has it.
    assert_eq!(
        pos_bytes([1, 2, 3]),
        ((1i64 << 38) | (3 << 12) | 2).to_be_bytes()
    );
    // A block entity's Y is a short: x/z byte, short, varint type, then the tag (End).
    assert_eq!(info_bytes(-60), [0x21, 0xFF, 0xC4, 5, 0]);

    set_wide_positions(true);
    let ys = [8_000_000, -8_000_000, 1_000_000, -64, 0];
    for y in ys {
        let p = [-33_000_000, y, 12_345_678];
        let bytes = pos_bytes(p);
        assert_eq!(bytes.len(), 12);
        assert_eq!(Pos::read(&mut Reader::new(&bytes)).unwrap(), p);

        let bytes = info_bytes(y);
        let back = BlockEntityInfo::read(&mut Reader::new(&bytes)).unwrap();
        assert_eq!((back.y, back.packed_xz, back.kind), (y, 0x21, 5));
    }
    // Packed fields keep the three ints: x, y, z in that order.
    let mut expect = Vec::new();
    for c in [7i32, -9, 11] {
        expect.extend_from_slice(&c.to_be_bytes());
    }
    assert_eq!(pos_bytes([7, -9, 11]), expect);
    set_wide_positions(false);
    assert_eq!(
        pos_bytes([1, 2, 3]),
        ((1i64 << 38) | (3 << 12) | 2).to_be_bytes()
    );
}
