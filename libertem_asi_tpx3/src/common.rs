/// Makes `size` a multiple of `alignment`, rounding up.
pub fn align_to(size: usize, alignment: usize) -> usize {
    // while waiting for div_ceil to be stable, have this monstrosity:
    let div = size / alignment;
    let rem = size % alignment;

    if rem > 0 {
        alignment * (div + 1)
    } else {
        alignment * div
    }
}
