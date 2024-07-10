use std::{fmt::Debug, mem::replace, str::FromStr};

#[derive(Debug, thiserror::Error)]
pub enum NumParseError {
    #[error("expected number, got: {got:X?} ({err})")]
    ExpectedNumber {
        got: Vec<u8>,
        err: Box<dyn std::error::Error + 'static>,
    },

    #[error("expected UTF-8, got: {got:X?} ({err})")]
    Utf8Expected {
        got: Vec<u8>,
        err: Box<dyn std::error::Error + 'static>,
    },
}

/// Parse a decimal number from an ASCII/UTF8 string
pub fn num_from_byte_slice<T: FromStr>(bytes: &[u8]) -> Result<T, NumParseError>
where
    <T as std::str::FromStr>::Err: Debug + std::error::Error + 'static,
{
    // This should not become a bottleneck, but in case it does,
    // there is the `atoi` crate, which provides this functionality
    // without going via UTF8 first.
    let s = std::str::from_utf8(bytes).map_err(|e| NumParseError::Utf8Expected {
        got: bytes.to_vec(),
        err: Box::new(e),
    })?;
    let result = s.parse().map_err(|e| NumParseError::ExpectedNumber {
        got: bytes.to_vec(),
        err: Box::new(e),
    })?;
    Ok(result)
}

/// Puts `new` into `right`, `right` into `left` and returns the old `left`
pub fn three_way_shift<T>(left: &mut T, right: &mut T, new: T) -> T {
    let old_right = replace(right, new);
    replace(left, old_right)
}
