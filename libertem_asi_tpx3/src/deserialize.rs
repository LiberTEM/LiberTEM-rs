use serde::{
    de::{self},
    Deserialize, Deserializer,
};

use serde::de::Visitor;
use std::fmt;

use crate::headers::{DType, FormatType};

impl<'de> Deserialize<'de> for FormatType {
    fn deserialize<D>(deserializer: D) -> Result<FormatType, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_u8(U8Visitor)
            .map(FormatType::from_u8)
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<DType, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u8(U8Visitor).map(DType::from_u8)
    }
}

struct U8Visitor;

impl<'de> Visitor<'de> for U8Visitor {
    type Value = u8;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer from 0 through 255")
    }

    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(value)
    }
}
