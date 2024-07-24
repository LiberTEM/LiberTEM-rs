use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::headers::{DType, FormatType};

impl Serialize for FormatType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = self;
        {
            let mut state = serializer.serialize_struct("FormatType", 1)?;
            state.serialize_field("type", &(*value as u8))?;
            state.end()
        }
    }
}

impl Serialize for DType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = self;
        {
            let mut state = serializer.serialize_struct("DType", 1)?;
            state.serialize_field("type", &(*value as u8))?;
            state.end()
        }
    }
}
