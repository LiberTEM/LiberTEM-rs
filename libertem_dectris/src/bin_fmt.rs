use serde::Serialize;
use std::io::{self, Write};

pub fn write_raw_msg(msg: &[u8]) {
    write_raw_msg_fh(msg, &mut io::stdout());
}

pub fn write_raw_msg_fh<W>(msg: &[u8], out: &mut W)
where
    W: Write,
{
    let length = (msg.len() as i64).to_le_bytes();
    out.write_all(&length).unwrap();
    out.write_all(msg).unwrap();
}

pub fn write_serializable<T>(value: &T)
where
    T: Serialize,
{
    let binding = serde_json::to_string(&value).expect("serialization should not fail");
    let msg_raw = binding.as_bytes();
    write_raw_msg(msg_raw);
}

pub fn write_serializable_fh<T, W>(value: &T, out: &mut W)
where
    T: Serialize,
    W: Write,
{
    let binding = serde_json::to_string(&value).expect("serialization should not fail");
    let msg_raw = binding.as_bytes();
    write_raw_msg_fh(msg_raw, out);
}
