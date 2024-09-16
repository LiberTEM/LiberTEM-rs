use std::{
    any::type_name,
    collections::HashMap,
    fmt::Debug,
    str::{FromStr, Split, Utf8Error},
    string::FromUtf8Error,
    time::Duration,
};

use common::{
    frame_stack::FrameMeta,
    generic_connection::{AcquisitionConfig, DetectorConnectionConfig},
};
use log::{trace, warn};
use num::Num;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyBytes, PyBytesMethods, PyType},
    Bound, PyResult,
};
use serde::{Deserialize, Serialize};

/// Size of the full prefix, in bytes, including the comma separator to the payload: 'MPX,<length>,'
pub const PREFIX_SIZE: usize = 15;

#[derive(Debug, thiserror::Error)]
pub enum FrameMetaParseError {
    #[error("frame header is not valid utf8: {err}")]
    Utf8Expected {
        #[from]
        err: FromUtf8Error,
    },

    #[error("frame header is not valid utf8: {err}")]
    Utf8Error {
        #[from]
        err: Utf8Error,
    },

    #[error("unknown version: {prefix}")]
    UnknownVersion { prefix: String },

    #[error("unknown variant for enum: {variant}")]
    UnknownVariant { variant: String },

    #[error("unexpected end of header")]
    Eof,

    #[error("value conversion error: {msg}")]
    ValueError { msg: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    U01,
    U08,
    U16,
    U32,
    U64,
    R64,
}

impl FromStr for DType {
    type Err = FrameMetaParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "U01" => Self::U01,
            "U08" => Self::U08,
            "U16" => Self::U16,
            "U32" => Self::U32,
            "U64" => Self::U64,
            "R64" => Self::R64,
            _ => {
                return Err(FrameMetaParseError::UnknownVariant {
                    variant: s.to_owned(),
                })
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Layout {
    L1x1,
    L2x2,
    LNx1,
    L2x2G,
    LNx1G,
}

impl FromStr for Layout {
    type Err = FrameMetaParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // input is padded with leading spaces:
        Ok(match s.trim_start() {
            "1x1" => Self::L1x1,
            "2x2" => Self::L2x2,
            "Nx1" => Self::LNx1,
            "2x2G" => Self::L2x2G,
            "Nx1G" => Self::LNx1G,
            _ => {
                return Err(FrameMetaParseError::UnknownVariant {
                    variant: s.to_owned(),
                })
            }
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NextPartError {
    #[error("unexpected end of header")]
    Eof,

    #[error("value conversion error: {msg}")]
    ValueError { msg: String },
}

impl From<NextPartError> for FrameMetaParseError {
    fn from(value: NextPartError) -> Self {
        match value {
            NextPartError::Eof => Self::Eof,
            NextPartError::ValueError { msg } => Self::ValueError { msg },
        }
    }
}

fn next_part_from_str<T: FromStr>(parts: &mut Split<char>) -> Result<T, NextPartError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let part_str = parts.next().ok_or(NextPartError::Eof)?;
    let value: T = part_str.parse().map_err(|e| NextPartError::ValueError {
        msg: format!(
            "unexpected value: '{part_str}', expected {} ({e:?})",
            type_name::<T>()
        ),
    })?;
    Ok(value)
}

fn next_part_from_str_with_map<T: FromStr>(
    parts: &mut Split<char>,
    f: impl Fn(&str) -> Result<&str, NextPartError>,
) -> Result<T, NextPartError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let part_str = parts.next().ok_or(NextPartError::Eof)?;
    let part_str = f(part_str)?;
    let value: T = part_str.parse().map_err(|e| NextPartError::ValueError {
        msg: format!(
            "unexpected value: '{part_str}', expected {} ({e:?})",
            type_name::<T>()
        ),
    })?;
    Ok(value)
}

fn next_part_from_str_radix<T: Num>(parts: &mut Split<char>, radix: u32) -> Result<T, NextPartError>
where
    <T as num::Num>::FromStrRadixErr: std::fmt::Debug,
{
    let part_str = parts.next().ok_or(NextPartError::Eof)?;
    let value = T::from_str_radix(part_str, radix).map_err(|e| NextPartError::ValueError {
        msg: format!(
            "unexpected value: '{part_str}', expected {} ({e:?}); radix={radix}",
            type_name::<T>()
        ),
    })?;
    Ok(value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MQ1A {
    pub timestamp_ext: String,

    pub acquisition_time_shutter_ns: u64,

    pub counter_depth: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColourMode {
    Single,
    Multi,
}

impl FromStr for ColourMode {
    type Err = FrameMetaParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let int_val: u8 = s.parse().map_err(|_| FrameMetaParseError::ValueError {
            msg: format!("invalid colour mode: {s}"),
        })?;
        Ok(match int_val {
            0 => Self::Single,
            1 => Self::Multi,
            _ => {
                return Err(FrameMetaParseError::UnknownVariant {
                    variant: s.to_owned(),
                })
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gain {
    SLGM,
    LGM,
    HGM,
    SHGM,
}

impl FromStr for Gain {
    type Err = FrameMetaParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let int_val: u8 = s.parse().map_err(|_| FrameMetaParseError::ValueError {
            msg: format!("invalid gain: {s}"),
        })?;
        Ok(match int_val {
            0 => Self::SLGM,
            1 => Self::LGM,
            2 => Self::HGM,
            3 => Self::SHGM,
            _ => {
                return Err(FrameMetaParseError::UnknownVariant {
                    variant: s.to_owned(),
                })
            }
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdFrameMeta {
    mpx_length: usize,

    sequence: u32,

    data_offset: u16,

    num_chips: u8,

    width_in_pixels: u32,

    height_in_pixels: u32,

    width_in_pixels_raw: u32,

    height_in_pixels_raw: u32,

    pub dtype: DType,

    pub layout: Layout,

    chip_select: u8,

    timestamp: String,

    acquisition_shutter_time: f64,

    counter: u8,

    colour_mode: ColourMode,

    gain_mode: Gain,

    pub mq1a: Option<MQ1A>,
}

impl QdFrameMeta {
    // this is only used internally for testing... could also make everything
    // pub or pub(crate) instead...
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn new(
        mpx_length: usize,
        sequence: u32,
        data_offset: u16,
        num_chips: u8,
        width_in_pixels_raw: u32,
        height_in_pixels_raw: u32,
        width_in_pixels: u32,
        height_in_pixels: u32,
        dtype: DType,
        layout: Layout,
        chip_select: u8,
        timestamp: String,
        acquisition_shutter_time: f64,
        counter: u8,
        colour_mode: ColourMode,
        gain_mode: Gain,
        mq1a: Option<MQ1A>,
    ) -> Self {
        Self {
            mpx_length,
            sequence,
            data_offset,
            num_chips,
            width_in_pixels_raw,
            height_in_pixels_raw,
            width_in_pixels,
            height_in_pixels,
            dtype,
            layout,
            chip_select,
            timestamp,
            acquisition_shutter_time,
            counter,
            colour_mode,
            gain_mode,
            mq1a,
        }
    }

    /// Parse frame header, including the MQ1 prefix
    pub fn parse_bytes(input: &[u8], mpx_length: usize) -> Result<Self, FrameMetaParseError> {
        if input.len() < 16 {
            return Err(FrameMetaParseError::Eof);
        }

        // first, split off the header if there is more stuff after it:
        let data_offset: u32 = std::str::from_utf8(&input[11..16])?.parse().map_err(|e| {
            FrameMetaParseError::ValueError {
                msg: format!("could not parse data offset: {e:?} ({:X?})", &input[11..16]),
            }
        })?;

        let input = if input.len() > data_offset as usize {
            &input[..data_offset as usize]
        } else {
            input
        };

        let input_string = std::str::from_utf8(input)?;

        let mut parts = input_string.split(',');

        let magic = parts.next().ok_or(FrameMetaParseError::Eof)?;
        if magic != "MQ1" {
            return Err(FrameMetaParseError::UnknownVersion {
                prefix: magic.to_owned(),
            });
        }

        let sequence: u32 = next_part_from_str(&mut parts)?;
        let data_offset: u16 = next_part_from_str(&mut parts)?;
        let num_chips: u8 = next_part_from_str(&mut parts)?;
        let width_in_pixels_raw: u32 = next_part_from_str(&mut parts)?;
        let height_in_pixels_raw: u32 = next_part_from_str(&mut parts)?;
        let dtype: DType = next_part_from_str(&mut parts)?;
        let layout: Layout = next_part_from_str(&mut parts)?;
        let chip_select: u8 = next_part_from_str_radix(&mut parts, 16)?;
        let timestamp: String = next_part_from_str(&mut parts)?;
        let acquisition_shutter_time: f64 = next_part_from_str(&mut parts)?;
        let counter = next_part_from_str(&mut parts)?;
        let colour_mode = next_part_from_str(&mut parts)?;
        let gain_mode = next_part_from_str(&mut parts)?;

        // thresholds 0..7?

        let mq1a = if parts.any(|item| item == "MQ1A") {
            let timestamp_ext = next_part_from_str(&mut parts)?;
            let acquisition_time_shutter_ns =
                next_part_from_str_with_map(&mut parts, |s| Ok(&s[0..s.len() - 2]))?;
            let counter_depth = next_part_from_str_with_map(&mut parts, |s| Ok(s.trim()))?;

            Some(MQ1A {
                timestamp_ext,
                acquisition_time_shutter_ns,
                counter_depth,
            })
        } else {
            None
        };

        // fixups: layout -> width/height mapping
        let width_in_pixels = if dtype == DType::R64 {
            match layout {
                Layout::L1x1 => width_in_pixels_raw,
                Layout::LNx1 => 1024,
                Layout::LNx1G => 1024, // FIXME: gap support?

                Layout::L2x2 => 512,
                Layout::L2x2G => 514,
            }
        } else {
            width_in_pixels_raw
        };

        let height_in_pixels = if dtype == DType::R64 {
            match layout {
                Layout::L1x1 => height_in_pixels_raw,
                Layout::LNx1 => 256,
                Layout::LNx1G => 256, // FIXME: gap support?

                Layout::L2x2 => 512,
                Layout::L2x2G => 514,
            }
        } else {
            height_in_pixels_raw
        };

        Ok(Self {
            mpx_length,
            sequence,
            data_offset,
            num_chips,
            width_in_pixels_raw,
            height_in_pixels_raw,
            width_in_pixels,
            height_in_pixels,
            dtype,
            layout,
            chip_select,
            timestamp,
            acquisition_shutter_time,
            counter,
            colour_mode,
            gain_mode,
            mq1a,
        })
    }

    /// Get the length of the message for the frame, starting with and including
    /// the 'MQ1' prefix. This is the same value that is sent in the MPX prefix.
    pub fn get_mpx_length(&self) -> usize {
        self.mpx_length
    }

    /// size of the frame header on the wire, including the MPX prefix:
    pub fn get_total_size_header(&self) -> usize {
        self.data_offset as usize + PREFIX_SIZE
    }

    /// Get the sequence number of the frame, starting from 1, as in the frame
    /// header.
    pub fn get_sequence(&self) -> u32 {
        self.sequence
    }
}

impl FrameMeta for QdFrameMeta {
    fn get_data_length_bytes(&self) -> usize {
        // minus one here because... mpx_length includes the comma, data_offset doesn't?
        self.mpx_length - self.data_offset as usize - 1
    }

    fn get_dtype_string(&self) -> String {
        let raw_dtype = if let Some(mq1a) = &self.mq1a {
            match mq1a.counter_depth {
                1 | 6 => "uint8",
                12 => "uint16",
                24 => "uint32",
                _ => {
                    // FIXME: unknown counter depth; can't determine raw dtype!
                    "uint8"
                }
            }
        } else {
            // FIXME: not really correct, but we can't know the data type if the
            // detector doesn't include the mq1a header!
            "uint8"
        };

        match self.dtype {
            DType::U01 | DType::U08 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::R64 => raw_dtype,
        }
        .to_owned()
    }

    fn get_shape(&self) -> (u64, u64) {
        (self.width_in_pixels as u64, self.height_in_pixels as u64)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AcqHeaderParseError {
    #[error("acquisition header is not valid utf8: {err}")]
    Utf8Expected {
        #[from]
        err: FromUtf8Error,
    },

    #[error("acquisition header is not valid utf8: {err}")]
    Utf8Error {
        #[from]
        err: Utf8Error,
    },

    #[error("syntax error: line='{line}'")]
    SyntaxError { line: String },

    #[error("syntax error: {msg} (token: '{token}')")]
    ValueError { token: String, msg: String },

    #[error("missing key: {key}")]
    MissingKey { key: String },
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct QdAcquisitionHeader {
    frames_in_acquisition: usize,
    frames_per_trigger: usize,
    scan_x: Option<usize>,
    scan_y: Option<usize>,
    raw_kv: HashMap<String, String>,
}

#[pymethods]
impl QdAcquisitionHeader {
    #[classmethod]
    fn new_from_bytes(_cls: &Bound<'_, PyType>, input: Bound<'_, PyBytes>) -> PyResult<Self> {
        Self::parse_bytes(input.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("invalid acquisition header: {e}")))
    }

    fn frames_in_acquisition(&self) -> usize {
        self.frames_in_acquisition
    }

    fn frames_per_trigger(&self) -> usize {
        self.frames_per_trigger
    }
}

impl QdAcquisitionHeader {
    pub fn num_frames(&self) -> usize {
        self.frames_in_acquisition
    }

    pub fn nav_shape(&self) -> Option<(usize, usize)> {
        if let (Some(scan_x), Some(scan_y)) = (self.scan_x, self.scan_y) {
            return Some((scan_y, scan_x));
        }
        None
    }
}

fn get_key_and_parse<T: FromStr>(
    raw_kv: &HashMap<String, String>,
    key: &str,
) -> Result<T, AcqHeaderParseError>
where
    <T as FromStr>::Err: ToString,
{
    let owned_key = key.to_owned();
    let val = raw_kv
        .get(&owned_key)
        .ok_or(AcqHeaderParseError::MissingKey { key: owned_key })?;
    val.trim()
        .parse()
        .map_err(|e: <T as FromStr>::Err| AcqHeaderParseError::ValueError {
            token: val.clone(),
            msg: e.to_string(),
        })
}

fn get_key_and_parse_optional<T: FromStr>(
    raw_kv: &HashMap<String, String>,
    key: &str,
) -> Result<Option<T>, AcqHeaderParseError>
where
    <T as FromStr>::Err: ToString,
{
    let owned_key = key.to_owned();
    let maybe_val = raw_kv.get(&owned_key);
    Ok(if let Some(val) = maybe_val {
        Some(val.trim().parse().map_err(|e: <T as FromStr>::Err| {
            AcqHeaderParseError::ValueError {
                token: val.clone(),
                msg: e.to_string(),
            }
        })?)
    } else {
        None
    })
}

impl QdAcquisitionHeader {
    pub fn parse_bytes(input: &[u8]) -> Result<Self, AcqHeaderParseError> {
        // the acquisition header can actually be latin-1 (µA...) so let's try that:
        let input_string = encoding_rs::mem::decode_latin1(input);
        trace!(
            "parsing acquisition header: {}",
            String::from_utf8_lossy(input)
        );
        // trace!("parsing acquisition header: {}", String::from_utf8_lossy(&input[1003..]));

        // let input= input.strip_suffix(&[0]).unwrap_or(input);
        // let input_string = std::str::from_utf8(input)?;
        let mut raw_kv: HashMap<String, String> = HashMap::new();
        let mut parts = input_string.split('\n');

        let _ = parts.next(); // first line: 'HDR,'...

        for line in parts {
            if line.trim() == "End" {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                raw_kv
                    .entry(k.trim().to_string())
                    .or_insert(v.trim().to_owned());
            } else {
                warn!("ignoring broken acquisition header line: {line}")
            }
        }
        let frames_in_acquisition: u32 =
            get_key_and_parse(&raw_kv, "Frames in Acquisition (Number)")?;

        let frames_per_trigger: u32 = get_key_and_parse(&raw_kv, "Frames per Trigger (Number)")?;

        let scan_x = get_key_and_parse_optional(&raw_kv, "ScanX")?;
        let scan_y = get_key_and_parse_optional(&raw_kv, "ScanY")?;

        Ok(Self {
            frames_in_acquisition: frames_in_acquisition as usize,
            frames_per_trigger: frames_per_trigger as usize,
            scan_x,
            scan_y,
            raw_kv,
        })
    }

    pub fn get_raw_kv(&self) -> &HashMap<String, String> {
        &self.raw_kv
    }

    pub fn get_scan_x(&self) -> Option<usize> {
        self.scan_x
    }

    pub fn get_scan_y(&self) -> Option<usize> {
        self.scan_y
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecoveryStrategyError {
    #[error("invalid variant: {variant}")]
    InvalidVariant { variant: String },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum RecoveryStrategy {
    #[default]
    ImmediateReconnect,
    DrainThenReconnect,
}

impl FromStr for RecoveryStrategy {
    type Err = RecoveryStrategyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "immediate_reconnect" => Self::ImmediateReconnect,
            "drain_then_reconnect" => Self::DrainThenReconnect,
            _ => {
                return Err(RecoveryStrategyError::InvalidVariant {
                    variant: s.to_owned(),
                })
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct QdDetectorConnConfig {
    pub data_host: String,
    pub data_port: usize,

    /// number of frames per frame stack; approximated because of compression
    pub frame_stack_size: usize,

    /// approx. number of bytes per frame, used for sizing frame stacks together
    /// with `frame_stack_size`
    pub bytes_per_frame: usize,

    pub drain_on_connect: Option<Duration>,

    pub recovery_strategy: RecoveryStrategy,

    num_slots: usize,
    enable_huge_pages: bool,
    shm_handle_path: String,
}

impl QdDetectorConnConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_host: &str,
        data_port: usize,
        frame_stack_size: usize,
        bytes_per_frame: usize,
        num_slots: usize,
        enable_huge_pages: bool,
        shm_handle_path: &str,
        drain_on_connect: Option<Duration>,
        recovery_strategy: RecoveryStrategy,
    ) -> Self {
        Self {
            data_host: data_host.to_owned(),
            data_port,
            frame_stack_size,
            bytes_per_frame,
            num_slots,
            enable_huge_pages,
            shm_handle_path: shm_handle_path.to_owned(),
            drain_on_connect,
            recovery_strategy,
        }
    }
}

impl DetectorConnectionConfig for QdDetectorConnConfig {
    fn get_shm_num_slots(&self) -> usize {
        self.num_slots
    }

    fn get_shm_slot_size(&self) -> usize {
        self.frame_stack_size * self.bytes_per_frame
    }

    fn get_shm_enable_huge_pages(&self) -> bool {
        self.enable_huge_pages
    }

    fn get_shm_handle_path(&self) -> String {
        self.shm_handle_path.clone()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct QdAcquisitionConfig {
    acq_header: QdAcquisitionHeader,
    frame_header: QdFrameMeta,
}

impl QdAcquisitionConfig {
    pub fn new(acq_header: QdAcquisitionHeader, frame_header: QdFrameMeta) -> Self {
        Self {
            acq_header,
            frame_header,
        }
    }
}

impl AcquisitionConfig for QdAcquisitionConfig {
    fn num_frames(&self) -> usize {
        self.acq_header.frames_in_acquisition
    }
}

#[pymethods]
impl QdAcquisitionConfig {
    fn frames_in_acquisition(&self) -> usize {
        self.acq_header.frames_in_acquisition
    }

    fn frames_per_trigger(&self) -> usize {
        self.acq_header.frames_per_trigger
    }

    fn nav_shape(&self) -> Option<(usize, usize)> {
        self.acq_header.nav_shape()
    }

    fn detector_shape(&self) -> (u32, u32) {
        (
            self.frame_header.height_in_pixels,
            self.frame_header.width_in_pixels,
        )
    }
}

#[cfg(test)]
mod test {
    use common::frame_stack::FrameMeta;

    use crate::base_types::{DType, Layout};

    use super::{QdAcquisitionHeader, QdFrameMeta};

    #[test]
    fn test_parse_frame_header_example_1_single() {
        let inp = "MQ1,000001,00384,01,0256,0256,U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,6,";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 384 + 256 * 256 + 1).unwrap();

        assert_eq!(fm.get_sequence(), 1);
        assert_eq!(fm.get_mpx_length(), 384 + 256 * 256 + 1);
        assert_eq!(fm.data_offset, 384);
        assert_eq!(fm.num_chips, 1);
        assert_eq!(fm.width_in_pixels, 256);
        assert_eq!(fm.height_in_pixels, 256);
        assert_eq!(fm.dtype, DType::U08);
        assert_eq!(fm.layout, Layout::L1x1);
        assert_eq!(fm.get_total_size_header(), 384 + 15);

        let mq1a = fm.mq1a.unwrap();
        assert_eq!(mq1a.acquisition_time_shutter_ns, 555_000);
        assert_eq!(mq1a.counter_depth, 6);
    }

    #[test]
    fn test_parse_frame_header_example_2_quad() {
        let inp = "MQ1,000001,00768,04,0514,0514,U16,  2x2G,0F,2024-06-14 11:31:30.060732,0.001000,0,0,0,3.500000E+1,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,052,511,000,000,000,000,000,000,100,255,100,125,100,100,081,100,074,030,128,004,255,136,128,188,176,000,000,3RX,055,511,000,000,000,000,000,000,100,255,100,125,100,100,079,100,077,030,128,004,255,138,128,191,188,000,000,3RX,052,511,000,000,000,000,000,000,100,255,100,125,100,100,077,100,075,030,128,004,255,140,128,197,177,000,000,3RX,049,511,000,000,000,000,000,000,100,255,100,125,100,100,070,100,073,030,128,004,255,140,128,183,179,000,000,MQ1A,2024-06-14T09:31:30.060732344Z,1000000ns,12,";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 768 + (2 * 514 * 514) + 1).unwrap();

        assert_eq!(fm.get_sequence(), 1);
        assert_eq!(fm.get_mpx_length(), 768 + (2 * 514 * 514) + 1);
        assert_eq!(fm.data_offset, 768);
        assert_eq!(fm.num_chips, 4);
        assert_eq!(fm.width_in_pixels, 514);
        assert_eq!(fm.height_in_pixels, 514);
        assert_eq!(fm.dtype, DType::U16);
        assert_eq!(fm.layout, Layout::L2x2G);
        assert_eq!(fm.get_total_size_header(), 768 + 15);

        let mq1a = fm.mq1a.unwrap();
        assert_eq!(mq1a.acquisition_time_shutter_ns, 1_000_000);
        assert_eq!(mq1a.counter_depth, 12);
    }

    #[test]
    fn test_quad_raw_correct_shape() {
        let inp = r"MQ1,000001,00768,04,1024,0256,R64,  2x2G,0F,2024-04-25 15:50:41.049778,0.000309,0,0,0,3.400000E+1,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,051,511,000,000,000,000,000,000,100,255,100,125,100,100,081,100,074,030,128,004,255,136,128,188,176,000,000,3RX,054,511,000,000,000,000,000,000,100,255,100,125,100,100,079,100,077,030,128,004,255,138,128,191,188,000,000,3RX,051,511,000,000,000,000,000,000,100,255,100,125,100,100,077,100,075,030,128,004,255,140,128,197,177,000,000,3RX,048,511,000,000,000,000,000,000,100,255,100,125,100,100,070,100,073,030,128,004,255,140,128,183,179,000,000,MQ1A,2024-04-25T13:50:41.049778723Z,309200ns,6,";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 768 + (514 * 514) + 1).unwrap();

        assert_eq!(fm.get_sequence(), 1);
        assert_eq!(fm.get_mpx_length(), 768 + (514 * 514) + 1);
        assert_eq!(fm.data_offset, 768);
        assert_eq!(fm.num_chips, 4);
        assert_eq!(fm.width_in_pixels, 514);
        assert_eq!(fm.height_in_pixels, 514);
        assert_eq!(fm.dtype, DType::R64);
        assert_eq!(fm.layout, Layout::L2x2G);
        assert_eq!(fm.get_total_size_header(), 768 + 15);

        let mq1a = fm.mq1a.unwrap();
        assert_eq!(mq1a.acquisition_time_shutter_ns, 309_200);
        assert_eq!(mq1a.counter_depth, 6);
    }

    #[test]
    fn test_parse_frame_header_example_3_raw_6bit() {
        let inp = "MQ1,000001,00384,01,0256,0256,R64,   1x1,01,2020-05-25 17:48:56.475280,0.001000,0,0,0,,MQ1A,2020-05-25T15:48:56.475280991Z,1000000ns,6,";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 384 + 256 * 256 + 1).unwrap();

        assert_eq!(fm.get_sequence(), 1);
        assert_eq!(fm.get_mpx_length(), 384 + 256 * 256 + 1);
        assert_eq!(fm.data_offset, 384);
        assert_eq!(fm.num_chips, 1);
        assert_eq!(fm.width_in_pixels, 256);
        assert_eq!(fm.height_in_pixels, 256);
        assert_eq!(fm.dtype, DType::R64);
        assert_eq!(fm.layout, Layout::L1x1);
        assert_eq!(fm.get_total_size_header(), 384 + 15);

        let mq1a = fm.mq1a.clone().unwrap();
        assert_eq!(mq1a.acquisition_time_shutter_ns, 1_000_000);
        assert_eq!(mq1a.counter_depth, 6);

        assert_eq!(fm.get_data_length_bytes(), 256 * 256);
        assert_eq!(fm.get_shape(), (256, 256));
    }

    #[test]
    fn test_parse_frame_header_from_mock() {
        let inp = "MQ1,000001,00340,01,0016,0016,U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,6\n";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 384 + 256 * 256 + 1).unwrap();

        eprintln!("{fm:?}");
    }

    #[test]
    fn test_parse_frame_header_from_mock_with_additional_stuff() {
        let inp = "MQ1,000001,00340,01,0016,0016,U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,6\n";
        let inp_bytes = inp.as_bytes();
        let inp_plus_data: Vec<u8> = inp_bytes
            .iter()
            .chain(vec![0; 16 * 16].iter())
            .copied()
            .collect();
        let fm = QdFrameMeta::parse_bytes(&inp_plus_data, 384 + 256 * 256 + 1).unwrap();

        eprintln!("{fm:?}");
    }

    #[test]
    fn test_parse_frame_header_example_4_u16_quad() {
        let inp = "MQ1,000001,00768,04,0512,0512,U16,   2x2,0F,2022-03-28 15:06:08.361851,0.000832,0,0,0,1.000000E+1,5.000000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,033,511,000,000,000,000,000,000,100,255,100,125,100,100,065,100,069,030,128,004,255,145,128,199,191,511,511,3RX,030,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,064,030,128,004,255,143,128,201,193,511,511,3RX,032,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,071,030,128,004,255,147,128,191,184,511,511,3RX,030,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,074,030,128,004,255,147,128,191,184,511,511,MQ1A,2022-03-28T14:06:08.361851667Z,831990ns,12,";
        let inp_bytes = inp.as_bytes();
        let fm = QdFrameMeta::parse_bytes(inp_bytes, 768 + 512 * 512 * 2 + 1).unwrap();

        assert_eq!(fm.get_sequence(), 1);
        assert_eq!(fm.get_mpx_length(), 768 + 512 * 512 * 2 + 1);
        assert_eq!(fm.data_offset, 768);
        assert_eq!(fm.num_chips, 4);
        assert_eq!(fm.width_in_pixels, 512);
        assert_eq!(fm.height_in_pixels, 512);
        assert_eq!(fm.dtype, DType::U16);
        assert_eq!(fm.layout, Layout::L2x2);
        assert_eq!(fm.get_total_size_header(), 768 + 15);

        let mq1a = fm.mq1a.clone().unwrap();
        assert_eq!(mq1a.acquisition_time_shutter_ns, 831_990);
        assert_eq!(mq1a.counter_depth, 12);

        assert_eq!(fm.get_data_length_bytes(), 2 * 512 * 512);
        assert_eq!(fm.get_shape(), (512, 512));
    }

    #[test]
    fn test_parse_acquisition_header_example_1() {
        let inp = r"HDR,	
Time and Date Stamp (day, mnth, yr, hr, min, s):	18/05/2020 16:51:48
Chip ID:	W529_F5,-,-,-
Chip Type (Medipix 3.0, Medipix 3.1, Medipix 3RX):	Medipix 3RX
Assembly Size (NX1, 2X2):	   1x1
Chip Mode  (SPM, CSM, CM, CSCM):	SPM
Counter Depth (number):	6
Gain:	SLGM
Active Counters:	Alternating
Thresholds (keV):	1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
DACs:	175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511
bpc File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.bpc,,,
DAC File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.dacs,,,
Gap Fill Mode:	Distribute
Flat Field File:	None
Dead Time File:	Dummy (C:\<NUL>\)
Acquisition Type (Normal, Th_scan, Config):	Normal
Frames in Acquisition (Number):	16384
Frames per Trigger (Number):	128
Trigger Start (Positive, Negative, Internal):	Rising Edge LVDS
Trigger Stop (Positive, Negative, Internal):	Internal
Sensor Bias (V):	120 V
Sensor Polarity (Positive, Negative):	Positive
Temperature (C):	Board Temp 37.384918 Deg C
Humidity (%):	Board Humidity 1.331848 
Medipix Clock (MHz):	120MHz
Readout System:	Merlin Quad
Software Version:	0.67.0.9
End	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
";
        let inp_bytes = inp.as_bytes();
        let header = QdAcquisitionHeader::parse_bytes(inp_bytes).unwrap();
        assert_eq!(header.num_frames(), 16384);
    }

    #[test]
    fn test_parse_acquisition_header_example_2_scan_xy() {
        let inp = r"HDR,	
Time and Date Stamp (day, mnth, yr, hr, min, s):	6/14/2024 11:31:21 AM
Chip ID:	W530_D4,W530_E4,W530_C4,W530_L5
Chip Type (Medipix 3.0, Medipix 3.1, Medipix 3RX):	Medipix 3RX
Assembly Size (NX1, 2X2):	  2x2G
Chip Mode  (SPM, CSM, CM, CSCM):	SPM
Counter Depth (number):	12
Gain:	SLGM
Active Counters:	Alternating
Thresholds (keV):	3.500000E+1,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
DACs:	052,511,000,000,000,000,000,000,100,255,100,125,100,100,081,100,074,030,128,004,255,136,128,188,176,000,000; 055,511,000,000,000,000,000,000,100,255,100,125,100,100,079,100,077,030,128,004,255,138,128,191,188,000,000; 052,511,000,000,000,000,000,000,100,255,100,125,100,100,077,100,075,030,128,004,255,140,128,197,177,000,000; 049,511,000,000,000,000,000,000,100,255,100,125,100,100,070,100,073,030,128,004,255,140,128,183,179,000,000
bpc File:	c:\Merlin_Quad_Config\W530_D4\W530_D4_SPM.bpc,c:\Merlin_Quad_Config\W530_E4\W530_E4_SPM.bpc,c:\Merlin_Quad_Config\W530_C4\W530_C4_SPM.bpc,c:\Merlin_Quad_Config\W530_L5\W530_L5_SPM.bpc
DAC File:	c:\Merlin_Quad_Config\W530_D4\W530_D4_SPM.dacs,c:\Merlin_Quad_Config\W530_E4\W530_E4_SPM.dacs,c:\Merlin_Quad_Config\W530_C4\W530_C4_SPM.dacs,c:\Merlin_Quad_Config\W530_L5\W530_L5_SPM.dacs
Gap Fill Mode:	None
Flat Field File:	C:\USERS\MERLIN\DOCUMENTS\SETUP\FLATFIELDS\200KV\FLATFIED_RDP_2024_06_13_200KV_T0_35.MIB
Dead Time File:	Dummy (C:\<NUL>\)
Acquisition Type (Normal, Th_scan, Config):	Normal
Frames in Acquisition (Number):	580
Frames per Trigger (Number):	580
Trigger Start (Positive, Negative, Internal):	Rising Edge
Trigger Stop (Positive, Negative, Internal):	Rising Edge
Sensor Bias (V):	120 V
Sensor Polarity (Positive, Negative):	Positive
Temperature (C):	Board Temp 50.630413 Deg C
Humidity (%):	Board Humidity -0.590759 
Medipix Clock (MHz):	160MHz
Readout System:	Merlin Quad
Software Version:	1.6.1.217
ScanX:	28
ScanY:	20
End	                                                                                                
";
        let inp_bytes = inp.as_bytes();
        let header = QdAcquisitionHeader::parse_bytes(inp_bytes).unwrap();
        assert_eq!(header.num_frames(), 580);
        assert_eq!(header.get_scan_x(), Some(28));
        assert_eq!(header.get_scan_y(), Some(20));

        assert_eq!(header.nav_shape(), Some((20, 28)));
    }

    #[test]
    fn test_parse_acquisition_header_example_3_quad_u16() {
        let inp = r"HDR,	
Time and Date Stamp (day, mnth, yr, hr, min, s):	28/03/2022 15:01:56
Chip ID:	W530_J6,W530_L6,W530_K5,W530_B5
Chip Type (Medipix 3.0, Medipix 3.1, Medipix 3RX):	Medipix 3RX
Assembly Size (NX1, 2X2):	   2x2
Chip Mode  (SPM, CSM, CM, CSCM):	SPM
Counter Depth (number):	12
Gain:	SLGM
Active Counters:	Alternating
Thresholds (keV):	1.000000E+1,5.000000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
DACs:	033,511,000,000,000,000,000,000,100,255,100,125,100,100,065,100,069,030,128,004,255,145,128,199,191,511,511; 030,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,064,030,128,004,255,143,128,201,193,511,511; 032,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,071,030,128,004,255,147,128,191,184,511,511; 030,511,000,000,000,000,000,000,100,255,100,125,100,100,066,100,074,030,128,004,255,147,128,191,184,511,511
bpc File:	c:\MERLIN_Quad_Config\W530_J6\W530_J6_SPM.bpc,c:\MERLIN_Quad_Config\W530_L6\W530_L6_SPM.bpc,c:\MERLIN_Quad_Config\W530_K5\W530_K5_SPM.bpc,c:\MERLIN_Quad_Config\W530_B5\W530_B5_SPM.bpc
DAC File:	c:\MERLIN_Quad_Config\W530_J6\W530_J6_SPM.dacs,c:\MERLIN_Quad_Config\W530_L6\W530_L6_SPM.dacs,c:\MERLIN_Quad_Config\W530_K5\W530_K5_SPM.dacs,c:\MERLIN_Quad_Config\W530_B5\W530_B5_SPM.dacs
Gap Fill Mode:	None
Flat Field File:	None
Dead Time File:	Dummy (C:\<NUL>\)
Acquisition Type (Normal, Th_scan, Config):	Normal
Frames in Acquisition (Number):	65536
Frames per Trigger (Number):	65536
Trigger Start (Positive, Negative, Internal):	Rising Edge
Trigger Stop (Positive, Negative, Internal):	Rising Edge
Sensor Bias (V):	120 V
Sensor Polarity (Positive, Negative):	Positive
Temperature (C):	Board Temp 52.389328 Deg C
Humidity (%):	Board Humidity 0.103516 
Medipix Clock (MHz):	120MHz
Readout System:	Merlin Quad
Software Version:	0.77.0.16
End	                                                                                                                                                                                                         
";
        let inp_bytes = inp.as_bytes();
        let header = QdAcquisitionHeader::parse_bytes(inp_bytes).unwrap();
        assert_eq!(header.num_frames(), 65536);
        assert_eq!(header.get_scan_x(), None);
        assert_eq!(header.get_scan_y(), None);
    }
}
