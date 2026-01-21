use serde::{Deserialize, Serialize, de::DeserializeOwned};
use url::Url;

#[derive(thiserror::Error, Debug)]
pub enum ServalError {
    #[error("request failed: {msg}")]
    RequestFailed { msg: String },

    #[error("serialization error: {msg}")]
    SerializationError { msg: String },

    #[error("URL error: {msg}")]
    URLError { msg: String },
}

impl From<url::ParseError> for ServalError {
    fn from(value: url::ParseError) -> Self {
        Self::URLError {
            msg: value.to_string(),
        }
    }
}

impl From<serde_json::Error> for ServalError {
    fn from(value: serde_json::Error) -> Self {
        Self::SerializationError {
            msg: value.to_string(),
        }
    }
}

impl From<reqwest::Error> for ServalError {
    fn from(value: reqwest::Error) -> Self {
        Self::RequestFailed {
            msg: value.to_string(),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Serialize, Deserialize, Debug)]
pub enum TriggerMode {
    /// Start: Positive Edge External Trigger Input, Stop: Negative Edge
    #[serde(rename = "PEXSTART_NEXSTOP")]
    PexStartNexStop,

    /// Start: Negative Edge External Trigger Input, Stop: Positive Edge
    #[serde(rename = "NEXSTART_PEXSTOP")]
    NexStartPexStop,

    /// Start: Positive Edge External Trigger Input, Stop: HW timer
    #[serde(rename = "PEXSTART_TIMERSTOP")]
    PexStartTimerStop,

    /// Start: Negative Edge External Trigger Input, Stop: HW timer
    #[serde(rename = "NEXSTART_TIMERSTOP")]
    NexStartTimerStop,

    #[serde(rename = "AUTOTRIGSTART_TIMERSTOP")]
    AutoTriggerStartTimerStop,

    #[serde(rename = "CONTINUOUS")]
    Continuous,

    #[serde(rename = "SOFTWARESTART_TIMERSTOP")]
    SoftwareStartTimerStop,

    #[serde(rename = "SOFTWARESTART_SOFTWARESTOP")]
    SoftwareStartSoftwareStop,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorConfig {
    #[serde(rename = "BiasVoltage")]
    pub bias_voltage: u64,

    #[serde(rename = "BiasEnabled")]
    pub bias_enabled: bool,

    #[serde(rename = "nTriggers")]
    pub n_triggers: u64,

    /// Exposure time in seconds
    #[serde(rename = "ExposureTime")]
    pub exposure_time: f32,

    /// Trigger period in seconds
    #[serde(rename = "TriggerPeriod")]
    pub trigger_period: f32,

    #[serde(rename = "TriggerMode")]
    pub trigger_mode: TriggerMode,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorBoard {
    #[serde(rename = "ChipboardId")]
    pub chipboard_id: String,

    #[serde(rename = "IpAddress")]
    pub ip_address: String,

    #[serde(rename = "FirmwareVersion")]
    pub firmare_version: String,

    #[serde(rename = "Chips")]
    pub chips: Vec<DetectorChip>,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorChip {
    #[serde(rename = "Index")]
    pub index: usize,

    #[serde(rename = "Id")]
    pub id: u64,

    #[serde(rename = "Name")]
    pub name: String,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorInfo {
    #[serde(rename = "IfaceName")]
    pub iface_name: String,

    #[serde(rename = "SW_version")]
    pub sw_version: String,

    #[serde(rename = "FW_version")]
    pub fw_version: String,

    #[serde(rename = "PixCount")]
    pub pix_count: u64,

    #[serde(rename = "RowLen")]
    pub row_len: u16,

    #[serde(rename = "NumberOfChips")]
    pub number_of_chips: u16,

    #[serde(rename = "NumberOfRows")]
    pub number_of_rows: u16,

    #[serde(rename = "MpxType")]
    pub mpx_type: u64,

    #[serde(rename = "Boards")]
    pub boards: Vec<DetectorBoard>,

    #[serde(rename = "SuppAcqModes")]
    pub supp_acq_modes: u64,

    #[serde(rename = "ClockReadout")]
    pub clock_readout: f32,

    #[serde(rename = "MaxPulseCount")]
    pub max_pulse_count: u64,

    #[serde(rename = "MaxPulseHeight")]
    pub max_pulse_height: f32,

    #[serde(rename = "MaxPulsePeriod")]
    pub max_pulse_period: f32,

    #[serde(rename = "TimerMaxVal")]
    pub timer_max_val: f32,

    #[serde(rename = "TimerMinVal")]
    pub timer_min_val: f32,

    #[serde(rename = "TimerStep")]
    pub timer_step: f32,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub enum DetectorOrientation {
    #[serde(rename = "UP")]
    Up,

    #[serde(rename = "RIGHT")]
    Right,

    #[serde(rename = "DOWN")]
    Down,

    #[serde(rename = "LEFT")]
    Left,

    #[serde(rename = "UP_MIRRORED")]
    UpMirrored,

    #[serde(rename = "RIGHT_MIRRORED")]
    RightMirrored,

    #[serde(rename = "DOWN_MIRRORED")]
    DownMirrored,

    #[serde(rename = "LEFT_MIRRORED")]
    LeftMirrored,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub enum ChipOrientation {
    LtRBtT,
    RtLBtT,
    LtRTtB,
    RtLTtB,
    BtTLtR,
    TtBLtR,
    BtTRtL,
    TtBRtL,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorLayout {
    #[serde(rename = "Orientation")]
    orientation: DetectorOrientation,

    #[serde(rename = "Original")]
    original: DetectorLayoutInner,

    #[serde(rename = "Rotated")]
    rotated: DetectorLayoutInner,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorLayoutChip {
    #[serde(rename = "Chip")]
    chip: u16,

    #[serde(rename = "X")]
    x: u16,

    #[serde(rename = "Y")]
    y: u16,

    #[serde(rename = "Orientation")]
    orientation: ChipOrientation,
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct DetectorLayoutInner {
    #[serde(rename = "Width")]
    width: u16,

    #[serde(rename = "Height")]
    height: u16,

    #[serde(rename = "Chips")]
    chips: Vec<DetectorLayoutChip>,
}

pub struct ServalClient {
    base_url: Url,
}

impl ServalClient {
    pub fn new(base_url: &str) -> Self {
        ServalClient {
            base_url: Url::parse(base_url).unwrap(),
        }
    }

    fn get_request<T>(&self, path: &str) -> Result<T, ServalError>
    where
        T: DeserializeOwned,
    {
        let url = self.base_url.join(path)?;
        let resp = reqwest::blocking::get(url)?;
        let resp_text = resp.text()?;
        let config: T = serde_json::from_str(&resp_text)?;
        Ok(config)
    }

    pub fn get_detector_config(&self) -> Result<DetectorConfig, ServalError> {
        self.get_request("/detector/config/")
    }

    pub fn get_detector_info(&self) -> Result<DetectorInfo, ServalError> {
        self.get_request("/detector/info/")
    }

    pub fn get_detector_layout(&self) -> Result<DetectorLayout, ServalError> {
        self.get_request("/detector/layout/")
    }

    pub fn start_measurement(&self) {
        todo!();
    }
}

#[cfg(test)]
mod test {}
