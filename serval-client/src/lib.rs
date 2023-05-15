use serde::{Deserialize, Serialize};
use url::Url;

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

pub struct ServalClient {
    base_url: Url,
}

impl ServalClient {
    pub fn new(base_url: &str) -> Self {
        ServalClient {
            base_url: Url::parse(base_url).unwrap(),
        }
    }

    pub fn set_detector_config(&self) {
        todo!();
    }

    pub fn get_detector_config(&self) -> DetectorConfig {
        let url = self.base_url.join("/detector/config/").unwrap();
        println!("{url}");
        let resp = reqwest::blocking::get(url).unwrap();
        let resp_text = resp.text().unwrap();
        println!("{}", resp_text);
        let config: DetectorConfig = serde_json::from_str(&resp_text).unwrap();
        config
    }

    pub fn start_acquisition(&self) {
        todo!();
    }
}

#[cfg(test)]
mod test {
    use crate::ServalClient;

    #[test]
    fn test_stuff() {
        let client = ServalClient::new("http://localhost:8080");

        println!("{:?}", client.get_detector_config());
        panic!("at the disco?");
    }
}
