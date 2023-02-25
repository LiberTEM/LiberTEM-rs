use std::fmt::Debug;

use egui::ColorImage;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize, Debug, Clone)]
pub struct AcquisitionStarted {
    pub id: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AcquisitionEnded {
    pub id: String,
}

#[derive(Deserialize, Debug, Clone)]
pub enum ResultDType {
    #[serde(rename = "float32")]
    F32, // :shrug:
}

#[derive(Deserialize, Debug, Clone)]
pub enum ResultEncoding {
    #[serde(rename = "bslz4")]
    BSLZ4,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ChannelDeltaResult {
    pub bbox: (u16, u16, u16, u16),
    pub full_shape: (u16, u16),
    pub delta_shape: (u16, u16),
    pub dtype: ResultDType,
    pub encoding: ResultEncoding,
    pub channel_name: String,
    pub udf_name: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AcquisitionResult {
    pub id: String,
    pub channels: Vec<ChannelDeltaResult>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UpdateParamsInner {
    pub cx: f32,
    pub cy: f32,
    pub ri: f32,
    pub ro: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UpdateParams {
    pub parameters: UpdateParamsInner,
}

#[derive(Clone, Debug)]
pub struct BBox {
    pub ymin: u16,
    pub ymax: u16,
    pub xmin: u16,
    pub xmax: u16,
}

#[derive(Clone, Debug, Default)]
pub struct ProcessingStats {
    pub num_pending: usize,
    pub finished: usize,
}

#[derive(Clone)]
pub enum AcqMessage {
    AcquisitionStarted(AcquisitionStarted),
    AcquisitionEnded(String),
    Stats(ProcessingStats),
    UpdatedData {
        id: String,
        udf_name: String,
        channel_name: String,
        img: ColorImage,
        data: Array2<f32>,
        bbox: BBox,
    },
}

#[derive(Clone)]
pub enum MessagePart {
    Empty,
    AcquisitionStarted(AcquisitionStarted),
    AcquisitionEnded(AcquisitionEnded),
    AcquisitionResultHeader(AcquisitionResult),
    UpdateParams(UpdateParamsInner),
    AcquisitionBinaryPart(Vec<u8>),
}

impl Debug for MessagePart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => {
                f.debug_tuple("Empty").finish()
            }
            Self::AcquisitionStarted(arg0) => {
                f.debug_tuple("AcquisitionStarted").field(arg0).finish()
            }
            Self::AcquisitionEnded(arg0) => f.debug_tuple("AcquisitionEnded").field(arg0).finish(),
            Self::AcquisitionResultHeader(arg0) => f
                .debug_tuple("AcquisitionResultHeader")
                .field(arg0)
                .finish(),
            Self::AcquisitionBinaryPart(_arg0) => f.write_str("AcquisitionBinaryPart { .. } "),
            Self::UpdateParams(inner) => f.debug_tuple("UpdateParams").field(inner).finish(),
        }
    }
}

impl Debug for AcqMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AcquisitionStarted(arg0) => {
                f.debug_tuple("AcquisitionStarted").field(arg0).finish()
            }
            Self::AcquisitionEnded(arg0) => f.debug_tuple("AcquisitionEnded").field(arg0).finish(),
            Self::UpdatedData {
                id,
                bbox,
                img: _,
                udf_name,
                channel_name,
                data: _,
            } => f
                .debug_struct("UpdatedData")
                .field("id", id)
                .field("bbox", bbox)
                .field("udf_name", udf_name)
                .field("channel_name", channel_name)
                .finish(),
            Self::Stats(s) => f.debug_tuple("Stats").field(s).finish(),
        }
    }
}

#[derive(Debug)]
pub enum CommError {
    /// An unexpected message was received which we don't know how to handle in this state
    UnknownMessageError,

    ///
    UnknownEventError { event: String },

    /// An error occured while deserializing json data
    SerializationError(serde_json::Error),

    /// The JSON contents were not as expected
    FormatError,

    /// Some websocket error happened while trying to receive or send messages
    WebsocketError(tungstenite::Error),

    /// There are currently no messages available on the socket
    NoMessagesAvailable,

    /// The websocket connection has closed
    Close,
}

impl From<tungstenite::Error> for CommError {
    fn from(err: tungstenite::Error) -> Self {
        CommError::WebsocketError(err)
    }
}

impl From<serde_json::Error> for CommError {
    fn from(err: serde_json::Error) -> Self {
        CommError::SerializationError(err)
    }
}

impl MessagePart {
    pub fn from_binary(data: Vec<u8>) -> Self  {
        Self::AcquisitionBinaryPart(data)
    }

    pub fn from_text(message: String) -> Result<Self, CommError> {
        let obj: Value = serde_json::from_str(&message)?;
        let event = obj
            .as_object()
            .and_then(|obj| obj.get("event"))
            .and_then(|event| event.as_str())
            .ok_or(CommError::FormatError)?;

        match event {
            "ACQUISITION_STARTED" => Ok(Self::AcquisitionStarted(serde_json::from_str(&message)?)),
            "ACQUISITION_ENDED" => Ok(Self::AcquisitionEnded(serde_json::from_str(&message)?)),
            "RESULT" => Ok(Self::AcquisitionResultHeader(serde_json::from_str(
                &message,
            )?)),
            "UPDATE_PARAMS" => {
                let params: UpdateParams = serde_json::from_str(&message)?;
                Ok(Self::UpdateParams(params.parameters))
            }
            event => Err(CommError::UnknownEventError { event: event.to_string() } ),
        }
    }
}

/// Messages sent from the UI thread to the background processing thread,
/// ultimately being sent over the websocket connection:
#[derive(Clone, Debug)]
pub enum ControlMessage {
    UpdateParams { params: UpdateParams }, // yuck: nesting...
}
