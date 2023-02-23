use std::{fmt::Debug, net::TcpStream};

use egui::ColorImage;
use serde::Deserialize;
use serde_json::Value;
use websocket::{sync::Client, Message, OwnedMessage, WebSocketError};

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
pub struct ChannelResult {
    pub bbox: (u16, u16, u16, u16),
    pub shape: (u16, u16),
    pub delta_shape: (u16, u16),
    pub dtype: ResultDType,
    pub encoding: ResultEncoding,
    pub name: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AcquisitionResult {
    pub id: String,
    pub channels: Vec<ChannelResult>,
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
    AcquisitionEnded(AcquisitionEnded),
    Stats(ProcessingStats),
    UpdatedData {
        id: String,
        img: ColorImage,
        bbox: BBox,
    },
}

#[derive(Clone)]
pub enum MessagePart {
    AcquisitionStarted(AcquisitionStarted),
    AcquisitionEnded(AcquisitionEnded),
    AcquisitionResultHeader(AcquisitionResult),
    AcquisitionBinaryPart(Vec<u8>),
}

impl Debug for MessagePart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AcquisitionStarted(arg0) => {
                f.debug_tuple("AcquisitionStarted").field(arg0).finish()
            }
            Self::AcquisitionEnded(arg0) => f.debug_tuple("AcquisitionEnded").field(arg0).finish(),
            Self::AcquisitionResultHeader(arg0) => f
                .debug_tuple("AcquisitionResultHeader")
                .field(arg0)
                .finish(),
            Self::AcquisitionBinaryPart(_arg0) => f.write_str("AcquisitionBinaryPart { .. } "),
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
            Self::UpdatedData { id, bbox, img: _ } => f
                .debug_struct("UpdatedData")
                .field("id", id)
                .field("bbox", bbox)
                .finish(),
            Self::Stats(s) => f.debug_tuple("Stats").field(s).finish(),
        }
    }
}

#[derive(Debug)]
pub enum CommError {
    /// An unexpected message was received which we don't know how to handle in this state
    UnknownMessageError(Option<websocket::OwnedMessage>),

    /// An error occured while deserializing json data
    SerializationError(serde_json::Error),

    /// The JSON contents were not as expected
    FormatError,

    /// Some websocket error happened while trying to receive or send messages
    WebsocketError(WebSocketError),

    /// There are currently no messages available on the socket
    NoMessagesAvailable,

    /// The websocket connection has closed
    Close,
}

impl From<WebSocketError> for CommError {
    fn from(err: WebSocketError) -> Self {
        match err {
            WebSocketError::NoDataAvailable => CommError::NoMessagesAvailable,
            other => CommError::WebsocketError(other),
        }
    }
}

impl From<serde_json::Error> for CommError {
    fn from(err: serde_json::Error) -> Self {
        CommError::SerializationError(err)
    }
}

/// Get a message and already handle ping/pong websocket stuff
fn get_message(client: &mut Client<TcpStream>) -> Result<OwnedMessage, CommError> {
    let message = client.recv_message()?;

    if let websocket::OwnedMessage::Ping(msg) = &message {
        client.send_message(&Message::pong(&msg[..]))?;
        Err(CommError::NoMessagesAvailable)
    } else {
        Ok(message)
    }
}

impl MessagePart {
    pub fn from_socket(client: &mut Client<TcpStream>) -> Result<Self, CommError> {
        let ws_message = get_message(client)?;
        let message = match ws_message {
            OwnedMessage::Text(msg) => msg,
            OwnedMessage::Binary(msg) => return Ok(Self::AcquisitionBinaryPart(msg)),
            OwnedMessage::Close(_) => return Err(CommError::Close),
            OwnedMessage::Ping(_) | OwnedMessage::Pong(_) => {
                return Err(CommError::UnknownMessageError(Some(ws_message)))
            }
        };

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
            _ => Err(CommError::UnknownMessageError(None)),
        }
    }
}
