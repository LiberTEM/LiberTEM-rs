use std::net::TcpStream;

use egui::{TextureHandle, ColorImage, plot::Polygon, Vec2};
use log::error;
use ndarray::Array2;
use serde::Deserialize;
use serde_json::Value;
use websocket::{sync::Client, OwnedMessage, Message};

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
pub struct AcquisitionResult {
    pub bbox: (u16, u16, u16, u16),
    pub shape: (u16, u16),
    pub delta_shape: (u16, u16),
    pub dtype: ResultDType,
    pub encoding: ResultEncoding,
    pub id: String,
}

#[derive(Clone, Debug)]
pub struct BBox 
{
    pub ymin: u16,
    pub ymax: u16,
    pub xmin: u16,
    pub xmax: u16,
}

#[derive(Clone)]
pub enum AcqMessage {
    AcquisitionStarted(AcquisitionStarted),
    AcquisitionEnded(AcquisitionEnded),
    AcquisitionResult(AcquisitionResult, Vec<u8>),
    UpdatedData(String, ColorImage, BBox),
}

impl core::fmt::Debug for AcqMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AcquisitionStarted(arg0) => {
                f.debug_tuple("AcquisitionStarted").field(arg0).finish()
            }
            Self::AcquisitionEnded(arg0) => f.debug_tuple("AcquisitionEnded").field(arg0).finish(),
            Self::AcquisitionResult(..) => f.write_str("AcquisitionResult { .. }"),
            Self::UpdatedData(arg0, arg1, arg2) => f.debug_tuple("UpdatedData").field(arg0).field(arg2).finish(),
        }
    }
}

fn get_text_msg(client: &mut Client<TcpStream>) -> Option<String> {
    let message = client.recv_message();
    let message = &match message {
        Ok(websocket::OwnedMessage::Text(msg)) => msg,
        Ok(websocket::OwnedMessage::Ping(msg)) => {
            client.send_message(&Message::pong(msg)).unwrap();
            return None;
        },
        Ok(m) => {
            error!("can't handle message here: {m:?}");
            return None;
        }
        Err(e) => {
            error!("error receiving message: {e:?}");
            return None;
        }
    };

    Some(message.to_string())
}

fn get_binary_msg(client: &mut Client<TcpStream>) -> Option<Vec<u8>> {
    let message = client.recv_message();
    let message = &match message {
        Ok(websocket::OwnedMessage::Binary(msg)) => msg,
        Ok(websocket::OwnedMessage::Ping(msg)) => {
            client.send_message(&Message::pong(msg)).unwrap();
            return None;
        },
        Ok(m) => {
            error!("can't handle message here: {m:?}");
            return None;
        }
        Err(e) => {
            error!("error receiving message: {e:?}");
            return None;
        }
    };

    Some(message.to_owned())
}

impl AcqMessage {
    pub fn from_socket(client: &mut Client<TcpStream>) -> Option<Self> {
        let message = &get_text_msg(client)?;
        let obj: Value = match serde_json::from_str(message) {
            Ok(obj) => obj,
            Err(_e) => return None,
        };
        let obj = obj.as_object()?;
        let event = obj.get("event")?;
        let event = event.as_str()?;

        match event {
            "ACQUISITION_STARTED" => {
                let msg: Result<AcquisitionStarted, _> = serde_json::from_str(message);
                match msg {
                    Ok(msg) => Some(Self::AcquisitionStarted(msg)),
                    Err(err) => {
                        error!("deserialization failed: {:?}", err);
                        None
                    }
                }
            }
            "ACQUISITION_ENDED" => {
                let msg: Result<AcquisitionEnded, _> = serde_json::from_str(message);
                match msg {
                    Ok(msg) => Some(Self::AcquisitionEnded(msg)),
                    Err(err) => {
                        error!("deserialization failed: {:?}", err);
                        None
                    }
                }
            }
            "RESULT" => {
                let msg: Result<AcquisitionResult, _> = serde_json::from_str(message);
                match msg {
                    Ok(msg) => {
                        // receive compound binary message:
                        let data = get_binary_msg(client)?;
                        Some(Self::AcquisitionResult(msg, data))
                    }
                    Err(err) => {
                        error!("deserialization failed: {:?}", err);
                        None
                    }
                }
            }
            _ => {
                error!("unknown event type: {}", event);
                None
            }
        }
    }
}
