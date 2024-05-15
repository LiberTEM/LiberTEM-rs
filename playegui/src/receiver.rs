use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use crossbeam::channel::{Receiver, Sender, TryRecvError};
use futures_util::{stream::SplitSink, Sink, SinkExt, StreamExt};
use log::{error, info, warn};
use tokio_tungstenite::connect_async;
use tungstenite::Message;
use url::Url;

use crate::{
    app::ConnectionStatus,
    messages::{CommError, ControlMessage, MessagePart, UpdateParams, UpdateParamsInner},
};

fn message_part_from_msg(
    msg: Result<Message, tungstenite::Error>,
) -> Result<MessagePart, CommError> {
    let msg = msg?;
    match msg {
        Message::Text(msg) => Ok(MessagePart::from_text(msg)?),
        Message::Binary(msg) => Ok(MessagePart::from_binary(msg)),
        Message::Ping(_) => Ok(MessagePart::Empty),
        _ => Err(CommError::UnknownMessageError),
        // Message::Ping(_) => todo!(),
        // Message::Pong(_) => todo!(),
        // Message::Close(_) => todo!(),
        // Message::Frame(_) => todo!(),
    }
}

fn set_status(status: &mut Arc<Mutex<ConnectionStatus>>, new_status: ConnectionStatus) {
    let mut s = status.lock().unwrap();
    *s = new_status;
}

async fn send_parameter_update<S>(
    ws_write: &mut SplitSink<S, Message>,
    params: &UpdateParams,
) -> Result<(), tungstenite::Error>
where
    S: Sink<Message>,
    tungstenite::Error: From<<S as Sink<Message>>::Error>,
{
    let msg = serde_json::to_string(params).unwrap();
    ws_write.send(Message::Text(msg)).await?;
    Ok(())
}

pub(crate) fn receiver_thread(
    control_channel: Receiver<ControlMessage>,
    result_channel: Sender<MessagePart>,
    stop_event: Arc<AtomicBool>,
    mut status: Arc<Mutex<ConnectionStatus>>,
) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut last_params: Option<UpdateParams> = None;

    rt.block_on(async {
        'outer: loop {
            info!("Connecting...");
            let (socket, _) = match connect_async(Url::parse("ws://localhost:8444").unwrap()).await
            {
                Ok(s) => s,
                Err(e) => {
                    error!("Could not connect: {e}; trying to reconnect...");
                    set_status(&mut status, ConnectionStatus::Error);
                    if stop_event.load(Ordering::Relaxed) {
                        break 'outer;
                    }
                    std::thread::sleep(Duration::from_millis(1000));
                    continue;
                }
            };
            info!("Connected.");
            set_status(&mut status, ConnectionStatus::Connected);

            let (mut ws_write, mut ws_read) = socket.split();

            // If we have remembered parameters from the UI thread, send them
            // again, in case the server has restarted:
            if let Some(params) = &last_params {
                match send_parameter_update(&mut ws_write, params).await {
                    Ok(_) => {}
                    Err(err) => {
                        error!("Could not set parameters ({err:?}), reconnecting...");
                        continue;
                    }
                }
            }

            'inner: loop {
                if stop_event.load(Ordering::Relaxed) {
                    break 'outer;
                }
                let msg = ws_read.next().await;
                let msg = match msg {
                    Some(msg) => match message_part_from_msg(msg) {
                        Ok(msg) => msg,
                        Err(err) => {
                            error!("Error receiving message: {err:?}, reconnecting...");
                            break 'inner;
                        }
                    },
                    None => {
                        warn!("Disconnected, trying to reconnect...");
                        break 'inner;
                    }
                };
                let send_res = result_channel.send(msg);
                if send_res.is_err() {
                    if stop_event.load(Ordering::Relaxed) {
                        break 'outer;
                    } else {
                        panic!("Other side of the channel is dead, cannot continue");
                    }
                }

                match control_channel.try_recv() {
                    Ok(ControlMessage::UpdateParams { params }) => {
                        match send_parameter_update(&mut ws_write, &params).await {
                            Ok(_) => {}
                            Err(err) => {
                                error!("Could not set parameters ({err:?}), reconnecting...");
                                break 'inner;
                            }
                        }
                        last_params = Some(params);
                    }
                    Err(TryRecvError::Disconnected) => break 'outer,
                    Err(TryRecvError::Empty) => {}
                }
            }
        }
    });

    info!("stopped receiver thread...");
}
