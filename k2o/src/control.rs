use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::TryRecvError;
use log::debug;
use opentelemetry::trace::{TraceContextExt, Tracer};

use crate::{
    events::{AcquisitionParams, EventBus, EventMsg, Events, MessagePump},
    tracing::get_tracer,
};

///
/// Global acquisition state
///
#[derive(PartialEq, Eq, Debug, Clone, Default)]
pub enum AcquisitionState {
    #[default]
    Startup,
    Idle,
    Armed {
        params: AcquisitionParams,
        /// The current acquisition id
        acquisition_id: usize,
    },
    /// Current receiving data
    AcquisitionStarted {
        params: AcquisitionParams,
        /// The reference frame id for this acquisition
        frame_id: u32,
        /// The current acquisition id
        acquisition_id: usize,
    },
    /// No longer receiving new data, but still processing
    /// (=> there is still data in the queues)
    AcquisitionFinishing {
        params: AcquisitionParams,
        /// The reference frame id for this acquisition
        frame_id: u32,
        /// The current acquisition id
        acquisition_id: usize,
    },
    Shutdown,
}

///
/// Message pump thread connecting the "outer" world to the event bus.
///
/// It will run until it receives `EventMsg::Shutdown` or one of the channels is
/// disconnected.
///
pub fn control_loop(events: &Events, pump: &Option<MessagePump>) {
    let tracer = get_tracer();

    let events_rx = events.subscribe();

    tracer.in_span("control_loop", |cx| {
        let _span = cx.span();

        loop {
            if let Some(pump) = pump.as_ref() {
                pump.do_pump_timeout(events, Duration::from_millis(100))
                    .expect("message pump disconnected");
            }
            match events_rx.try_recv() {
                Ok(EventMsg::Shutdown) => break,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break,
                Ok(_) => continue,
            }
        }
    });
}

#[derive(Debug)]
pub enum StateError {
    InvalidTransition {
        from: AcquisitionState,
        msg: EventMsg,
    },
}

pub struct StateTracker {
    pub state: AcquisitionState,
}

/// Keep track of the acquisition state, changed by incoming events
impl StateTracker {
    pub fn new() -> Self {
        StateTracker {
            state: AcquisitionState::Startup,
        }
    }

    pub fn next_state(&self, event: &EventMsg) -> Result<AcquisitionState, StateError> {
        match &self.state {
            AcquisitionState::Startup => {
                // only Init is a valid transition here:
                match event {
                    EventMsg::Init => Ok(AcquisitionState::Idle),
                    _ => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                }
            }
            AcquisitionState::Shutdown => {
                // can't handle any messages when shut down (should also not happen...)
                Err(StateError::InvalidTransition {
                    from: self.state.clone(),
                    msg: event.clone(),
                })
            }
            AcquisitionState::Idle => {
                match event {
                    // invalid transitions:
                    EventMsg::Init => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionStartedSector {
                        sector_id: _,
                        frame_id: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionStarted {
                        frame_id: _,
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::ArmSectors {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionEnded { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::CancelAcquisition { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::ProcessingDone { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in AcquisitionFinishing state

                    // valid transitions:
                    EventMsg::Arm {
                        params,
                        acquisition_id,
                    } => Ok(AcquisitionState::Armed {
                        params: params.clone(),
                        acquisition_id: *acquisition_id,
                    }),
                    EventMsg::Shutdown => Ok(AcquisitionState::Shutdown),
                    EventMsg::AcquisitionError { msg: _ } => Ok(AcquisitionState::Shutdown),
                }
            }
            AcquisitionState::Armed {
                params,
                acquisition_id: _,
            } => {
                match event {
                    // invalid transitions:
                    EventMsg::Init => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::Arm {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::ProcessingDone { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in AcquisitionFinishing state

                    // valid transitions to self:
                    EventMsg::ArmSectors {
                        params,
                        acquisition_id,
                    } => Ok(AcquisitionState::Armed {
                        params: params.clone(),
                        acquisition_id: *acquisition_id,
                    }),
                    EventMsg::AcquisitionStartedSector {
                        sector_id: _,
                        frame_id: _,
                        acquisition_id,
                    } => Ok(AcquisitionState::Armed {
                        params: params.clone(),
                        acquisition_id: *acquisition_id,
                    }),

                    // valid transitions:
                    EventMsg::AcquisitionStarted {
                        frame_id,
                        params,
                        acquisition_id,
                    } => Ok(AcquisitionState::AcquisitionStarted {
                        params: params.clone(),
                        frame_id: *frame_id,
                        acquisition_id: *acquisition_id,
                    }),
                    EventMsg::AcquisitionEnded { acquisition_id: _ } => Ok(AcquisitionState::Idle),
                    EventMsg::CancelAcquisition { acquisition_id: _ } => Ok(AcquisitionState::Idle),
                    EventMsg::Shutdown => Ok(AcquisitionState::Shutdown),
                    EventMsg::AcquisitionError { msg: _ } => Ok(AcquisitionState::Shutdown),
                }
            }
            AcquisitionState::AcquisitionStarted {
                params,
                frame_id,
                acquisition_id,
            } => {
                match event {
                    // invalid transitions:
                    EventMsg::Init => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in Idle state
                    EventMsg::AcquisitionStarted {
                        frame_id: _,
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in Armed state
                    EventMsg::Arm {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in Idle state
                    EventMsg::ArmSectors {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in Armed state
                    EventMsg::ProcessingDone { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }), // should only come in AcquisitionFinishing state

                    // valid transitions:
                    EventMsg::AcquisitionStartedSector {
                        sector_id: _,
                        frame_id: _,
                        acquisition_id: _,
                    } => Ok(AcquisitionState::AcquisitionStarted {
                        params: params.clone(),
                        frame_id: *frame_id,
                        acquisition_id: *acquisition_id,
                    }),
                    EventMsg::AcquisitionEnded { acquisition_id } => {
                        Ok(AcquisitionState::AcquisitionFinishing {
                            params: params.clone(),
                            frame_id: *frame_id,
                            acquisition_id: *acquisition_id,
                        })
                    }
                    EventMsg::CancelAcquisition { acquisition_id: _ } => Ok(AcquisitionState::Idle),
                    EventMsg::Shutdown => Ok(AcquisitionState::Shutdown),
                    EventMsg::AcquisitionError { msg: _ } => Ok(AcquisitionState::Shutdown),
                }
            }
            AcquisitionState::AcquisitionFinishing {
                params: _,
                frame_id: _,
                acquisition_id: _acquisition_id_outer,
            } => {
                match event {
                    // invalid transitions:
                    EventMsg::Init => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionStartedSector {
                        sector_id: _,
                        frame_id: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionStarted {
                        frame_id: _,
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::Arm {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::ArmSectors {
                        params: _,
                        acquisition_id: _,
                    } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),
                    EventMsg::AcquisitionEnded { .. } => Err(StateError::InvalidTransition {
                        from: self.state.clone(),
                        msg: event.clone(),
                    }),

                    // valid transitions:
                    EventMsg::CancelAcquisition { acquisition_id: _ } => Ok(AcquisitionState::Idle),
                    EventMsg::Shutdown => Ok(AcquisitionState::Shutdown),
                    EventMsg::ProcessingDone { acquisition_id: _ } => Ok(AcquisitionState::Idle),
                    EventMsg::AcquisitionError { msg: _ } => Ok(AcquisitionState::Shutdown),
                }
            }
        }
    }

    pub fn set_state_from_msg(&mut self, msg: &EventMsg) -> Result<AcquisitionState, StateError> {
        let next_state = self.next_state(msg)?;
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        if next_state != self.state {
            debug!(
                "StateTracker:\n   old={:?}\n   event={:?}\n   -> {:?}\n   ts={:?}",
                self.state, msg, next_state, ts,
            );
        }
        self.state = next_state;
        Ok(self.state.clone())
    }
}

impl Default for StateTracker {
    fn default() -> Self {
        Self::new()
    }
}
