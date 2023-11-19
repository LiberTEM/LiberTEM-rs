extern crate crossbeam;
extern crate crossbeam_channel;
extern crate jemallocator;

use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use clap::Parser;
use crossbeam_channel::{unbounded, RecvTimeoutError};
use ipc_test::SharedSlabAllocator;
use k2o::acquisition::{acquisition_loop, AcquisitionResult};
use k2o::assemble::{assembler_main, AssemblyResult};
use k2o::block::BlockRouteInfo;
use k2o::block::K2Block;
use k2o::block_is::K2ISBlock;
use k2o::block_summit::K2SummitBlock;
use k2o::cli_args::{Args, Mode};
use k2o::control::control_loop;
use k2o::events::{
    AcquisitionParams, AcquisitionSize, AcquisitionSync, Events, MessagePump, WriterSettings,
    WriterType,
};
use k2o::events::{ChannelEventBus, EventBus, EventMsg};
use k2o::frame::K2Frame;
use k2o::frame_is::K2ISFrame;
use k2o::frame_summit::K2SummitFrame;
use k2o::helpers::CPU_AFF_WRITER;
use k2o::helpers::{recv_and_get_init, set_cpu_affinity};
use k2o::recv::recv_decode_loop;
use k2o::tracing::init_tracer;
use log::info;
use tokio::runtime::Runtime;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn start_threads<
    const PACKET_SIZE: usize, // FIXME: use B::PACKET_SIZE instead
    F: K2Frame,
    B: K2Block,
>(
    args: &Args,
    events: &Events,
    pump: &Option<MessagePump>,
) {
    let ids = 0..=7u8;

    let (recycle_blocks_tx, recycle_blocks_rx) = unbounded::<B>();

    // make sure this is created before the other threads are started, so we don't
    // miss any events!
    let shm_path = Path::new(&args.shm_path);
    let slot_size: usize = F::get_size_bytes();

    info!("Initializing shared memory...");
    let shm = SharedSlabAllocator::new(1000, slot_size, true, shm_path).expect("create shm");
    info!("Shared memory initialized.");

    crossbeam::scope(|s| {
        let (assembly_tx, assembly_rx) = unbounded::<(B, BlockRouteInfo)>();
        for sector_id in ids {
            let port: u32 = 2001 + (sector_id as u32);
            let tx = assembly_tx.clone();
            let recycle_clone_rx = recycle_blocks_rx.clone();
            let recycle_clone_tx = recycle_blocks_tx.clone();
            let events_rx = events.subscribe();
            let local_addr = "192.168.10.99".to_string();

            s.builder()
                .name(format!("recv-decode-{}", sector_id))
                .spawn(move |_| {
                    recv_decode_loop::<B, PACKET_SIZE>(
                        sector_id,
                        port,
                        &tx,
                        &recycle_clone_rx,
                        &recycle_clone_tx,
                        &events_rx,
                        events,
                        local_addr,
                    );
                })
                .expect("could not spawn recv+decode thread");
        }

        let (full_frames_tx, full_frames_rx) = unbounded::<AssemblyResult<F>>();
        let (recycle_frames_tx, recycle_frames_rx) = unbounded();

        let asm_events_rx = events.subscribe();
        let shm_handle = shm.get_handle().os_handle;

        // assembly main thread:
        s.builder()
            .name("assembly".to_string())
            .spawn({
                let shm_handle = shm_handle.clone();
                move |_| {
                    //set_cpu_affinity(CPU_AFF_ASSEMBLY);
                    //frame_assembler(&assembly_rx, &full_frames_tx);
                    let asm_shm = SharedSlabAllocator::connect(&shm_handle).unwrap();

                    assembler_main(
                        &assembly_rx,
                        &full_frames_tx,
                        &recycle_blocks_tx,
                        asm_events_rx,
                        asm_shm,
                        &Duration::from_millis(100),
                    );
                }
            })
            .expect("could not spawn assembly thread");

        // writer thread(s):
        let w1rx = full_frames_rx;
        let writer_events_rx = events.subscribe();
        s.builder()
            .name("ff_writer".to_string())
            .spawn({
                let shm_handle = shm_handle.clone();
                move |_| {
                    set_cpu_affinity(CPU_AFF_WRITER);

                    let writer_shm = SharedSlabAllocator::connect(&shm_handle).unwrap();

                    acquisition_loop(
                        &w1rx,
                        &recycle_frames_tx,
                        &writer_events_rx,
                        events,
                        writer_shm,
                    );
                }
            })
            .expect("could not spawn ff_writer thread");

        s.builder()
            .name("retire_thread".to_string())
            .spawn({
                // let shm_handle = shm_handle.clone();
                move |_| {
                    let mut shm = SharedSlabAllocator::connect(&shm_handle).unwrap();
                    loop {
                        match recycle_frames_rx.recv_timeout(Duration::from_millis(100)) {
                            Ok(AcquisitionResult::Frame(frame, _))
                            | Ok(AcquisitionResult::DroppedFrame(frame, _))
                            | Ok(AcquisitionResult::DroppedFrameOutside(frame)) => {
                                frame.free_payload(&mut shm)
                            }
                            Ok(AcquisitionResult::DoneAborted { .. })
                            | Ok(AcquisitionResult::DoneSuccess { .. })
                            | Ok(AcquisitionResult::ShutdownIdle)
                            | Ok(AcquisitionResult::DoneShuttingDown { .. }) => {
                                events.send(&EventMsg::Shutdown);
                                info!("retire thread closing");
                                break;
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                continue;
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                info!("retire thread closing");
                                break;
                            }
                        }
                    }
                }
            })
            .unwrap();

        events.send(&EventMsg::Init {});

        let method: WriterType = args.write_mode.into();
        let writer_settings = WriterSettings::Enabled {
            method,
            filename: args.write_to.to_owned(),
        };
        let writer_settings = WriterSettings::Disabled;

        events.send(&EventMsg::Arm {
            params: AcquisitionParams {
                size: AcquisitionSize::NumFrames(1800),
                // size: AcquisitionSize::Continuous,
                //sync: AcquisitionSync::WaitForSync,
                sync: AcquisitionSync::Immediately,
                binning: k2o::events::Binning::Bin1x,
                writer_settings,
            },
            acquisition_id: 0,
        });

        control_loop(events, pump)
    })
    .unwrap();
}

pub fn main() {
    let args = Args::parse();
    let thread_builder = std::thread::Builder::new();

    let env = env_logger::Env::default()
        .filter_or("LIBERTEM_K2IS_LOG_LEVEL", "info")
        .write_style_or("LIBERTEM_K2IS_LOG_STYLE", "always");
    env_logger::init_from_env(env);

    // for waiting until tracing is initialized:
    let barrier = Arc::new(Barrier::new(2));
    let barrier_bg = Arc::clone(&barrier);
    thread_builder
        .name("tracing".to_string())
        .spawn(move || {
            let rt = Runtime::new().unwrap();

            rt.block_on(async {
                init_tracer().unwrap();
                barrier_bg.wait();

                // do we need to keep this thread alive like this? I think so!
                // otherwise we get:
                // OpenTelemetry trace error occurred. cannot send span to the batch span processor because the channel is closed
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            });
        })
        .unwrap();

    let mode = match args.mode {
        None => {
            info!("auto-initializing mode...");
            let packet_size = recv_and_get_init() as usize;
            match packet_size {
                K2ISBlock::PACKET_SIZE => Mode::IS,
                K2SummitBlock::PACKET_SIZE => Mode::Summit,
                _ => panic!("unknown packet size: {}", packet_size),
            }
        }
        Some(mode) => mode,
    };

    info!("writing to {}", args.write_to);
    let events: Events = ChannelEventBus::new();
    let pump = MessagePump::new(&events);

    match mode {
        Mode::IS => {
            info!("IS mode...");
            start_threads::<{ K2ISBlock::PACKET_SIZE }, K2ISFrame, K2ISBlock>(
                &args,
                &events,
                &Some(pump),
            );
        }
        Mode::Summit => {
            info!("Summit mode...");
            start_threads::<{ K2SummitBlock::PACKET_SIZE }, K2SummitFrame, K2SummitBlock>(
                &args,
                &events,
                &Some(pump),
            );
        }
    }
}
