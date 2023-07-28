use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crossbeam_channel::unbounded;
use k2o::{
    block::{BlockRouteInfo, K2Block, K2ISBlock},
    events::{
        AcquisitionParams, AcquisitionSize, AcquisitionSync, ChannelEventBus, EventBus, EventMsg,
        Events,
    },
    frame::{K2Frame, K2ISFrame},
    recv::recv_decode_loop,
};

fn mean(data: &[u128]) -> Option<f32> {
    let sum = data.iter().sum::<u128>() as f32;
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f32),
        _ => None,
    }
}

fn std_deviation(data: &[u128]) -> Option<f32> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data
                .iter()
                .map(|value| {
                    let diff = data_mean - (*value as f32);

                    diff * diff
                })
                .sum::<f32>()
                / count as f32;

            Some(variance.sqrt())
        }
        _ => None,
    }
}

fn start_threads<
    const PACKET_SIZE: usize, // FIXME: use B::PACKET_SIZE here
    F: K2Frame,
    B: K2Block,
>(
    events: &Events,
) {
    let ids = 0..=7u8;

    let (recycle_blocks_tx, recycle_blocks_rx) = unbounded::<B>();

    crossbeam::scope(|s| {
        let (decoded_tx, decoded_rx) = unbounded::<(B, BlockRouteInfo)>();
        for sector_id in ids {
            let port: u32 = 2001 + (sector_id as u32);
            let tx = decoded_tx.clone();
            let recycle_clone_rx = recycle_blocks_rx.clone();
            let recycle_clone_tx = recycle_blocks_tx.clone();
            let events_rx = events.subscribe();
            let local_addr = if sector_id < 4 {
                "192.168.10.99".to_string()
            } else {
                "192.168.10.98".to_string()
            };

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

        // "warmup"
        for _ in 0..1024 {
            recycle_blocks_tx.send(B::empty(0)).unwrap();
        }

        events.send(&EventMsg::ArmSectors {
            params: AcquisitionParams {
                size: AcquisitionSize::Continuous,
                sync: AcquisitionSync::Immediately,
                binning: k2o::events::Binning::Bin1x,
            },
        });

        let mut stats: HashMap<u32, Vec<Instant>, _> = HashMap::new();

        while let Ok((block, _)) = decoded_rx.recv_timeout(Duration::from_secs(60)) {
            let ts = block.get_decoded_timestamp();
            let frame_id = block.get_frame_id();
            stats
                .entry(frame_id)
                .and_modify(|timestamps| timestamps.push(ts))
                .or_insert_with(|| vec![ts]);
            recycle_blocks_tx.send(block).unwrap();
        }

        events.send(&EventMsg::Shutdown);

        // reference timestamp: the first one
        let ref_ts = stats
            .values()
            .map(|timestamps| {
                timestamps
                    .iter()
                    .min()
                    .expect("timestamps are not empty, by construction")
            })
            .min()
            .expect("stats map should not be empty");

        // could visualize with a box plot. but really just want a bunch of
        // numbers to quantify this! average of standard deviations (given equal sample size):
        // sqrt((s_1^2 + s_2^2 + ... + s_k^2) / k)

        let frame_times: Vec<f32> = stats
            .values()
            .map(|timestamps| {
                let min_instant = timestamps.iter().min().unwrap();
                let max_instant = timestamps.iter().max().unwrap();
                max_instant.duration_since(*min_instant).as_secs_f32()
            })
            .collect();

        println!("frame times: {:?}", frame_times);
        println!(
            "max frame time: {}",
            frame_times
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        );
        println!(
            "min frame time: {}",
            frame_times
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        );

        let stds: Vec<f32> = stats
            .values()
            .map(|timestamps| {
                let durations = timestamps.iter().map(|ts| ts.duration_since(*ref_ts));
                let micros: Vec<u128> = durations.map(|dur| dur.as_micros()).collect();

                std_deviation(&micros[..]).expect("micros are not empty, by construction")
            })
            .collect();

        let sum_stds: f32 = stds.iter().map(|s| s * s).sum();
        let mean_std = (sum_stds / stds.len() as f32).sqrt();
        println!("mean of std is {mean_std}");
    })
    .unwrap();
}

pub fn main() {
    let events: Events = ChannelEventBus::new();

    start_threads::<{ K2ISBlock::PACKET_SIZE }, K2ISFrame, K2ISBlock>(&events);
}
