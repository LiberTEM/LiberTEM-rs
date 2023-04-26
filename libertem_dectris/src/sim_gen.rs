use bs_sys::compress_lz4;
use serde_json::json;

use crate::bin_fmt::{write_raw_msg_fh, write_serializable_fh};

/// Generate mock data in our binary zmq dump format
pub fn make_sim_data(num_frames: usize) -> Vec<u8> {
    let mut out = Vec::new();

    let first_header = json!({
        "header_detail": "basic",
        "htype": "dheader-1.0",
        "series": 34,
    });
    write_serializable_fh(&first_header, &mut out);

    // FIXME: can't have all the fields, it will reach some recursion limit
    let detector_config = json!({
        "auto_summation": true,
        "beam_center_x": 0.0,
        "beam_center_y": 0.0,
        "bit_depth_image": 16,
        "bit_depth_readout": 16,
        "chi_increment": 0.0,
        "chi_start": 0.0,
        "compression": "bslz4",
        "count_time": 0.0009999,
        "countrate_correction_applied": true,
        "countrate_correction_count_cutoff": 3825,
        "data_collection_date": "2022-05-10T18:20:50.027+02:00",
        "description": "Dectris QUADRO Si 250K",
        "detector_distance": 0.0,
        "detector_number": "E-01-0350",
        "detector_readout_time": 1e-7,
        "detector_translation": [
            0.0,
            0.0,
            0.0
        ],
        "eiger_fw_version": "release-2020.2.2",
        "element": "",
        "flatfield_correction_applied": true,
        "frame_count_time": 36400.0,
        "frame_period": 36400.0,
        "frame_time": 0.001,
        "kappa_increment": 0.0,
        "kappa_start": 0.0,
        "nimages": 1,
        "ntrigger": num_frames,
        "number_of_excluded_pixels": 0,
        // "omega_increment": 0.0,
        // "omega_start": 0.0,
        // "phi_increment": 0.0,
        // "phi_start": 0.0,
        // "photon_energy": 8041.0,
        "pixel_mask_applied": true,
        "roi_mode": "",
        "sensor_material": "Si",
        "sensor_thickness": 0.00045,
        "software_version": "1.8.0",
        "threshold_energy": 4020.5,
        "trigger_mode": "exte",
        // "two_theta_increment": 0.0,
        // "two_theta_start": 0.0,
        // "virtual_pixel_correction_applied": true,
        "wavelength": 1.5419002416764116,
        "x_pixel_size": 0.000075,
        "x_pixels_in_detector": 512,
        "y_pixel_size": 0.000075,
        "y_pixels_in_detector": 512
    });
    write_serializable_fh(&detector_config, &mut out);

    for frame_idx in 0..num_frames {
        let pixel_data: Vec<u16> = (0..512 * 512)
            .map(|_| (frame_idx % u16::MAX as usize) as u16)
            .collect();
        let compressed_data = compress_lz4(&pixel_data, None).unwrap();
        let compressed_data: Vec<u8> = (0..12)
            .map(|_| 0u8)
            .chain(compressed_data.into_iter())
            .collect();

        let size = compressed_data.len();
        let hash = format!("{:?}", md5::compute(&compressed_data));

        let dimage = json!({
            "frame": frame_idx,
            "hash": hash,
            "htype": "dimage-1.0",
            "series": 34
        });
        write_serializable_fh(&dimage, &mut out);

        let dimaged = json!({
          "encoding": "bs16-lz4<",
          "htype": "dimage_d-1.0",
          "shape": [
            512,
            512
          ],
          "size": size,
          "type": "uint16"
        });
        write_serializable_fh(&dimaged, &mut out);

        write_raw_msg_fh(&compressed_data, &mut out);

        let footer = json!({
            "htype": "dconfig-1.0",
            "real_time": 500000,  // FIXME: adjust timestamps?
            "start_time": 5396052835800u64,
            "stop_time": 5396053335799u64,
        });
        write_serializable_fh(&footer, &mut out);
    }

    // acquisition footer
    let footer = json!({
        "htype": "deseries_end-1.0",
        "series": 34,
    });
    write_serializable_fh(&footer, &mut out);

    out
}
