extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    //println!("cargo:rustc-link-search=/path/to/lib");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    //println!("cargo:rustc-link-lib=bz2");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=vendor/wrapper.h");

    let mut build = cc::Build::new();

    build
        .include("vendor/bitshuffle/lz4")
        .include("vendor/bitshuffle")
        .file("vendor/bitshuffle/src/bitshuffle.c")
        .file("vendor/bitshuffle/src/bitshuffle_core.c")
        .file("vendor/bitshuffle/src/iochain.c")
        .file("vendor/bitshuffle/lz4/lz4.c")
        // FIXME: extract from bitshuffle setup.py directly?
        .define("BSHUF_VERSION_MAJOR", "0")
        .define("BSHUF_VERSION_MINOR", "4")
        .define("BSHUF_VERSION_POINT", "2")
        // compiler flags stolen from setup.py:
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-std=c99")
        .flag_if_supported("-fno-strict-aliasing")
        .flag_if_supported("-fPIC")
        .flag_if_supported("/Ox")
        .flag_if_supported("/fp:fast")
        .flag_if_supported("-w");

    build.compile("bitshuffle");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut bindings_builder = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("vendor/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-Ivendor/bitshuffle/src/")
        .allowlist_function("bshuf_compress_lz4")
        .allowlist_function("bshuf_decompress_lz4")
        .allowlist_function("bshuf_compress_lz4_bound");

    if let Ok(extra_include_path) = env::var("BINDGEN_C_INCLUDE_PATH") {
        let arg = format!("-I{extra_include_path}");
        bindings_builder = bindings_builder.clang_arg(arg);
    }

    let bindings = bindings_builder
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
