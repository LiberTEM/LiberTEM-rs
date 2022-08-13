# TODO

- [ ] sim: support for different trigger modes
- [ ] general: error handling improvements
    - [ ] check where unwrap/expect can be avoided
    - [ ] introduce proper err handling and propagate to python or re-start things
- [ ] release GIL for long-running functions
    - [ ] `get_next_stack` is a good candidate to check
    - [ ] check the rest using giltracer
- [ ] check for cancellation in rust loops using `Python::check_signals`
    - [ ] for sim, must release GIL, as otherwise the main thread that gets the signal doens't get a chance to actually run...
- [ ] look into multiversion for perf critical parts (avx, avx2, avx512, sse4?)
