/// Because `K2Frame` is quite complicated and can't be easily made object-safe,
/// we instead have a separate `ResultFrame` trait that is a supertrait of
/// `K2Frame`, which is object-safe but still enough for consumers to work with.
pub trait ResultFrame {}
