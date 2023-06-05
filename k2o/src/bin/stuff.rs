use k2o::helpers::preallocate;

fn main() {
    preallocate("/tmp/stuff", 1024, 7, k2o::helpers::AllocateMode::ZeroFill);
}

trait Thing {
    fn do_thingy_things(self);
}

trait Builder {
    fn build_thing(how_deep: u32) -> Box<dyn Thing>;
}

struct Builder1 {}

struct Thing1 {
    how_deep: u32,
}

impl Thing for Thing1 {
    fn do_thingy_things(self) {
        println!("I'm doing thingy things here, look at me!");
    }
}

impl Builder for Builder1 {
    fn build_thing(how_deep: u32) -> Box<dyn Thing> {
        Box::new(Thing1 { how_deep })
    }
}
