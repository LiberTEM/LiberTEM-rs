use std::ops::Deref;

use common::frame_stack::FrameStackHandle;
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
    types::{PyBytes, PyType},
};
use serde::{Deserialize, Serialize};

use crate::common::{DectrisFrameMeta, PixelType};
