#![cfg(test)]

use std::{
    env::temp_dir,
    ops::Deref,
    path::{Path, PathBuf},
};

use rand::{distr::Alphanumeric, Rng};
pub struct TempDir {
    pub path: PathBuf,
}

impl TempDir {
    pub fn new(prefix: &str) -> Self {
        let tmp_base = temp_dir();
        loop {
            let rnd: String = rand::rng()
                .sample_iter(&Alphanumeric)
                .take(15)
                .map(char::from)
                .collect();
            let rand_name = tmp_base.join(format!("{}-tmp{}", prefix, rnd));
            if !rand_name.exists() {
                std::fs::create_dir(&rand_name).unwrap();
                return Self { path: rand_name };
            }
        }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.path).unwrap();
    }
}

impl Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}
