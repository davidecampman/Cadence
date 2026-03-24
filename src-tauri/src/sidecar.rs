use std::path::PathBuf;
use std::process::{Child, Command};

/// Manages the Python backend process lifecycle.
pub struct PythonSidecar {
    resource_dir: PathBuf,
    process: Option<Child>,
}

impl PythonSidecar {
    pub fn new(resource_dir: PathBuf) -> Self {
        Self {
            resource_dir,
            process: None,
        }
    }

    /// Resolve the Python executable path.
    ///
    /// In production builds, we bundle a standalone Python runtime under
    /// `resources/python-runtime/`. In development, we fall back to the
    /// system Python.
    fn python_exe(&self) -> PathBuf {
        let bundled = if cfg!(target_os = "windows") {
            self.resource_dir.join("python-runtime").join("python.exe")
        } else {
            self.resource_dir
                .join("python-runtime")
                .join("bin")
                .join("python3")
        };

        if bundled.exists() {
            bundled
        } else {
            // Dev fallback: use system Python
            PathBuf::from(if cfg!(target_os = "windows") {
                "python"
            } else {
                "python3"
            })
        }
    }

    /// Start the sentinel-server process.
    pub fn start(&mut self) -> Result<(), String> {
        if self.process.is_some() {
            return Ok(()); // Already running
        }

        let python = self.python_exe();
        log::info!("Starting Python sidecar: {:?}", python);

        let child = Command::new(&python)
            .args([
                "-m",
                "sentinel.server",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ])
            .env("SENTINEL_DESKTOP", "1")
            .spawn()
            .map_err(|e| format!("Failed to spawn Python backend: {}", e))?;

        log::info!("Python sidecar started with PID {}", child.id());
        self.process = Some(child);
        Ok(())
    }

    /// Gracefully stop the Python backend.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.process.take() {
            log::info!("Stopping Python sidecar (PID {})", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for PythonSidecar {
    fn drop(&mut self) {
        self.stop();
    }
}
