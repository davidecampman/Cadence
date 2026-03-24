mod sidecar;

use sidecar::PythonSidecar;
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

/// State shared across the Tauri app.
struct AppState {
    sidecar: Mutex<PythonSidecar>,
}

/// Tauri command: get the backend URL for the frontend to connect to.
#[tauri::command]
fn backend_url() -> String {
    "http://localhost:8000".to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let resource_dir = app
                .path()
                .resource_dir()
                .expect("failed to resolve resource dir");

            let sidecar = PythonSidecar::new(resource_dir);

            app.manage(AppState {
                sidecar: Mutex::new(sidecar),
            });

            // Start the Python backend
            let state = app.state::<AppState>();
            let mut sc = state.sidecar.lock().unwrap();
            if let Err(e) = sc.start() {
                log::error!("Failed to start Python sidecar: {}", e);
                // Don't panic — the app can still show an error state in the UI
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_url])
        .build(tauri::generate_context!())
        .expect("error building Sentinel desktop app")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                // Gracefully stop the Python backend on exit
                let state = app.state::<AppState>();
                let mut sc = state.sidecar.lock().unwrap();
                sc.stop();
            }
        });
}
