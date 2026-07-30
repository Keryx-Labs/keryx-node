fn main() {
    let proto_root = std::path::Path::new("../proto");
    let messages = proto_root.join("messages.proto");
    let rpc = proto_root.join("rpc.proto");

    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&[messages.as_path()], &[proto_root])
        .unwrap_or_else(|e| panic!("protobuf compile error: {e}"));

    println!("cargo:rerun-if-changed={}", messages.display());
    println!("cargo:rerun-if-changed={}", rpc.display());
    // Ensure Windows .exe icon resources rebuild when icon assets change.
    println!("cargo:rerun-if-changed=icons/icon.ico");
    println!("cargo:rerun-if-changed=icons/icon.png");
    println!("cargo:rerun-if-changed=icons/32x32.png");
    println!("cargo:rerun-if-changed=icons/128x128.png");
    println!("cargo:rerun-if-changed=icons/128x128@2x.png");
    println!("cargo:rerun-if-changed=tauri.conf.json");
    println!("cargo:rerun-if-changed=windows/app.manifest");

    let mut windows = tauri_build::WindowsAttributes::new();
    windows = windows.app_manifest(include_str!("windows/app.manifest"));
    let attrs = tauri_build::Attributes::new().windows_attributes(windows);
    tauri_build::try_build(attrs).expect("failed to run tauri build script");
}
