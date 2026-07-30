# Keryx Node Launcher

Windows Tauri control panel for installing and running `keryxd`, styled after [keryx-labs.com](https://keryx-labs.com/).

Uses **Tauri 2 + TypeScript** (WebView2): Rust backend for process management and gRPC sync monitoring, TS frontend for the UI.

## Requirements

- Node.js 20+
- Rust toolchain (same as building `keryxd`)
- WebView2 (preinstalled on Windows 10/11)
- A built `keryxd` binary (`cargo build --release` in the repo root), or point **Binary source** at an existing exe
- `protoc` on `PATH` (gRPC codegen for the sync monitor)

## Develop

```bash
cd launcher
npm install
npm start
```

`npm start` runs `tauri dev` (Vite on `:1420` + Rust host).

## Package

```bash
# optional: copy release binary for the installer bundle
mkdir -p bundled
cp ../target/release/keryxd.exe bundled/

npm run dist
```

Output: `launcher/src-tauri/target/release/bundle/nsis/` — Windows **NSIS** setup (`.exe`).

Uses NSIS Modern UI (Windows 10/11 style). Defaults to `C:\Program Files\Keryx Node Launcher` (`perMachine`) with a directory page to change the path. Administrator elevation is required.

The WiX `.msi` path is not used: its folder browser is locked to the legacy Windows Installer UI and cannot be modernized.

The launcher is **portable / self-contained**: settings live in `keryx-launcher.json` next to the exe (not `%APPDATA%`). First-run defaults put the node binary under `{install}/bin` and chain data under `{install}/data`. A new install folder starts fresh; uninstall removes the JSON.

Tauri downloads NSIS automatically on first Windows build.

## Features

- First-run onboarding wizard (download latest GitHub release or local binary, network, sync mode, SSD/HDD + RAM/cache/rate suggestions from this PC)
- Choose install directory (binary + optional DLLs under `install/bin`, data under `--appdir`)
- Optional `datadir.zip` extract via PowerShell `Expand-Archive` (handles large archives; network-aware destination)
- Trusted peers, RocksDB preset/cache/WAL/rate limit, RAM scale
- Network / RPC / logging flags mirroring common `keryxd` CLI options
- Live command preview, stdout/stderr log pane, start/stop
- Sync progress via local gRPC (`tonic`) + IBD log parsing

## Layout

- `src/` — Vite/TS UI (`api.ts` → Tauri `invoke` / events)
- `src-tauri/` — Rust host (install/spawn, config store, sync monitor)
- `proto/` — gRPC protos shared with the node wire format
- `bundled/` — optional `keryxd` binary shipped inside the installer
