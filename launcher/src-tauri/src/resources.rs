//! Detect machine RAM / disk and recommend node performance settings.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiskInfo {
    /// "ssd" | "hdd" | "unknown"
    pub media: String,
    pub drive: Option<String>,
    pub free_gb: Option<f64>,
    /// "high" | "low"
    pub confidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceRecommendation {
    /// "ssd" | "hdd"
    pub profile: String,
    pub rocksdb_preset: String,
    pub ram_scale: f64,
    pub rocksdb_cache_size: u64,
    pub rocksdb_rate_limit_mb: u64,
    pub budget_mb: u64,
    pub estimate_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineResources {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub disk: DiskInfo,
    pub recommendation: PerformanceRecommendation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendInput {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    /// "ssd" | "hdd"
    pub profile: String,
    pub drive: Option<String>,
}

const SCALE_STEPS: [f64; 8] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];

pub fn detect_machine_resources(path: &str) -> MachineResources {
    let (total_ram_mb, available_ram_mb) = detect_ram();
    let disk = detect_disk(path);
    let profile = if disk.media == "ssd" { "ssd" } else { "hdd" };
    let mut recommendation = recommend_performance(RecommendInput {
        total_ram_mb,
        available_ram_mb,
        profile: profile.into(),
        drive: disk.drive.clone(),
    });

    let mut warning = None;
    if total_ram_mb < 8_000 {
        warning = Some(
            "This PC may struggle; close other apps or use a machine with more RAM.".into(),
        );
    }
    if let Some(free) = disk.free_gb {
        if free < 50.0 {
            let msg = format!(
                "Low free disk space on {}: ~{free:.0} GB. Chain data needs tens of GB.",
                disk.drive.as_deref().unwrap_or("disk")
            );
            warning = Some(match warning {
                Some(w) => format!("{w} {msg}"),
                None => msg,
            });
        }
    }
    // Only fall back to HDD when media type could not be determined.
    if disk.media == "unknown" {
        recommendation.profile = "hdd".into();
        recommendation.rocksdb_preset = "hdd".into();
    }

    MachineResources {
        total_ram_mb,
        available_ram_mb,
        disk,
        recommendation,
        warning,
    }
}

pub fn recommend_performance(input: RecommendInput) -> PerformanceRecommendation {
    let profile = if input.profile.eq_ignore_ascii_case("ssd") {
        "ssd"
    } else {
        "hdd"
    };
    let is_ssd = profile == "ssd";
    let rocksdb_preset = if is_ssd { "default" } else { "hdd" };

    let mut budget_mb = ((input.available_ram_mb as f64 * 0.55)
        .min(input.total_ram_mb as f64 * 0.45)) as u64;
    budget_mb = budget_mb.max(3_500);
    // Don't claim more RAM than the machine has.
    if input.total_ram_mb > 0 {
        budget_mb = budget_mb.min(input.total_ram_mb.saturating_sub(2_048).max(2_048));
    }

    let mut chosen_scale = 0.25_f64;
    let mut chosen_cache = 64_u64;
    let mut chosen_estimate = estimate_process_mb(is_ssd, chosen_scale, chosen_cache);

    for &scale in &SCALE_STEPS {
        let cache = suggest_cache_mb(scale, budget_mb);
        let est = estimate_process_mb(is_ssd, scale, cache);
        if est <= budget_mb {
            chosen_scale = scale;
            chosen_cache = cache;
            chosen_estimate = est;
        } else {
            break;
        }
    }

    let rate = if is_ssd {
        0
    } else {
        suggest_rate_limit_mb(input.total_ram_mb, input.drive.as_deref())
    };

    PerformanceRecommendation {
        profile: profile.into(),
        rocksdb_preset: rocksdb_preset.into(),
        ram_scale: chosen_scale,
        rocksdb_cache_size: chosen_cache,
        rocksdb_rate_limit_mb: rate,
        budget_mb,
        estimate_mb: chosen_estimate,
    }
}

pub fn estimate_process_mb(is_ssd: bool, ram_scale: f64, cache_mb: u64) -> u64 {
    let memtable_mb = if is_ssd {
        ((256.0 * ram_scale) as u64).max(64)
    } else {
        ((768.0 * ram_scale) as u64).max(512)
    };
    let caches_mb = ((64.0 + 128.0 + 128.0 + 64.0) * ram_scale) as u64;
    let baseline_mb = 1_500_u64;
    cache_mb
        .saturating_add(memtable_mb)
        .saturating_add(caches_mb)
        .saturating_add(baseline_mb)
}

fn suggest_cache_mb(ram_scale: f64, budget_mb: u64) -> u64 {
    let raw = ((256.0 * ram_scale) as u64).max(64);
    let cap = ((budget_mb as f64) * 0.25) as u64;
    round256(raw.min(cap).max(64))
}

fn suggest_rate_limit_mb(total_ram_mb: u64, drive: Option<&str>) -> u64 {
    let system_drive = drive
        .map(|d| d.eq_ignore_ascii_case("C") || d.eq_ignore_ascii_case("C:"))
        .unwrap_or(false);
    if total_ram_mb < 16_000 || system_drive {
        24
    } else {
        48
    }
}

fn round256(v: u64) -> u64 {
    if v <= 256 {
        return 256;
    }
    ((v + 128) / 256) * 256
}

fn detect_ram() -> (u64, u64) {
    #[cfg(windows)]
    {
        let script = r#"
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("{0}|{1}" -f $os.TotalVisibleMemorySize, $os.FreePhysicalMemory)
"#;
        if let Ok(out) = run_powershell(script) {
            let line = out.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() == 2 {
                let total_kb: u64 = parts[0].parse().unwrap_or(0);
                let free_kb: u64 = parts[1].parse().unwrap_or(0);
                if total_kb > 0 {
                    return (total_kb / 1024, free_kb / 1024);
                }
            }
        }
    }
    // Fallback: assume 16 GB / 8 GB free
    (16_384, 8_192)
}

fn detect_disk(path: &str) -> DiskInfo {
    let drive = drive_letter(path);
    let free_gb = drive.as_ref().and_then(|d| free_space_gb(d));

    #[cfg(windows)]
    if let Some(ref letter) = drive {
        let letter_clean = letter.trim_end_matches(':');
        // Robust media detection:
        // 1) Partition → Disk → PhysicalDisk (common case)
        // 2) Volume AccessPath → Partition (when DriveLetter lookup fails)
        // 3) Closest PhysicalDisk by volume size (partial / broken partition maps)
        // Then classify via MediaType, BusType (NVMe), SpindleSpeed, model name.
        let script = format!(
            r#"
$ErrorActionPreference = 'SilentlyContinue'
$letter = '{letter_clean}'
$pd = $null
$label = ''
$pd = Get-Partition -DriveLetter $letter | Get-Disk | Get-PhysicalDisk | Select-Object -First 1
if (-not $pd) {{
  $vol = Get-Volume -DriveLetter $letter
  if ($vol) {{
    $label = [string]$vol.FileSystemLabel
    $part = Get-Partition | Where-Object {{
      $_.AccessPaths -and (
        $_.AccessPaths -contains $vol.Path -or
        $_.AccessPaths -contains ($letter + ':\')
      )
    }} | Select-Object -First 1
    if ($part) {{
      $pd = Get-Disk -Number $part.DiskNumber | Get-PhysicalDisk | Select-Object -First 1
    }}
  }}
}}
if (-not $pd) {{
  $vol = Get-Volume -DriveLetter $letter
  if ($vol) {{
    $label = [string]$vol.FileSystemLabel
    $vs = [int64]$vol.Size
    $cand = Get-PhysicalDisk | ForEach-Object {{
      $_ | Add-Member -NotePropertyName Diff -NotePropertyValue ([math]::Abs([int64]$_.Size - $vs)) -Force -PassThru
    }} | Sort-Object Diff | Select-Object -First 1
    if ($cand) {{
      $tol = [math]::Max(512MB, [int64]($vs * 0.01))
      if ($cand.Diff -le $tol) {{ $pd = $cand }}
    }}
  }}
}}
$media = 'unknown'
$confidence = 'low'
$hint = [string]$label
if ($pd) {{
  $hint = ($label + ' ' + [string]$pd.FriendlyName + ' ' + [string]$pd.Model).ToLowerInvariant()
  $mt = ([string]$pd.MediaType).ToLowerInvariant()
  $bus = ([string]$pd.BusType).ToLowerInvariant()
  $spin = -1
  try {{ $spin = [int64]$pd.SpindleSpeed }} catch {{}}
  if ($mt -eq 'ssd' -or $mt -eq 'scm') {{ $media = 'ssd'; $confidence = 'high' }}
  elseif ($mt -eq 'hdd') {{ $media = 'hdd'; $confidence = 'high' }}
  elseif ($bus -eq 'nvme') {{ $media = 'ssd'; $confidence = 'high' }}
  elseif ($hint -match 'nvme|\bssd\b') {{ $media = 'ssd'; $confidence = 'medium' }}
  elseif ($hint -match '\bhdd\b|barracuda|seagate|wdc\s*wd|wd\s*red|wd\s*blue|wd\s*black') {{ $media = 'hdd'; $confidence = 'medium' }}
  elseif ($spin -eq 0 -and $bus -ne 'usb' -and $bus -ne 'fileback') {{ $media = 'ssd'; $confidence = 'medium' }}
  elseif ($spin -gt 0) {{ $media = 'hdd'; $confidence = 'medium' }}
}} elseif ($hint.ToLowerInvariant() -match 'nvme|\bssd\b') {{
  $media = 'ssd'; $confidence = 'medium'
}} elseif ($hint.ToLowerInvariant() -match '\bhdd\b') {{
  $media = 'hdd'; $confidence = 'medium'
}}
Write-Output ('{{0}}|{{1}}' -f $media, $confidence)
"#
        );
        if let Ok(out) = run_powershell(&script) {
            let line = out
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty() && l.contains('|'))
                .unwrap_or("unknown|low");
            let mut parts = line.split('|');
            let media_raw = parts.next().unwrap_or("unknown").trim();
            let confidence_raw = parts.next().unwrap_or("low").trim();
            let media = match media_raw.to_ascii_lowercase().as_str() {
                "ssd" => "ssd",
                "hdd" => "hdd",
                _ => "unknown",
            };
            let confidence = match confidence_raw.to_ascii_lowercase().as_str() {
                "high" => "high",
                "medium" => "medium",
                _ => "low",
            };
            return DiskInfo {
                media: media.into(),
                drive: Some(format!("{letter_clean}:")),
                free_gb,
                confidence: confidence.into(),
            };
        }
    }

    DiskInfo {
        media: "unknown".into(),
        drive: drive.map(|d| {
            if d.ends_with(':') {
                d
            } else {
                format!("{d}:")
            }
        }),
        free_gb,
        confidence: "low".into(),
    }
}

fn drive_letter(path: &str) -> Option<String> {
    let p = Path::new(path);
    // Windows: "F:\foo" or "F:/foo"
    let s = p.to_string_lossy();
    let bytes = s.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' {
        let letter = (bytes[0] as char).to_ascii_uppercase();
        if letter.is_ascii_alphabetic() {
            return Some(letter.to_string());
        }
    }
    None
}

fn free_space_gb(drive_letter: &str) -> Option<f64> {
    #[cfg(windows)]
    {
        let letter = drive_letter.trim_end_matches(':');
        let script = format!(
            r#"
try {{
  $d = Get-PSDrive -Name {letter} -ErrorAction Stop
  Write-Output $d.Free
}} catch {{
  Write-Output 0
}}
"#
        );
        if let Ok(out) = run_powershell(&script) {
            let line = out.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("0");
            if let Ok(bytes) = line.parse::<f64>() {
                if bytes > 0.0 {
                    return Some(bytes / (1024.0 * 1024.0 * 1024.0));
                }
            }
        }
    }
    let _ = drive_letter;
    None
}

#[cfg(windows)]
fn run_powershell(script: &str) -> Result<String, String> {
    let output = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .creation_flags(0x0800_0000)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!(
            "powershell failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(not(windows))]
fn run_powershell(_script: &str) -> Result<String, String> {
    Err("not windows".into())
}
