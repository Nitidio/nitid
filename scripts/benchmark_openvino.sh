#!/usr/bin/env bash
# Check an Intel machine, install what the OpenVINO benchmark needs, download COCO val2017 and
# run tools/benchmark_openvino.py on every OpenVINO device it finds.
#
# Run it from a clone of the repository:
#
#     scripts/benchmark_openvino.sh                 # full run: latency, then accuracy
#     scripts/benchmark_openvino.sh --latency-only  # skip the COCO mAP pass
#
# Nothing here uses sudo. Missing drivers or a non-performance power profile are reported with the
# commands to fix them; the benchmark then runs on the devices that are available.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/datasets/coco"
OUTPUT_DIR="$REPO_DIR/runs/benchmark/openvino-$(hostname)-$(date +%Y%m%d-%H%M)"
MODELS="model1s model1m model1l model1x"
LATENCY_ONLY=0
ASSUME_YES=0
MIN_FREE_GB=20

usage() {
    cat <<EOF
Usage: $0 [options]

  --models "A B ..."  models to benchmark (default: $MODELS)
  --data-dir DIR      COCO root; val2017/ and annotations/ are downloaded here if missing
                      (default: $DATA_DIR)
  --output DIR        results directory (default: runs/benchmark/openvino-<host>-<time>)
  --latency-only      skip the COCO mAP pass
  --yes               do not stop to ask when a check fails
  -h, --help          show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --latency-only) LATENCY_ONLY=1; shift ;;
        --yes) ASSUME_YES=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

WARNINGS=()
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok() { printf '    \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33mwarn\033[0m %s\n' "$*"; WARNINGS+=("$*"); }
fail() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

confirm() {
    [[ $ASSUME_YES -eq 1 ]] && return 0
    read -r -p "    $1 [y/N] " answer
    [[ "$answer" =~ ^[YySs]$ ]]
}

pkg_version() { dpkg-query -W -f '${Version}' "$1" 2>/dev/null || true; }

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
exec > >(tee -a "$OUTPUT_DIR/run.log") 2>&1
cd "$REPO_DIR"

# ── 1. Hardware ───────────────────────────────────────────────────────────────
info "Hardware"
[[ "$(uname -s)" == "Linux" ]] || fail "this script supports Linux only"
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
ok "CPU: $CPU_MODEL ($(nproc) threads, $(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo) RAM)"
[[ "$CPU_MODEL" == *"Intel"* ]] || warn "not an Intel CPU; only the CPU device can be benchmarked"
if lspci -nn 2>/dev/null | grep -qiE 'VGA.*8086|Display.*8086'; then
    ok "Intel GPU: $(lspci -nn | grep -iE 'VGA.*8086|Display.*8086' | head -1 | cut -d' ' -f2-)"
else
    warn "no Intel GPU found by lspci"
fi
if ls /dev/accel/accel* >/dev/null 2>&1; then
    ok "NPU kernel device: $(echo /dev/accel/accel*)"
else
    warn "no /dev/accel device: the NPU is missing or its kernel driver (intel_vpu) is not loaded"
fi
if ls /dev/dri/renderD* >/dev/null 2>&1 && ! id -nG | grep -qw render; then
    warn "user $(id -un) is not in the 'render' group; fix: sudo usermod -aG render $(id -un) (then log in again)"
fi

# ── 2. Drivers ────────────────────────────────────────────────────────────────
info "Drivers"
if command -v dpkg-query >/dev/null; then
    gpu_icd="$(pkg_version intel-opencl-icd)"
    gpu_ze="$(pkg_version libze-intel-gpu1)"
    [[ -n "$gpu_ze" ]] || gpu_ze="$(pkg_version intel-level-zero-gpu)"
    if [[ -n "$gpu_icd" && -n "$gpu_ze" ]]; then
        ok "GPU compute runtime: intel-opencl-icd $gpu_icd, level-zero GPU $gpu_ze"
    else
        warn "GPU compute runtime incomplete (intel-opencl-icd='$gpu_icd', level-zero GPU='$gpu_ze');" \
            "install it from https://github.com/intel/compute-runtime/releases"
    fi
    npu_ze="$(pkg_version intel-level-zero-npu)"
    if [[ -n "$npu_ze" ]]; then
        ok "NPU user-space driver: intel-level-zero-npu $npu_ze, compiler $(pkg_version intel-driver-compiler-npu)"
    else
        warn "NPU user-space driver not installed (intel-level-zero-npu); OpenVINO will not see the NPU." \
            "Install it from https://github.com/intel/linux-npu-driver/releases"
    fi
else
    warn "dpkg not available; driver versions are not checked"
fi

# ── 3. Power and load ─────────────────────────────────────────────────────────
info "Power and load"
profile="$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
if [[ "$profile" == "performance" || "$profile" == "unknown" ]]; then
    ok "platform profile: $profile"
else
    warn "platform profile is '$profile'; fix: echo performance | sudo tee /sys/firmware/acpi/platform_profile"
fi
if [[ "$governor" == "performance" || "$governor" == "unknown" ]]; then
    ok "CPU governor: $governor"
else
    warn "CPU governor is '$governor'; fix: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
fi
for supply in /sys/class/power_supply/*; do
    if [[ "$(cat "$supply/type" 2>/dev/null)" == "Battery" ]]; then
        status="$(cat "$supply/status" 2>/dev/null || echo unknown)"
        if [[ "$status" == "Discharging" ]]; then
            warn "running on battery; plug in the charger"
        else
            ok "battery: $status"
        fi
    fi
done
if [[ -r /sys/class/powercap/intel-rapl:0/energy_uj ]]; then
    ok "RAPL package energy is readable: power will be reported"
else
    warn "RAPL is not readable, so power will not be reported; fix: sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj"
fi
load="$(cut -d' ' -f1 /proc/loadavg)"
if awk "BEGIN {exit !($load < 2.0)}"; then
    ok "1-minute load average: $load"
else
    warn "the machine is busy (1-minute load $load); other workloads distort the results"
    ps -eo pid,user,%cpu,comm --sort=-%cpu | head -6 | sed 's/^/        /'
    confirm "Continue anyway?" || fail "stopped: wait until the machine is idle"
fi

# ── 4. Tools and disk ─────────────────────────────────────────────────────────
info "Tools and disk"
for tool in git python3; do
    command -v "$tool" >/dev/null || fail "'$tool' is required; install it with: sudo apt install $tool"
done
ok "git, python3 $(python3 -c 'import platform; print(platform.python_version())')"
mkdir -p "$DATA_DIR"
free_gb="$(df -Pk "$REPO_DIR" | awk 'NR==2 {printf "%d", $4/1048576}')"
if (( free_gb < MIN_FREE_GB )); then
    warn "only ${free_gb} GB free; the environment, COCO and the exported models need about ${MIN_FREE_GB} GB"
    confirm "Continue anyway?" || fail "stopped: free some disk space"
else
    ok "${free_gb} GB free"
fi
if command -v uv >/dev/null; then
    UV="$(command -v uv)"
else
    UV="$REPO_DIR/.venv-tools/bin/uv"
    if [[ ! -x "$UV" ]]; then
        info "Installing uv into .venv-tools"
        python3 -m venv "$REPO_DIR/.venv-tools" || fail "python3 -m venv failed; install it with: sudo apt install python3-venv"
        "$REPO_DIR/.venv-tools/bin/pip" install --quiet uv
    fi
fi
ok "$("$UV" --version)"

# ── 5. Python environment ─────────────────────────────────────────────────────
run_py() { "$UV" run --extra openvino --extra train python "$@"; }
info "Python environment (uv sync --inexact --extra openvino --extra train)"
# --inexact keeps packages from other extras: removing the 'track' extra's opencv-python would
# delete the cv2 module that opencv-python-headless also provides.
"$UV" sync --inexact --extra openvino --extra train
run_py -c 'import cv2' 2>/dev/null ||
    fail "cv2 does not import; repair it with: $UV pip install --reinstall opencv-python-headless"

DEVICES="$(run_py -c 'import openvino as ov; print(" ".join(ov.Core().available_devices))')"
OV_VERSION="$(run_py -c 'import openvino as ov; print(ov.get_version())')"
ok "OpenVINO $OV_VERSION sees: $DEVICES"
MATRIX="CPU:FP32"
for device in GPU NPU; do
    if [[ " $DEVICES " == *" $device "* ]]; then
        MATRIX+=" $device:FP16"
    else
        warn "OpenVINO does not see a $device; it is left out"
    fi
done

# ── 6. COCO val2017 ───────────────────────────────────────────────────────────
info "COCO val2017 in $DATA_DIR"
download() {  # $1 = URL, $2 = directory or file that marks it as extracted
    local url="$1" marker="$2" zip
    zip="$DATA_DIR/$(basename "$url")"
    if [[ -e "$DATA_DIR/$marker" ]]; then
        ok "$marker already present"
        return
    fi
    echo "    downloading $url (this can take a while)"
    if command -v curl >/dev/null; then
        curl -fsSL --retry 3 -C - -o "$zip" "$url"
    elif command -v wget >/dev/null; then
        wget -q -c -O "$zip" "$url"
    else
        python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
            "$url" "$zip"
    fi
    # unzip is often missing on minimal installs; Python's zipfile does the same job.
    python3 -m zipfile -e "$zip" "$DATA_DIR"
    rm -f "$zip"
}
download http://images.cocodataset.org/zips/val2017.zip val2017
download http://images.cocodataset.org/annotations/annotations_trainval2017.zip annotations/instances_val2017.json
images="$(find "$DATA_DIR/val2017" -name '*.jpg' | wc -l)"
[[ "$images" -eq 5000 ]] || fail "expected 5000 images in $DATA_DIR/val2017, found $images; delete it and rerun"
ok "5000 images and instances_val2017.json"

# ── 7. Benchmark ──────────────────────────────────────────────────────────────
echo
if (( ${#WARNINGS[@]} )); then
    info "Warnings (also in $OUTPUT_DIR/run.log):"
    printf '    - %s\n' "${WARNINGS[@]}"
    confirm "Run the benchmark with these warnings?" || fail "stopped before benchmarking"
fi
printf '%s\n' "${WARNINGS[@]+"${WARNINGS[@]}"}" > "$OUTPUT_DIR/warnings.txt"

info "Phase 1: latency on $MATRIX (models: $MODELS)"
# shellcheck disable=SC2086  # MODELS and MATRIX are word lists
run_py -m tools.benchmark_openvino --coco "$DATA_DIR" --models $MODELS --matrix $MATRIX \
    --skip-val --output "$OUTPUT_DIR/latency" || true
[[ -s "$OUTPUT_DIR/latency/results.json" ]] ||
    fail "phase 1 produced no results; see the traceback above and $OUTPUT_DIR/run.log"
grep -q '"status": "failed"' "$OUTPUT_DIR/latency/results.json" &&
    warn "phase 1: some combinations failed; see the Failures section of latency/results.md"

if [[ $LATENCY_ONLY -eq 0 ]]; then
    info "Phase 2: COCO mAP on $MATRIX (reuses the exported models)"
    mkdir -p "$OUTPUT_DIR/accuracy"
    ln -sfn ../latency/ir "$OUTPUT_DIR/accuracy/ir"
    # shellcheck disable=SC2086
    run_py -m tools.benchmark_openvino --coco "$DATA_DIR" --models $MODELS --matrix $MATRIX \
        --runs 200 --warmup 200 --output "$OUTPUT_DIR/accuracy" || true
    [[ -s "$OUTPUT_DIR/accuracy/results.json" ]] ||
        fail "phase 2 produced no results; see the traceback above and $OUTPUT_DIR/run.log"
fi

# ── 8. Package the results ────────────────────────────────────────────────────
ARCHIVE="$OUTPUT_DIR.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$OUTPUT_DIR")" --exclude='*/ir' --exclude='*.bin' \
    --exclude='*.xml' "$(basename "$OUTPUT_DIR")"
echo
info "Done. Summary: $OUTPUT_DIR/latency/results.md"
[[ $LATENCY_ONLY -eq 0 ]] && info "Accuracy:      $OUTPUT_DIR/accuracy/results.md"
info "Send this file back: $ARCHIVE"
