#!/usr/bin/env bash
# Iterate-only integration tests for the kbox CLI and Python API.
set -uo pipefail

KBOX=./build/tools/kbox
PY_RUN=(${KBOX_TEST_PYTHON:-$(command -v python3 2>/dev/null || echo python3)})
KBOX_ABS=$(cd "$(dirname "$KBOX")" && pwd)/$(basename "$KBOX")
TMPROOT=$(mktemp -d /tmp/kbox_iterate_XXXX)

cleanup() {
    jobs -pr | xargs -r kill 2>/dev/null || true
    rm -rf "$TMPROOT"
}
trap cleanup EXIT

echo "=== kbox Iterate Integration Tests ==="

if [ ! -x "$KBOX" ]; then
    echo "Building kbox..."
    make build/tools/kbox build/tools/kbox_worker_daemon || { echo "Build failed"; exit 1; }
fi

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"; shift
    printf "  %-58s " "$name"
    if output=$("$@" 2>&1); then
        echo "PASS"; PASS=$((PASS + 1)); return 0
    else
        rc=$?
        if [ $rc -eq 77 ]; then
            echo "SKIP"; SKIP=$((SKIP + 1))
        else
            echo "FAIL (rc=$rc)"
            echo "$output" | head -20
            FAIL=$((FAIL + 1))
        fi
        return 1
    fi
}

run_test_grep() {
    local name="$1" pattern="$2"; shift 2
    printf "  %-58s " "$name"
    if output=$("$@" 2>&1); then
        if echo "$output" | grep -q "$pattern"; then
            echo "PASS"; PASS=$((PASS + 1)); return 0
        fi
        echo "FAIL (pattern not found: $pattern)"
        echo "$output" | head -20
        FAIL=$((FAIL + 1))
        return 1
    fi

    rc=$?
    if echo "$output" | grep -q "$pattern"; then
        echo "PASS"; PASS=$((PASS + 1)); return 0
    fi
    if [ $rc -eq 77 ]; then
        echo "SKIP"; SKIP=$((SKIP + 1))
    else
        echo "FAIL (rc=$rc)"
        echo "$output" | head -20
        FAIL=$((FAIL + 1))
    fi
    return 1
}

run_test_expect_fail() {
    local name="$1"; shift
    printf "  %-58s " "$name"
    if output=$("$@" 2>&1); then
        echo "FAIL (expected failure)"
        FAIL=$((FAIL + 1))
        return 1
    fi
    echo "PASS"
    PASS=$((PASS + 1))
    return 0
}

wait_for_count() {
    local log="$1" pattern="$2" want="$3" pid="$4" timeout_s="${5:-20}"
    local deadline=$((SECONDS + timeout_s))
    while [ $SECONDS -lt $deadline ]; do
        if [ -f "$log" ]; then
            count=$(grep -c "$pattern" "$log" 2>/dev/null || true)
            if [ "${count:-0}" -ge "$want" ]; then
                return 0
            fi
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 0.1
    done
    return 1
}

watch_single_pt_reload() {
    local tdir="$TMPROOT/watch_single"
    mkdir -p "$tdir"
    cp examples/data/mlp.pt "$tdir/mlp.pt"
    cat > "$tdir/test_single_pt.py" <<EOF
import torch

def init_once():
    return {"h5": "$tdir/mlp.pt"}

def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
EOF
    local log="$tdir/watch.log"
    "$KBOX" iterate "$tdir/test_single_pt.py" >"$log" 2>&1 &
    local pid=$!
    wait_for_count "$log" "PASS: output\\[0\\]" 1 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    touch "$tdir/mlp.pt"
    wait_for_count "$log" "PASS: output\\[0\\]" 2 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

watch_suite_reload() {
    local tdir="$TMPROOT/watch_suite"
    mkdir -p "$tdir"
    cp -R examples/data/mlp_cases "$tdir/mlp_cases"
    cat > "$tdir/test_suite_pt.py" <<EOF
import torch

def init_once():
    return {"h5_suite": "$tdir/mlp_cases"}

def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
EOF
    local log="$tdir/watch.log"
    "$KBOX" iterate "$tdir/test_suite_pt.py" >"$log" 2>&1 &
    local pid=$!
    wait_for_count "$log" "Suite: 5/5 passed" 1 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    touch "$tdir/mlp_cases/small_batch.pt"
    wait_for_count "$log" "Suite: 5/5 passed" 2 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# CLI surface
run_test_grep "help: iterate-only command list" "iterate" \
    "$KBOX" --help
run_test_grep "help: mcp command listed" "mcp" \
    "$KBOX" --help
run_test_grep "version: prints kbox version" "kbox" \
    "$KBOX" version
run_test_grep "dispatch: iterate help works without PATH" "kbox iterate" \
    env -u PATH "$KBOX_ABS" iterate --help
run_test "shorthand: direct test file dispatch" \
    "$KBOX" examples/dev/test_scale.py --once
run_test_expect_fail "error: unknown command fails" \
    "$KBOX" nonexistent-command

# Full iterate suite except the intentionally failing debug example.
for f in $(find examples/dev -maxdepth 1 -name 'test_*.py' | sort); do
    case "$f" in
        *test_mlp_debug.py) continue ;;
    esac
    run_test "iterate: $(basename "$f")" \
        "$KBOX" iterate "$f" --once
done

run_test "iterate: isolated kernel_mode CUDA benchmark" \
    "$KBOX" iterate examples/dev/test_kernel_mode_cuda.py --once --bench --warmup 1 --iters 2 --isolated-kernel-benchmark

run_test "mcp: task service round-trip" \
    env PYTHONPATH=python "${PY_RUN[@]}" tests/test_kbox_mcp.py "$KBOX"
run_test "python: task service helper coverage" \
    env PYTHONPATH=python "${PY_RUN[@]}" tests/test_task_service.py

# Watch-mode regression tests for data-backed iterate flows.
run_test "watch: single .pt input file triggers rerun" \
    watch_single_pt_reload
run_test "watch: suite case file triggers rerun" \
    watch_suite_reload

# ── Helper module reload (regression for task 1) ─────────────────────
watch_helper_module_reload() {
    local tdir="$TMPROOT/watch_helper"
    mkdir -p "$tdir"
    # Create a helper module that provides the scale factor
    cat > "$tdir/my_helper.py" <<EOF
SCALE = 2.5
EOF
    cat > "$tdir/test_helper.py" <<'PYEOF'
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import my_helper

KERNEL_CODE = r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.5f;
}
"""

def init():
    x = torch.randn(1024, device="cuda")
    return {
        "kernel_source": KERNEL_CODE,
        "inputs": [x],
        "expected": [x * my_helper.SCALE],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
PYEOF
    local log="$tdir/watch.log"
    "$KBOX" iterate "$tdir/test_helper.py" >"$log" 2>&1 &
    local pid=$!
    wait_for_count "$log" "PASS:" 1 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    # Verify helper module is being watched
    if ! grep -q "my_helper" "$log"; then
        echo "helper module not in watched files"
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi
    # Modify the helper — should trigger reload + rerun
    cat > "$tdir/my_helper.py" <<EOF
SCALE = 2.5
EOF
    sleep 0.2
    touch "$tdir/my_helper.py"
    wait_for_count "$log" "PASS:" 2 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
run_test "watch: helper module reload triggers rerun" \
    watch_helper_module_reload

# ── Suite file add/remove ─────────────────────────────────────────────
watch_suite_add_remove() {
    local tdir="$TMPROOT/watch_suite_ar"
    mkdir -p "$tdir/cases"
    cp examples/data/mlp_cases/small_batch.pt "$tdir/cases/small_batch.pt"
    cat > "$tdir/test_suite_ar.py" <<EOF
import torch

def init_once():
    return {"h5_suite": "$tdir/cases"}

def run(inputs):
    hidden = torch.relu(inputs.x @ inputs.w1 + inputs.b1)
    return [hidden @ inputs.w2 + inputs.b2]
EOF
    local log="$tdir/watch.log"
    "$KBOX" iterate "$tdir/test_suite_ar.py" >"$log" 2>&1 &
    local pid=$!
    wait_for_count "$log" "Suite: 1/1 passed" 1 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    # Add a second case file — should trigger reload with 2 cases
    cp examples/data/mlp_cases/medium_batch.pt "$tdir/cases/medium_batch.pt"
    wait_for_count "$log" "Suite: 2/2 passed" 1 "$pid" 30 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
run_test "watch: suite case add triggers rerun with new count" \
    watch_suite_add_remove

# ── Inline kernel_source reload ──────────────────────────────────────
watch_inline_kernel_reload() {
    local tdir="$TMPROOT/watch_inline"
    mkdir -p "$tdir"
    cat > "$tdir/test_inline.py" <<'EOF'
import torch

KERNEL = r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 2.0f;
}
"""

def init():
    x = torch.ones(1024, device="cuda")
    return {
        "kernel_source": KERNEL,
        "inputs": [x],
        "expected": [x * 2.0],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
EOF
    local log="$tdir/watch.log"
    "$KBOX" iterate "$tdir/test_inline.py" >"$log" 2>&1 &
    local pid=$!
    wait_for_count "$log" "PASS:" 1 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    # Change kernel to multiply by 3.0 and update expected
    cat > "$tdir/test_inline.py" <<'EOF'
import torch

KERNEL = r"""
extern "C" __global__ void scale(const float *in0, float *out0, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out0[i] = in0[i] * 3.0f;
}
"""

def init():
    x = torch.ones(1024, device="cuda")
    return {
        "kernel_source": KERNEL,
        "inputs": [x],
        "expected": [x * 3.0],
    }

def run(inputs, kernel):
    return [kernel(inputs[0])]
EOF
    wait_for_count "$log" "PASS:" 2 "$pid" 25 || {
        cat "$log"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    }
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
run_test "watch: inline kernel_source reload" \
    watch_inline_kernel_reload

# ── Scratch and custom param paths ───────────────────────────────────
run_test "iterate: scratch buffer" \
    "$KBOX" iterate examples/dev/test_scratch.py --once
run_test "iterate: custom kernel params" \
    "$KBOX" iterate examples/dev/test_custom_params.py --once

# ── Install surface ──────────────────────────────────────────────────
install_surface() {
    local prefix="$TMPROOT/install"
    make install PREFIX="$prefix" >/dev/null || return 1
    [ -x "$prefix/bin/kbox" ] || return 1
    [ -x "$prefix/libexec/kbox/kbox_worker_daemon" ] || return 1
    [ -x "$prefix/libexec/kbox/kbox_iterate.py" ] || return 1
    [ -x "$prefix/libexec/kbox/kbox_isolated_kernel_mode.py" ] || return 1
    [ -x "$prefix/libexec/kbox/kbox_mcp.py" ] || return 1
}

run_test "install: helper scripts installed" \
    install_surface
run_test_grep "install: kbox binary exists" "kbox" \
    ls build/tools/kbox
run_test_grep "install: worker daemon exists" "kbox_worker_daemon" \
    ls build/tools/kbox_worker_daemon

echo "---"
echo "$PASS passed, $FAIL failed, $SKIP skipped"
[ $FAIL -eq 0 ]
