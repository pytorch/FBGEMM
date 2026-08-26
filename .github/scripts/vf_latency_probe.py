#!/usr/bin/env python3
"""
Standalone GPU virtualization latency probe.  No FBGEMM, no pytest, no compiler.

Run the same script on a bare-metal GPU and on the virtualized CI runner and diff
the output.  Every test prints one RESULT line so the two logs can be compared
mechanically.

Hypotheses under test:
  C1  the guest CPU is slow                      (control - must rule out first)
  C2  the guest gets a fraction of the GPU       (control)
  H1  fixed cost per host<->device round trip
  H2  the cost is in completion, not submission
  H3  the cost is interrupt delivery to the guest
  H4  small device->host copies are slow
  H5  host access to managed memory is slow (in FBGEMM's advised config)
  H7  allocation is slow
  H8  deallocation is slow (only managed free was ever measured)
  H9  pinned host allocation is slow
  H10 allocation churn at varying sizes is slow
  H11 the slow test's actual hot loop: sort + unique_consecutive + D->H sync
"""

import ctypes
import os
import statistics
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# HIP bindings via ctypes - keeps this independent of any compiler or framework
# --------------------------------------------------------------------------- #

def _load_hip():
    """Locate libamdhip64 without assuming the loader path resolves it.

    The unversioned .so is often only present in the -dev package, so the plain
    soname can fail on a runtime-only image.  Prefer the copy PyTorch bundles:
    it is guaranteed to exist wherever this script can run at all, and because
    torch has already dlopen'd it, we get the same handle and therefore the same
    HIP context rather than a second runtime in the process.

    Returns (handle, description).  A None handle is not fatal - the tests that
    need raw HIP report themselves skipped, and the torch-only tests, which are
    the ones that matter most, still run.
    """
    candidates = []
    try:
        import torch
        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib",
                                       "libamdhip64.so"))
    except Exception:
        pass
    candidates += [
        "/opt/rocm/lib/libamdhip64.so",
        "libamdhip64.so",
        "libamdhip64.so.7",
        "libamdhip64.so.6",
    ]
    for path in candidates:
        try:
            return ctypes.CDLL(path), path
        except OSError:
            continue
    # Last resort: torch links against it, so the symbols may already be global.
    try:
        h = ctypes.CDLL(None)
        h.hipMalloc  # probe for the symbol; raises AttributeError if absent
        return h, "(already-loaded global symbols)"
    except Exception as e:
        return None, f"NOT FOUND ({e})"


hip, HIP_LIB_PATH = _load_hip()

hipMemAdviseSetPreferredLocation = 3
hipMemAdviseSetAccessedBy = 5
hipMemcpyDeviceToHost = 2
hipCpuDeviceId = -1

if hip is not None:
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    hip.hipMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemAdvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceSynchronize.argtypes = []
    hip.hipHostMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    hip.hipHostFree.argtypes = [ctypes.c_void_p]


def require_hip(tag):
    """Report a raw-HIP test as skipped rather than crashing the run."""
    if hip is None:
        result(tag, status=f"SKIPPED - libamdhip64 unavailable: {HIP_LIB_PATH}")
        print()
        return False
    return True


def hip_check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed with hipError {rc}")


def ns():
    return time.perf_counter_ns()


def stats(samples_ns):
    """Return (median, mean, p99) in microseconds."""
    s = sorted(samples_ns)
    n = len(s)
    return (
        s[n // 2] / 1e3,
        sum(s) / n / 1e3,
        s[min(n - 1, int(n * 0.99))] / 1e3,
    )


def result(tag, **kv):
    body = "  ".join(f"{k}={v}" for k, v in kv.items())
    print(f"RESULT {tag:22} {body}", flush=True)


def histogram(samples_ns, tag):
    """Log-spaced histogram.  A scheduling quantum shows up as a spike, and as
    multiples of that spike - a mean would hide exactly that structure."""
    edges = [0, 10e3, 30e3, 100e3, 300e3, 1e6, 3e6, 6e6, 12e6, 24e6, 48e6, float("inf")]
    labels = ["<10us", "10-30us", "30-100us", "0.1-0.3ms", "0.3-1ms", "1-3ms",
              "3-6ms", "6-12ms", "12-24ms", "24-48ms", ">48ms"]
    counts = [0] * (len(edges) - 1)
    for v in samples_ns:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
    total = max(1, len(samples_ns))
    print(f"  histogram[{tag}]")
    for lab, c in zip(labels, counts):
        if c:
            bar = "#" * max(1, int(60 * c / total))
            print(f"    {lab:>10} {c:6d}  {bar}")


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #

def fingerprint():
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"  libamdhip64     : {HIP_LIB_PATH}")
    import torch
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    # The " VF" suffix on the marketing name is how a virtual function is
    # distinguished from a physical one, so report every source of it.
    print(f"  device name     : {torch.cuda.get_device_name(dev) or '(empty)'}")
    print(f"  props.name      : {getattr(props, 'name', None) or '(empty)'}")
    try:
        smi = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in smi.splitlines():
            if "Card Series" in line or "Market Name" in line:
                print(f"  rocm-smi        : {line.strip()}")
                break
    except Exception:
        pass
    print(f"  gcnArchName     : {props.gcnArchName}")
    print(f"  CUs             : {props.multi_processor_count}")
    print(f"  total memory    : {props.total_memory / 2**30:.1f} GiB")
    print(f"  torch           : {torch.__version__}")
    print(f"  HIP             : {torch.version.hip}")
    for var in ("HSA_ENABLE_INTERRUPT", "HSA_XNACK", "AMD_SERIALIZE_KERNEL",
                "HIP_VISIBLE_DEVICES", "HIP_LAUNCH_BLOCKING"):
        print(f"  {var:16}: {os.environ.get(var, '(unset)')}")
    # Is this a VM?  Is the GPU a virtual function?
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if any(k in line for k in ("Hypervisor vendor", "Virtualization type",
                                       "Model name", "CPU(s):")):
                print(f"  lscpu           : {line.strip()}")
    except Exception as e:
        print(f"  lscpu           : unavailable ({e})")
    print()


# --------------------------------------------------------------------------- #
# C1 - CPU control.  If the guest CPU is slow, every host-loop-heavy test is
# slow for reasons that have nothing to do with the GPU.
# --------------------------------------------------------------------------- #

def c1_cpu_control():
    print("-- C1  CPU control (no GPU) " + "-" * 48)
    t0 = ns()
    acc = 0
    for i in range(5_000_000):
        acc += i * i % 7
    pure_python_ms = (ns() - t0) / 1e6

    import numpy as np
    a = np.random.rand(1024, 1024).astype(np.float32)
    b = np.random.rand(1024, 1024).astype(np.float32)
    a @ b  # warm
    t0 = ns()
    for _ in range(10):
        a @ b
    numpy_ms = (ns() - t0) / 1e6 / 10

    result("C1_cpu", python_loop_ms=f"{pure_python_ms:.0f}", numpy_1k_matmul_ms=f"{numpy_ms:.2f}")
    print()


# --------------------------------------------------------------------------- #
# C2 - Bulk GPU throughput.  Rules out "we only get a slice of the GPU".
# --------------------------------------------------------------------------- #

def c2_gpu_throughput():
    print("-- C2  bulk GPU throughput " + "-" * 49)
    import torch
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = ns()
    iters = 20
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (ns() - t0) / 1e9
    tflops = (2 * n**3 * iters) / dt / 1e12
    result("C2_gemm", size=n, tflops=f"{tflops:.1f}", per_iter_ms=f"{dt/iters*1e3:.2f}")
    del a, b
    torch.cuda.empty_cache()
    print()


# --------------------------------------------------------------------------- #
# H1 - round-trip latency: the headline number.
# --------------------------------------------------------------------------- #

def h1_roundtrip(iters=2000, show_hist=True):
    print("-- H1  kernel launch + synchronize round trip " + "-" * 30)
    import torch
    t = torch.ones(1, device="cuda")
    for _ in range(50):
        t.add_(1.0)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = ns()
        t.add_(1.0)
        torch.cuda.synchronize()
        samples.append(ns() - t0)

    med, mean, p99 = stats(samples)
    result("H1_roundtrip", iters=iters, median_us=f"{med:.1f}",
           mean_us=f"{mean:.1f}", p99_us=f"{p99:.1f}")
    if show_hist:
        histogram(samples, "H1 launch+sync")
    print()
    return med


# --------------------------------------------------------------------------- #
# H2 - is the cost in submission or in completion?
# --------------------------------------------------------------------------- #

def h2_submit_vs_complete(iters=2000):
    print("-- H2  submission vs completion " + "-" * 44)
    import torch
    t = torch.ones(1, device="cuda")
    for _ in range(50):
        t.add_(1.0)
    torch.cuda.synchronize()

    # N launches, ONE sync at the end -> amortized submission cost
    t0 = ns()
    for _ in range(iters):
        t.add_(1.0)
    submit_only_ns = ns() - t0          # queueing only, sync excluded
    torch.cuda.synchronize()
    total_ns = ns() - t0

    per_submit_us = submit_only_ns / iters / 1e3
    per_total_us = total_ns / iters / 1e3
    result("H2_submit", iters=iters, per_submit_us=f"{per_submit_us:.2f}",
           per_iter_batched_us=f"{per_total_us:.2f}")

    # A bare synchronize on an idle queue -> pure completion-path cost
    torch.cuda.synchronize()
    samples = []
    for _ in range(500):
        t0 = ns()
        torch.cuda.synchronize()
        samples.append(ns() - t0)
    med, mean, p99 = stats(samples)
    result("H2_idle_sync", median_us=f"{med:.2f}", mean_us=f"{mean:.2f}", p99_us=f"{p99:.2f}")
    print()


# --------------------------------------------------------------------------- #
# H3 - interrupt delivery.  Re-runs H1 in a child process with interrupts off.
# HSA_ENABLE_INTERRUPT is read once at ROCr init, so it must be a new process.
# --------------------------------------------------------------------------- #

def h3_interrupt_vs_polling():
    print("-- H3  interrupt vs polling completion " + "-" * 37)
    env = dict(os.environ)
    env["HSA_ENABLE_INTERRUPT"] = "0"
    env["VF_PROBE_CHILD"] = "1"
    out = subprocess.run([sys.executable, os.path.abspath(__file__)],
                         env=env, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT H1_roundtrip")]
    if line:
        print(f"  with HSA_ENABLE_INTERRUPT=0 -> {line[0]}")
    else:
        print("  child run failed; stderr tail:")
        print("   ", (out.stderr or "(empty)").strip().splitlines()[-3:])
    print()


# --------------------------------------------------------------------------- #
# H4 - small device->host copy latency
# --------------------------------------------------------------------------- #

def h4_small_copy(iters=1000):
    print("-- H4  4-byte device->host copy " + "-" * 44)
    if not require_hip("H4_d2h_4B"):
        return
    dptr = ctypes.c_void_p()
    hip_check(hip.hipMalloc(ctypes.byref(dptr), 4), "hipMalloc")
    host = ctypes.create_string_buffer(4)
    for _ in range(20):
        hip.hipMemcpy(host, dptr, 4, hipMemcpyDeviceToHost)
    samples = []
    for _ in range(iters):
        t0 = ns()
        hip.hipMemcpy(host, dptr, 4, hipMemcpyDeviceToHost)
        samples.append(ns() - t0)
    med, mean, p99 = stats(samples)
    result("H4_d2h_4B", iters=iters, median_us=f"{med:.2f}",
           mean_us=f"{mean:.2f}", p99_us=f"{p99:.2f}")
    hip.hipFree(dptr)
    print()


# --------------------------------------------------------------------------- #
# H5 - host access to managed memory, one touch per 4 KiB page
# --------------------------------------------------------------------------- #

def h5_managed_host_access(pages=4096):
    """Host access cost to a managed range, in the configuration FBGEMM actually uses.

    new_managed_tensor() does cudaMallocManaged followed by
    SetPreferredLocation=host and SetAccessedBy=device, i.e. the data is
    deliberately host-resident with the GPU mapping it directly ("no page faults
    will be generated").  So the question is not fault-and-migrate cost, it is
    what plain host reads/writes to that mapping cost under virtualization -
    which is what test_uvm_slice's host-side loop does.

    Both configurations are measured so the advises can be shown to matter or not.
    """
    print("-- H5  host access to managed memory " + "-" * 39)
    if not require_hip("H5_managed"):
        return
    PAGE = 4096
    size = pages * PAGE

    for label, advise in (("plain", False), ("fbgemm_advised", True)):
        mptr = ctypes.c_void_p()
        rc = hip.hipMallocManaged(ctypes.byref(mptr), size, 1)  # hipMemAttachGlobal
        if rc != 0:
            result(f"H5_{label}", status=f"hipMallocManaged failed rc={rc}")
            continue
        if advise:
            hip.hipMemAdvise(mptr, size, hipMemAdviseSetPreferredLocation, hipCpuDeviceId)
            hip.hipMemAdvise(mptr, size, hipMemAdviseSetAccessedBy, 0)

        buf = (ctypes.c_ubyte * size).from_address(mptr.value)

        writes = []
        for p in range(pages):
            off = p * PAGE
            t0 = ns()
            buf[off] = 1
            writes.append(ns() - t0)
        reads = []
        for p in range(pages):
            off = p * PAGE
            t0 = ns()
            _ = buf[off]
            reads.append(ns() - t0)

        wm, wmean, wp99 = stats(writes)
        rm, rmean, rp99 = stats(reads)
        result(f"H5_{label}", pages=pages,
               write_median_us=f"{wm:.3f}", write_p99_us=f"{wp99:.3f}",
               read_median_us=f"{rm:.3f}", read_p99_us=f"{rp99:.3f}",
               total_ms=f"{(sum(writes)+sum(reads))/1e6:.1f}")
        histogram(writes, f"H5 {label} per-page host write")
        hip.hipFree(mptr)
    print()


# H6 (hipMemAdvise cost scales with range size) was dropped: it assumed the test
# generates multi-TB ranges, which the bare-metal evidence contradicts.


# --------------------------------------------------------------------------- #
# H7 - allocation latency
# --------------------------------------------------------------------------- #

def h7_alloc(iters=200):
    print("-- H7  allocation latency " + "-" * 50)
    if not require_hip("H7_alloc"):
        return
    for label, fn, sz in (("hipMalloc_1MiB", "malloc", 1 << 20),
                          ("hipMallocManaged_1MiB", "managed", 1 << 20)):
        samples = []
        for _ in range(iters):
            p = ctypes.c_void_p()
            t0 = ns()
            if fn == "malloc":
                rc = hip.hipMalloc(ctypes.byref(p), sz)
            else:
                rc = hip.hipMallocManaged(ctypes.byref(p), sz, 1)
            samples.append(ns() - t0)
            if rc == 0:
                hip.hipFree(p)
        med, mean, p99 = stats(samples)
        result(f"H7_{label}", iters=iters, median_us=f"{med:.2f}",
               mean_us=f"{mean:.2f}", p99_us=f"{p99:.2f}")
    print()


# --------------------------------------------------------------------------- #

# Each test runs in its own process with a wall-clock budget, so one wedged HIP
# call cannot take the suite down or discard the results already collected.
# --------------------------------------------------------------------------- #
# H8 - deallocation.  H7 timed allocation only; the one free that was measured
# (managed memory, in the UVM ladder) was 119x slower on the VF.  Ordinary
# device free has never been timed, and every caching-allocator eviction hits it.
# --------------------------------------------------------------------------- #

def h8_device_free(iters=300, size=1 << 20):
    print("-- H8  hipFree of device memory " + "-" * 44)
    if not require_hip("H8_device_free"):
        return
    allocs, frees = [], []
    for _ in range(iters):
        p = ctypes.c_void_p()
        t0 = ns()
        rc = hip.hipMalloc(ctypes.byref(p), size)
        allocs.append(ns() - t0)
        if rc != 0:
            continue
        t0 = ns()
        hip.hipFree(p)
        frees.append(ns() - t0)
    am, _, ap = stats(allocs)
    fm, fmean, fp = stats(frees)
    result("H8_device_free", iters=iters, size_MiB=size // 2**20,
           alloc_median_us=f"{am:.2f}", free_median_us=f"{fm:.2f}",
           free_mean_us=f"{fmean:.2f}", free_p99_us=f"{fp:.2f}")
    histogram(frees, "H8 hipFree device")
    print()


# --------------------------------------------------------------------------- #
# H9 - pinned host memory.  Never measured, and it is the other host-adjacent
# allocator besides managed memory.
# --------------------------------------------------------------------------- #

def h9_pinned(iters=200, size=1 << 20):
    print("-- H9  hipHostMalloc / hipHostFree (pinned) " + "-" * 32)
    if not require_hip("H9_pinned"):
        return
    allocs, frees = [], []
    for _ in range(iters):
        p = ctypes.c_void_p()
        t0 = ns()
        rc = hip.hipHostMalloc(ctypes.byref(p), size, 0)
        allocs.append(ns() - t0)
        if rc != 0:
            result("H9_pinned", status=f"hipHostMalloc failed rc={rc}")
            print()
            return
        t0 = ns()
        hip.hipHostFree(p)
        frees.append(ns() - t0)
    am, _, ap = stats(allocs)
    fm, _, fp = stats(frees)
    result("H9_pinned", iters=iters, size_MiB=size // 2**20,
           alloc_median_us=f"{am:.2f}", alloc_p99_us=f"{ap:.2f}",
           free_median_us=f"{fm:.2f}", free_p99_us=f"{fp:.2f}")
    print()


# --------------------------------------------------------------------------- #
# H10 - allocation churn at torch level, with sizes that vary per iteration so
# the caching allocator cannot simply hand back the same block.  This is what
# gradcheck does and what the H1 steady-state loop never did.
# --------------------------------------------------------------------------- #

def h10_alloc_churn(iters=2000):
    print("-- H10  torch alloc/free churn, varying sizes " + "-" * 30)
    import torch
    for _ in range(50):
        torch.empty(1024, device="cuda")
    torch.cuda.synchronize()
    samples = []
    for i in range(iters):
        n = 1024 + (i % 97) * 512          # varies, defeats block reuse
        t0 = ns()
        x = torch.empty(n, device="cuda")
        del x
        samples.append(ns() - t0)
    med, mean, p99 = stats(samples)
    result("H10_alloc_churn", iters=iters, median_us=f"{med:.2f}",
           mean_us=f"{mean:.2f}", p99_us=f"{p99:.2f}")
    histogram(samples, "H10 torch alloc+free")
    print()


# --------------------------------------------------------------------------- #
# H11 - the actual hot loop of the slow test.
#
# index_select_dim0's backward calls at::unique_consecutive, which the FBGEMM
# source itself documents as doing a D->H transfer that forces host-device
# synchronization (sparse_index_add.cu:144-152), followed by allocations whose
# size is only known after that sync.  gradcheck runs this thousands of times.
# at::unique_consecutive is torch.unique_consecutive in Python, so this needs no
# compiler and no FBGEMM - which matters, because this step runs before the
# FBGEMM wheel is installed.
# --------------------------------------------------------------------------- #

def h11_unique_consecutive_loop(iters=500, n=32, u=33):
    print("-- H11  sort + unique_consecutive + D->H sync loop " + "-" * 25)
    import torch
    idx = torch.randint(u, (n,), device="cuda")
    for _ in range(20):
        s, _o = idx.sort()
        uq, cnt = torch.unique_consecutive(s, return_counts=True)
        _ = uq.numel()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = ns()
        sorted_indices, orig_indices = idx.sort()
        unique_indices, unique_count = torch.unique_consecutive(
            sorted_indices, return_counts=True)
        num_unique = unique_indices.numel()      # D->H transfer
        offsets = unique_count.cumsum(0)         # size depends on the above
        samples.append(ns() - t0)
    med, mean, p99 = stats(samples)
    result("H11_unique_consec", iters=iters, n=n,
           median_us=f"{med:.1f}", mean_us=f"{mean:.1f}", p99_us=f"{p99:.1f}",
           est_84k_calls_s=f"{mean * 84000 / 1e6:.1f}")
    histogram(samples, "H11 unique_consecutive iteration")
    print()


TESTS = {
    "fingerprint": (fingerprint, 120),
    "c1": (c1_cpu_control, 180),
    "c2": (c2_gpu_throughput, 240),
    "h1": (h1_roundtrip, 240),
    "h2": (h2_submit_vs_complete, 240),
    "h3": (h3_interrupt_vs_polling, 360),
    "h4": (h4_small_copy, 180),
    "h5": (h5_managed_host_access, 300),
    "h7": (h7_alloc, 240),
    "h8": (h8_device_free, 240),
    "h9": (h9_pinned, 240),
    "h10": (h10_alloc_churn, 240),
    "h11": (h11_unique_consecutive_loop, 300),
}


def run_child(name):
    fn, _ = TESTS[name]
    if name not in ("fingerprint", "c1", "h3"):
        import torch
        if not torch.cuda.is_available():
            print("no GPU available", file=sys.stderr)
            sys.exit(1)
        torch.zeros(1, device="cuda")  # init context before timing anything
    fn()


def main():
    # H3 re-invokes the script with interrupts disabled; it only wants H1.
    if os.environ.get("VF_PROBE_CHILD"):
        import torch  # noqa: F401
        torch.zeros(1, device="cuda")
        h1_roundtrip(show_hist=False)
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        run_child(sys.argv[2])
        return

    for name, (_, budget) in TESTS.items():
        try:
            p = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__), "--test", name],
                timeout=budget, capture_output=True, text=True,
            )
            sys.stdout.write(p.stdout)
            if p.returncode != 0:
                print(f"!! {name} exited rc={p.returncode}")
                tail = (p.stderr or "").strip().splitlines()[-5:]
                for l in tail:
                    print(f"   {l}")
                print()
        except subprocess.TimeoutExpired as e:
            sys.stdout.write(e.stdout.decode() if e.stdout else "")
            print(f"!! {name} EXCEEDED its {budget}s budget and was killed.")
            print("   Note: a HIP call stuck in the kernel may defer SIGKILL; "
                  "check for a lingering process before re-running.\n")
        sys.stdout.flush()

    print("=" * 78)
    print("done")


if __name__ == "__main__":
    main()
