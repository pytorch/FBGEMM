#!/usr/bin/env python3
"""
Reproduce the conditions of the hanging UVM tests, without FBGEMM or pytest.

test_uvm_slice hangs on the MI350 CI runner and passes in ~10s on bare metal.
It never touches the allocated data - it allocates a managed tensor, slices it,
and compares storage pointers - so the cost has to be in allocation, the two
memory advises new_managed_tensor issues, or the free.

The shapes below are the exact ones Hypothesis generated on the runner, read out
of the CI log.  They are much larger than "a UVM test" suggests: 39 distinct
shapes up to 786 GiB, 2.09 TiB requested in total.  That is the condition this
script replicates, at the same sizes, through the same HIP calls, timing each
step separately.

It also reports the memory the environment actually has.  Nothing in the CI
pipeline has ever printed that - print_system_info covers CPU, PCI and kernel but
not memory - and the runner is a pod inside a KVM guest, so both guest RAM and a
cgroup limit are plausible constraints that our 3 TiB bare-metal box does not
have.

Ascending order with a bail-out, so the first size that misbehaves is identified
without wedging the machine on the one after it.
"""

import ctypes
import os
import sys
import time

# Exact shapes recorded on the MI350 VF runner, ascending by size.
REAL_SHAPES = [
    [1], [16], [1, 86], [107], [176], [463], [760], [946], [86, 86],
    [163, 105], [18, 351, 4], [246, 246], [943, 133], [256, 624],
    [96, 307, 23], [16, 270, 387], [61, 627, 234], [61, 627, 627],
    [1023, 513, 823], [133, 133, 943, 133], [163, 105, 249, 575],
    [300, 110, 320, 246], [255, 255, 939, 70], [943, 94, 377, 133],
    [300, 246, 320, 246], [246, 246, 320, 320], [868, 76, 383, 282],
    [300, 246, 320, 320], [320, 246, 320, 320], [943, 94, 943, 133],
    [255, 709, 939, 70], [943, 133, 943, 133], [133, 133, 943, 943],
    [255, 939, 939, 70], [939, 255, 939, 70], [939, 255, 939, 255],
    [255, 939, 939, 255], [943, 133, 943, 943], [939, 939, 939, 255],
]

# Stop climbing once a single step takes longer than this.  The next shape is
# often several times larger, so continuing past a slow one risks a wedge.
BAIL_SECONDS = float(os.environ.get("VF_REPRO_BAIL_SECONDS", "60"))
# Cap in GiB, so a cautious run can be done on a shared machine first.
MAX_GIB = float(os.environ.get("VF_REPRO_MAX_GIB", "1024"))

hipMemAdviseSetPreferredLocation = 3
hipMemAdviseSetAccessedBy = 5
hipCpuDeviceId = -1


def load_hip():
    cands = []
    try:
        import torch
        cands.append(os.path.join(os.path.dirname(torch.__file__), "lib",
                                  "libamdhip64.so"))
    except Exception:
        pass
    cands += ["/opt/rocm/lib/libamdhip64.so", "libamdhip64.so",
              "libamdhip64.so.7", "libamdhip64.so.6"]
    for c in cands:
        try:
            return ctypes.CDLL(c), c
        except OSError:
            continue
    return None, "NOT FOUND"


def read_first(path, keys):
    try:
        with open(path) as f:
            for line in f:
                for k in keys:
                    if line.startswith(k):
                        return line.strip()
    except Exception:
        pass
    return None


def mem_available_gib():
    """MemAvailable in GiB, to show whether an allocation actually commits."""
    line = read_first("/proc/meminfo", ("MemAvailable",))
    return int(line.split()[1]) / 1048576 if line else float("nan")


def report_memory(label):
    """Guest RAM and cgroup limits.  A pod limit is invisible to /proc/meminfo,
    so both are reported."""
    print(f"  [{label}]")
    for k in ("MemTotal", "MemAvailable", "SwapTotal", "Committed_AS"):
        line = read_first("/proc/meminfo", (k,))
        if line:
            val = int(line.split()[1]) / 1048576
            print(f"    {k:14}: {val:10.1f} GiB")
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory.current",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                print(f"    {os.path.basename(path):14}: max (unlimited)")
            else:
                print(f"    {os.path.basename(path):14}: {int(raw)/2**30:10.1f} GiB")
        except Exception:
            pass
    for path in ("/proc/sys/vm/overcommit_memory", "/proc/sys/vm/overcommit_ratio"):
        try:
            with open(path) as f:
                print(f"    {os.path.basename(path):14}: {f.read().strip()}")
        except Exception:
            pass


def main():
    hip, hip_path = load_hip()
    print("=" * 78)
    print("UVM ALLOCATION REPRODUCTION")
    print("=" * 78)
    print(f"  libamdhip64 : {hip_path}")
    print(f"  bail after  : {BAIL_SECONDS}s per step")
    print(f"  size cap    : {MAX_GIB} GiB")
    if hip is None:
        print("  libamdhip64 unavailable - cannot run")
        return 1

    hip.hipMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_size_t, ctypes.c_uint]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipMemAdvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_int, ctypes.c_int]

    try:
        import torch
        torch.zeros(1, device="cuda")
        print(f"  device      : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  device      : torch init failed ({e})")

    report_memory("memory at start")
    print()
    print(f"{'shape':32} {'GiB':>9} {'alloc_ms':>10} {'ms/GiB':>8} "
          f"{'advise2_ms':>11} {'free_ms':>9} {'commit_GiB':>11}  rc")
    print("-" * 100)

    for shape in REAL_SHAPES:
        nbytes = 4
        for d in shape:
            nbytes *= d
        gib = nbytes / 2**30
        if gib > MAX_GIB:
            print(f"{str(shape):32} {gib:10.3f}  skipped, above cap")
            continue

        p = ctypes.c_void_p()
        avail_before = mem_available_gib()
        t0 = time.perf_counter()
        rc = hip.hipMallocManaged(ctypes.byref(p), nbytes, 1)
        t_alloc = time.perf_counter() - t0
        if rc != 0:
            print(f"{str(shape):32} {gib:10.3f} {t_alloc*1e3:10.1f}  "
                  f"ALLOC FAILED rc={rc}", flush=True)
            report_memory("memory after failed alloc")
            break

        # Exactly what new_managed_tensor does after the allocation.
        t0 = time.perf_counter()
        hip.hipMemAdvise(p, nbytes, hipMemAdviseSetPreferredLocation, hipCpuDeviceId)
        t_adv1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        hip.hipMemAdvise(p, nbytes, hipMemAdviseSetAccessedBy, 0)
        t_adv2 = time.perf_counter() - t0

        # How much memory the allocation actually consumed: distinguishes a
        # lazy virtual reservation from an eager commit.
        committed = avail_before - mem_available_gib()

        t0 = time.perf_counter()
        hip.hipFree(p)
        t_free = time.perf_counter() - t0

        rate = t_alloc * 1e3 / gib if gib > 0.001 else float('nan')
        print(f"{str(shape):32} {gib:9.3f} {t_alloc*1e3:10.1f} {rate:8.2f} "
              f"{t_adv2*1e3:11.1f} {t_free*1e3:9.1f} {committed:11.1f}  {rc}", flush=True)

        worst = max(t_alloc, t_adv1, t_adv2, t_free)
        if worst > BAIL_SECONDS:
            print(f"\n  BAIL: a step took {worst:.1f}s at {gib:.1f} GiB; "
                  f"not attempting larger shapes.")
            report_memory("memory at bail")
            break

    print()
    report_memory("memory at end")
    print("=" * 78)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
