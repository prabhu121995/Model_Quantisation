"""Find and stop this project's running models (`status` / `stop` commands)."""

import psutil

# Command-line fragments that identify this project's processes.
_MARKERS = {
    "model-quantization": "model-quantization command",
    "model_quantization": "model-quantization command",
    "benchmark.py": "benchmark",
    "run_quantized.py": "run script",
    "quantize_qwen.py": "quantize script",
}


def _describe(proc: psutil.Process) -> str | None:
    try:
        cmd = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return None
    for marker, label in _MARKERS.items():
        if marker in cmd:
            return label
    return None


def find() -> list[tuple[psutil.Process, str]]:
    """This project's processes, excluding the current command and its launcher (uv)."""
    me = psutil.Process()
    skip = {me.pid} | {p.pid for p in me.parents()}
    found = []
    for proc in psutil.process_iter():
        if proc.pid in skip:
            continue
        label = _describe(proc)
        if label:
            found.append((proc, label))
    return found


def _mem_gb(proc: psutil.Process) -> float:
    try:
        return proc.memory_info().rss / 1024**3
    except psutil.Error:
        return 0.0


def status() -> None:
    procs = find()
    if not procs:
        print("Nothing running.")
        return
    print(f"{'PID':>7}  {'RAM GB':>6}  what")
    for proc, label in procs:
        print(f"{proc.pid:>7}  {_mem_gb(proc):>6.2f}  {label}")
    total = sum(_mem_gb(p) for p, _ in procs)
    free = psutil.virtual_memory().available / 1024**3
    print(f"\nUsing {total:.2f} GB RAM. Free RAM: {free:.2f} GB")


def stop() -> None:
    procs = find()
    if not procs:
        print("Nothing to stop.")
        return
    for proc, label in procs:
        print(f"Stopping {proc.pid} ({label})")
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs([p for p, _ in procs], timeout=10)
    for proc in alive:  # did not exit politely -> force
        try:
            proc.kill()
        except psutil.Error:
            pass
    print("Stopped.")


if __name__ == "__main__":
    status()
