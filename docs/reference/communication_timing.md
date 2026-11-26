# Read-only communication timing

Measured on the connected SO-101 on 2026-11-24. All five arm joints retained
their existing hold; the gripper remained unpowered. The benchmark sent no goal,
torque, gain, or voltage writes. The camera/UI stayed running while releasing its
motor port exclusively to the benchmark.

300 back-to-back position reads: median 1.40 ms, p99 1.60 ms, maximum 2.07 ms.
Each request reads all five arm joint positions.

Extended sweep, one second per rate and mode:

| Requested Hz | Period ms | Two-read p99 ms | Missed deadlines / samples |
| --- | --- | --- | --- |
| 100 | 10.00 | 3.16 | 0 / 100 |
| 150 | 6.67 | 3.37 | 1 / 150 |
| 200 | 5.00 | 3.11 | 1 / 200 |
| 250 | 4.00 | 3.07 | 0 / 250 |
| 400 | 2.50 | 3.11 | 398 / 400 |

The two-read cycle reads position and goal registers as a traffic proxy. It does
not measure an actual read–command control cycle. Short sweeps can miss rare
scheduling delays; non-monotonic miss counts demonstrate this variability.

100 Hz is a candidate for full-loop validation, not a proven maximum control
rate. It passed both sweeps and leaves over half a cycle beyond p99 read traffic.
Validate actual read/write service times, scheduling jitter, and joint tracking
at 40 Hz first, then higher rates during operator-triggered Go trials. Do not use
average serial throughput alone to select the control rate.

Reproduce with the viewer idle:

```powershell
python -X utf8 -m deskbuddy.timing_benchmark --seconds 3
python -X utf8 -m deskbuddy.timing_benchmark --seconds 1 --rates 100 150 200 250 400 --output runs/timing_benchmark_extended.json
```

The command refuses to run while another operation owns the motor port and
returns the port to the viewer without changing torque. Local JSON reports include
latency percentiles, wake-up lateness, and missed deadlines. Future completed
Go results include `control_timing` for the actual position-read/command-write
loop. Update frequency and trajectory speed are separate settings: a faster
feedback loop alone does not change the configured trajectory duration.
