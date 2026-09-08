# Intel AI PC single-run performance test

- Status: `success`
- Protocol: models loaded before timing; one complete measured run; no full-video warm-up
- CPU: Intel(R) Core(TM) i9-14900HX
- Video: `radio.mp4`
- Resolution: 1280x720
- Source frame rate: 24.000 FPS
- Source duration: 15.042 s
- Analysis FPS setting: 4.0
- YOLO requested device: `intel:gpu`
- YOLO actual device: `intel:gpu`
- YOLO backend: `openvino`
- YOLO precision: `fp16`
- Qwen devices: `cpu`

## Required metrics

| Metric | Value | Definition |
|---|---:|---|
| End-to-End Latency | 1194.002 s | Complete measured pipeline wall time |
| Video processing FPS | 0.302 FPS | Decoded frames divided by E2E time |
| Pipeline analysis throughput | 0.051 frames/s | Actual YOLO calls divided by E2E time |
| Process CPU mean / peak | 2163.471% / 2478.100% | psutil process sampling; may exceed 100% on multicore CPUs |
| Process memory mean / peak | 5.102 / 5.430 GiB | Process RSS |
| Real-time factor | 0.013x | Source duration divided by E2E time |

This short-video result is a basic local performance measurement. It is not a long-duration stability test.
