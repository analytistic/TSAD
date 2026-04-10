<p align="center">
<img width="500" src="assets/TSAD_sample.svg"/>
</p>


<h1 align="center">TSAD</h1>

<h2 align="center">Online Anomaly Detection for Event Streams: Paradigms, Benchmarks & Evaluator</h2>

<p align="center">
<img alt="PyPI" src="https://img.shields.io/pypi/v/ts_ad"> 
<!--<img alt="PyPI - Downloads" src="https://pepy.tech/badge/ts_ad"> -->
<img alt="License" src="https://img.shields.io/github/license/analytistic/TSAD">
</p>

Starting from the **online anomaly detection** requirements for real-world event sequences, we revisit existing detection paradigms and benchmarks. Real-world event streams are marked by high throughput, low latency, and continuously changing distributions — traditional offline batch evaluation fails to fully reflect model performance in online scenarios.


Our project focuses on addressing this gap, with the following goals:

* 🎯 **Core Requirements Definition**: Clarify key metrics for streaming anomaly detection — online inference latency, throughput, anomaly detection capability, and delayed alerting performance.
* 📐 **Evaluation Protocol Redesign**: Adopt event-based online scoring (accounting for detection latency & false alarm costs), sliding window with rolling calibration, and latency-sensitive precision/recall metrics.
* 🧪 **Reproducible Baselines**: Provide data preparation scripts, real event sequence samples from \`data/TSB-AD-U\`, and standardized online train/test splitting strategies.
* ⚡ **Lightweight Evaluator**: Release a streamlined streaming evaluator for end-to-end evaluation in real engineering pipelines (jointly measuring latency, throughput, and accuracy).

**TODO**
