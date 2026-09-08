# TransitEye 🚌👁️

**Turning public buses into a city-wide sensing network.**

Built for Smart India Hackathon 2026 — Problem Statement 26124 (Disaster Management / Urban Intelligence).

---

## The problem

City buses already have cameras running on them every single day, covering almost every major road. But right now those cameras only *record* — nobody's actually using them to understand what's happening on the road. Meanwhile, cities still rely on manual inspections and citizen complaints to find potholes, traffic jams, and unsafe driving. That's slow, incomplete, and reactive.

**TransitEye's idea is simple:** if the cameras are already there, riding the bus every day anyway, why not let them see and report what's going on?

## What it actually does

- **Spots road defects** — potholes and damaged road patches, detected live from bus camera footage, tagged with severity, GPS location, and a timestamp
- **Reads traffic density** — counts and classifies vehicles frame by frame to estimate how congested a stretch of road is, and renders it as a live heatmap
- **Catches plates** — automatic number plate recognition (ANPR) for hit-and-run and rash-driving incidents, pairing the plate number with GPS, timestamp, and a confidence score
- **Puts it all on a map** — a live GIS dashboard where every pothole, traffic hotspot, and flagged vehicle shows up in real time
- **Writes it up for you** — one-click exportable incident reports (PDF) for transport authorities, so this isn't just a dashboard nobody reads

## How it's built

Everything funnels through the same simple idea: **detect locally, send only what matters, show it on a map.**

```
Bus camera feed
      ↓
YOLOv8 detection (potholes / vehicles / plates)
      ↓
Lightweight metadata only — never raw video
      ↓
Firebase (real-time database)
      ↓
Live dashboard — map, heatmap, alerts, reports
```

We deliberately never send raw video off the bus — only small, structured alerts (a GPS point, a timestamp, a confidence score, a plate number). That keeps bandwidth low and means the system still works even when connectivity is patchy, which matters a lot on real Indian roads.

## What's in this repo

```
transit-eye/
├── dashboard/          → the live GIS dashboard (map, heatmaps, alerts)
├── data/                → training data for our fine-tuned models
├── docs/                 → notes and write-ups on our approach
├── models/               → trained model weights
│   ├── defect_detection/   → pothole/road-defect detection
│   ├── anpr/                → number-plate detection + OCR
│   └── traffic_density_model.pt  → our fine-tuned vehicle/density model
├── pipeline/            → the live detection scripts that tie it together
├── reports/              → generated incident reports and demo output
├── scripts/              → training, dataset prep, and utility scripts
├── index10.html          → the dashboard, standalone
├── detect_live_demo_final.py → pothole detection pipeline
└── requirements.txt
```

## Why we did it this way

- **YOLOv8** across the board — it's fast enough to run in real time, and mature enough that we could fine-tune it in the time we had rather than building detection from scratch
- **Fine-tuned specifically for Indian roads** — standard pretrained models don't recognise auto-rickshaws, and struggle more with occlusion, low light, and the general chaos of Indian traffic than they let on. We fine-tuned on a curated Indian-conditions dataset (auto-rickshaws, dense/occluded traffic, low-light and foggy scenes) to actually account for that, rather than pretending a Western-trained model would just work
- **Firebase** for the real-time layer — it meant we could get a live, multi-source dashboard working fast, without building our own backend from scratch
- **Metadata-only transmission** — raw video never leaves the bus. Only alerts do. That's both a bandwidth decision and a privacy one

## Honest limitations (we'd rather tell you than have you find out)

- This is a working proof-of-concept, not a finished production platform — a few things (edge deployment on real hardware, full route-level analytics, deduplicating the same pothole seen by multiple buses) are designed for, but not fully built yet
- Detection for occlusion and bad weather is *meaningfully improved* by our training approach, not *solved* — that's genuinely still an open problem in computer vision, not something we're claiming to have cracked
- Coverage is only as good as the bus network's coverage — areas with no bus service are outside what this system can see, by design

## Team

Built by **SixBytes** for Smart India Hackathon 2026.
