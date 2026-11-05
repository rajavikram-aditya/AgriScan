# AgriScan 🌱

AI-powered crop disease detection from a single leaf photo. Upload a leaf, get the disease, how severe it is, and a visual explanation of why the model decided that.

> **Status:** The ML pipeline (training, classification, severity estimation, Grad-CAM) is complete. The React web interface is a front-end demo; connecting it to the trained model needs a hosted inference API (see [Roadmap](#roadmap)).

## The problem

Up to 40% of global crop production is lost to pests and diseases every year, and plant diseases cost the global economy over $220 billion annually (FAO). Much of that loss happens in the gap between a farmer noticing a problem and someone identifying it. Most tools give a single label with no reasoning, so there is nothing to trust or double-check.

## What AgriScan does

Leaf image → preprocessing → MobileNetV2 classifier → disease label + confidence → severity estimation (two methods) → Grad-CAM heatmap → treatment info

## Features

- **Disease detection:** MobileNetV2 with transfer learning, fine-tuned on 22 classes of healthy and diseased leaves, ~93% accuracy.
- **Dual severity estimation:**
  - HSV/LAB colour segmentation measures the infected leaf area.
  - Grad-CAM-based estimation uses the model's own attention map.
  - Two independent methods give a cross-check instead of a single number.
- **Explainable output:** Grad-CAM heatmap shows which regions of the leaf drove the prediction.
- **Treatment info:** disease details and recommended next steps.
- **Web interface:** responsive React UI with crop selection, local leaf image preview, camera-friendly input, confidence result view, and accessible interaction states.

## Tech stack

| Layer | Tools |
|---|---|
| Model | Python, TensorFlow/Keras, MobileNetV2 |
| Image processing | OpenCV |
| Web UI | React, TypeScript, Vite, custom CSS |
| Original prototype | Streamlit |

## Run locally

```bash
pnpm install
pnpm dev
```

Open the local URL printed by Vite.

Check and build for production:

```bash
pnpm check
pnpm build
```

## Project structure

```
client/
├── public/assets/        # visual assets used by the interface
└── src/
    ├── pages/Home.tsx    # page UI and lightweight client-side demo behaviour
    └── index.css         # visual system and responsive layout styling
```

## Roadmap

- [ ] Host the TensorFlow model behind an inference API (FastAPI/Flask)
- [ ] Connect the Scan button to the API and show real predictions, severity, and heatmaps
- [ ] Expand crop and disease coverage
- [ ] Offline/mobile inference (TFLite)

## Team

- Aditya Vikram
- Suhani Jain
- Gaurav Verma

**Guided by:** Dr. Supriya Agrawal

Built at SVKM's NMIMS, Mukesh Patel School of Technology Management & Engineering (MPSTME).

## License

Add a license (e.g. MIT) before making the repo public.
