# AgriScan

AgriScan is a professional static HTML/CSS-first interface for a plant health scanning experience. The project replaces the previous Streamlit presentation with a responsive browser UI built with React, TypeScript, Vite, and custom CSS.

## Run locally

```bash
pnpm install
pnpm dev
```

Open the local URL printed by Vite. The production build can be checked with:

```bash
pnpm check
pnpm build
```

## What is included

The static interface includes the AgriScan field-notes visual system, responsive layout, crop selection, local leaf image preview, camera-friendly input, demo scan interaction, confidence result dossier, and accessible interaction states. The current scan result is a front-end demonstration; connecting the button to the TensorFlow model requires a hosted API or a server-side inference service.

## Project structure

- `client/src/pages/Home.tsx` contains the page UI and lightweight client-side demo behavior.
- `client/src/index.css` contains the visual system and responsive layout styling.
- `client/public/assets/` contains the visual assets used by the interface.
