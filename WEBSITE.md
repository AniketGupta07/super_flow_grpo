# VPS Project Website

Static project page for **Recreating Video Arenas via Automated Preference Scoring (VPS)**.
Lives on the `gh-pages` branch.

## Files
- `index.html` — the page
- `assets/style.css` — styling
- `assets/figures/` — figures (exported from the paper's PDFs/PNGs)

## Preview locally
```bash
python -m http.server 8000
# open http://localhost:8000
```

## Deploy with GitHub Pages
1. Push this `gh-pages` branch to GitHub.
2. Repo **Settings → Pages → Build and deployment**.
3. Source: **Deploy from a branch**; Branch: **`gh-pages`** / **`/ (root)`**.
4. The site publishes at `https://<user>.github.io/<repo>/`.

## Links (current state)
- **Paper** → `assets/paper/vps_paper.pdf` (NeurIPS 2026 E&D camera-ready) ✅
- **Code** → https://anonymous.4open.science/r/super_flow_grpo/ ✅
- **Model** → Hugging Face `best.pth` ✅
- **Dataset** → Hugging Face `supervideoVPS/super_video` ✅
- **arXiv** → disabled placeholder (search `TODO` in `index.html`): set its `href` and
  remove `aria-disabled="true"` when available.

The paper is accepted to the NeurIPS 2026 Track on Evaluations and Datasets; the hero shows the
venue, the full author list and Stanford affiliations, and the BibTeX cites the NeurIPS version.

## Remaining TODO
- **arXiv** button (see above), and optionally add an `eprint` field to the BibTeX.

## Media
- `assets/videos/baseline/` and `assets/videos/finetuned/` hold the WAN-2.1 before/after clips
  (indices 09, 16, 23, 29), shown as autoplaying looped `<video>` pairs in the Qualitative section.
