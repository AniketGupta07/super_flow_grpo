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

## TODO before publishing (search for `TODO` in `index.html`)
- **Authors / affiliations** — currently `Anonymous Authors` (double-blind). De-anonymize when allowed.
- **Links** — the Paper / arXiv / Code / Model / Dataset buttons are disabled placeholders
  (`data-link="..."`, `aria-disabled="true"`). Set each `href` and remove `aria-disabled="true"` once you have:
  - Paper / arXiv PDF
  - Code repository
  - Hugging Face model
  - Hugging Face dataset
- **BibTeX** — finalize the citation block.
