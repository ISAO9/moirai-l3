# Publishing the repository and getting DOIs

You need **two** DOIs, from two services:

1. **Zenodo DOI** — archives this code/weights; goes in the paper's Data Availability,
   `README.md`, `CITATION.cff`, and `.zenodo.json`.
2. **EarthArXiv DOI** — the preprint (the PDF in `paper/`); goes on the preprint's front
   page and is declared to GJI at submission.

Nothing here is created automatically — follow the steps below (each takes a few minutes).

---

## A. Push the repository to GitHub

    cd moirai-l3
    git init -b main
    git add .
    git commit -m "MOIRAI L3: code, preprint and reproducible figures"
    # create an empty repo named 'moirai-l3' on github.com first, then:
    git remote add origin https://github.com/<user>/moirai-l3.git
    git push -u origin main

Then replace every `<user>` placeholder (in `README.md`, `CITATION.cff`, `.zenodo.json`,
and the manuscript) with your GitHub username/org.

**Large files (weights, HDF5) do not go in git** — attach them to the GitHub Release
and/or the Zenodo record (see below). `.gitignore` already excludes them.

---

## B. Mint the Zenodo DOI

You can get the DOI **before** finalizing the paper (recommended), or let a GitHub release
mint it.

### Option 1 — reserve the DOI first (so it can appear in the paper)
1. Sign in at https://zenodo.org (log in with GitHub or ORCID).
2. **New upload** → click **Reserve DOI** (Zenodo shows a DOI like `10.5281/zenodo.XXXXXXX`).
3. Fill metadata (Zenodo can import from `.zenodo.json`), set license **MIT**, upload a
   zip of this repo and the trained weights.
4. Put the reserved DOI into: `README.md`, `CITATION.cff` (`doi:`), the manuscript's Data
   Availability, and the preprint banner; then click **Publish**.

### Option 2 — GitHub ↔ Zenodo integration (DOI on release)
1. At https://zenodo.org/account/settings/github/ flip the switch **ON** for `moirai-l3`.
2. On GitHub, **Create a new release** with tag `v0.1.0` and attach the weights.
3. Zenodo automatically archives the release and issues a DOI (a version DOI plus a
   permanent "concept" DOI that always points to the latest version — cite the concept DOI).

> Tip: cite the **concept DOI** in the paper so it never goes stale across versions.

---

## C. Post the preprint to EarthArXiv

1. Go to https://eartharxiv.org → **Submit**.
2. Upload `paper/MOIRAI_L3_preprint.pdf`.
3. Metadata: title, author (Isao Kurosawa, IVXA, Japan), abstract (the SUMMARY),
   subject **Seismology / Geophysics**, license **CC BY 4.0**.
4. In comments, note: *Author's Original Version, submitted to Geophysical Journal
   International; not peer reviewed.* Add the Zenodo DOI as a linked resource.
5. On acceptance (usually 1–2 days of moderation) EarthArXiv assigns a **preprint DOI**.
6. Put that DOI into the preprint front-page banner (rebuild) and keep it for the GJI
   submission form.

---

## D. Submit to GJI

GJI permits preprints. At initial submission via ScholarOne you must:
- state that the Author's Original Version is available as a preprint, and
- provide the preprint accession/DOI (the EarthArXiv DOI).

**Do not** deposit the post-review (revised) version anywhere until publication; only the
Author's Original Version may be public during review. On publication, update the EarthArXiv
and Zenodo records with the published article DOI and link.

---

## Checklist

- [ ] `<user>` replaced everywhere
- [ ] Zenodo DOI minted → in README, CITATION.cff, .zenodo.json, manuscript, preprint banner
- [ ] Weights attached to Zenodo record / GitHub release
- [ ] EarthArXiv preprint posted (CC BY 4.0) → preprint DOI in banner
- [ ] GJI submission declares the preprint DOI
- [ ] (on publication) preprint + Zenodo updated with the published DOI
