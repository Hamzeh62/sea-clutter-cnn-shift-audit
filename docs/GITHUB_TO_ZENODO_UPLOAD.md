> **2026-11-06 author confirmation:** MIT for author-owned code and CC BY 4.0 for author-owned results, figures and CNN checkpoints are approved; the publication and DOI fields remain pending until actual release. Original raw IPIX recordings remain excluded.

# Suggested publication sequence

1. Obtain author approval for rights/licensing, and verify contents of release staging files.
2. Upload the contents of `PAPER2_GITHUB_REPOSITORY.zip` to a *new* GitHub repository; do not upload the ZIP as the only repository file if the intent is browsable version-controlled code.
3. Choose whether to use GitHub's release archiving integration with Zenodo or directly upload the `PAPER2_ZENODO_UPLOAD_FILES.zip` contents. For this project, **direct Zenodo deposit is safer** because the research data/weights are preserved as real binary ZIP archives, not Git LFS pointers.
4. Open Zenodo and create a NEW upload draft. Upload the unpacked *outer* package: `PAPER2_SOFTWARE_AND_FIGURES_v1.0.0.zip`, `PAPER2_CHECKPOINTS_v1.0.0.zip`, `PAPER2_NUMERICAL_RESULTS_v1.0.0.zip`, `PAPER2_MANUSCRIPT_WORKING.pdf`, `PAPER2_METADATA_AND_VALIDATION.zip`.
5. Supply author, title, description, actual license, version, keywords, record type. Confirm all rights and then click Publish yourself.
6. After Zenodo registers a real DOI and GitHub URL exists, update `CITATION.cff`, README and the manuscript data/code availability statement, and create a final committed release tag.

Do not invent a DOI or claim the paper is already published. Do not present a draft reserved DOI as a registered published one.
