# Data Licensing, Redistribution Terms, and Git History Scrubbing

**Author:** Deepanshu <Deepanshuop@gmail.com>  
**Status:** Policy & Compliance Specification  
**Applicability:** All datasets (DEAR, MIT-Adobe FiveK, AVA, KonIQ-10k) and repository assets.

---

## 1. Upstream Dataset Licenses & Redistribution Restrictions

### 1.1 MIT-Adobe FiveK / DEAR Dataset
- **Upstream License**: The MIT-Adobe FiveK dataset is distributed by Adobe Systems Inc. and the Massachusetts Institute of Technology strictly for academic and non-commercial research purposes.
- **Redistribution Constraint**: The license prohibits direct redistribution of raw image files, uncompressed DNGs, or full-resolution photographic renders without prior written consent.
- **Compliance Policy in AesFix**:
  1. Only lightweight numerical manifests, pre-registered split hashes (`manifest_sha256`), and derived tabular metrics are tracked in version control.
  2. Full-resolution images, intermediate rendered pairs, and human study candidate crops must be stored outside the git working tree (e.g., in `D:\aesfix-data\` or cloud scratch storage).

---

## 2. Git History Scrubbing Requirement

### 2.1 Audit Finding & Remediation
During Phase 8 development, binary image assets (20 pairs of overexposure evaluation candidates in `results/human_study_candidates/` and 6 test renders in `results/visuals/`) were temporarily committed to the local repository. 

To maintain strict compliance with data licensing constraints and prevent repository bloat:
1. **Immediate Action**: All binary image files under `results/visuals/*.jpg` and `results/human_study_candidates/` have been removed from the git index (`git rm --cached`) and permanently added to `.gitignore`.
2. **Pre-Release History Scrubbing**: Prior to public release or code publication accompanying submission, repository history must undergo automated rewriting using `git-filter-repo`:
   ```bash
   git filter-repo --path results/human_study_candidates/ --invert-paths
   git filter-repo --path results/visuals/ --invert-paths
   ```
   This completely purges the historical blobs from git commit objects.

---

## 3. Storage and Cache Management Guidelines

1. **Local Disk Footprint**:
   - Drive `D:\aesfix-cache\` holds model weights and temporary preprocessor caches.
   - SigLIP-SO400M must be retained strictly in its vision-only fp16 format (`~856 MB`), preventing duplicate multi-gigabyte HuggingFace hub downloads.
   - Target free space on Drive `D:` must remain $\ge 5.0\text{ GB}$ at all times.
2. **Kaggle Cloud Synchronization**:
   - Downloads from cloud execution kernels must strictly target JSON summary manifests, evaluation tables (CSV), and figure PNGs.
   - Cloud runs must not write large visual image batches to `/kaggle/working/` to prevent bandwidth and quota exhaustion.
