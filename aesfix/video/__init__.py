"""Video/temporal aesthetic assessment track (BITE314L course project).

Reuses the Phase-0 scoring backbone (CLIP scorer + fuzzy fusion) from
baselines.legacy_fuzzy.our_system unchanged, applied per sampled frame.
Separate from aesfix/data (dataset loaders for the image-critique research
track) since this is orchestration code, not a dataset loader.
"""
