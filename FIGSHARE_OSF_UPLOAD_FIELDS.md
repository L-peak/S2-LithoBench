# Figshare / OSF upload fields (English, copy and paste)

Both services were reachable from the author's network in this test run
(Figshare and OSF returned HTTP 200/202), and both can mint DOIs.

## Title

S2-LithoBench v1.0: code and derived data for a validation-aware Sentinel-2
lithological mapping benchmark

## Author

Zhaoyang Li, Henan Polytechnic University, Jiaozuo, Henan 454000, China
(3286533389@qq.com)

## Category

Earth and environmental sciences / Geoscience / Remote sensing

## Description

This deposit contains the complete reproducible pipeline and the derived data
for the S2-LithoBench benchmark. The benchmark compares four nested feature
sets (spectral bands; spectral indices; terrain attributes; local texture),
six classifiers and two validation protocols (stratified random and spatially
blocked five-fold cross-validation), followed by a robustness battery
(ten blocking realizations, buffered validation, block-size sensitivity,
geological-unit GroupKFold, class-balanced and polygon-capped sampling,
majority-class and label-shuffled controls, and XGBoost) for lithological
mapping in two arid Australian study areas. All source data are open:
Sentinel-2 L2A (Earth Search STAC, Element 84/AWS), the Geoscience Australia
1:1,000,000 surface-geology map (CC BY 4.0) and Copernicus DEM GLO-30 (AWS).

Contents: eight ordered Python scripts; sample tables with all extracted
features (n = 2,014 and n = 1,278); metric tables, confusion matrices,
out-of-fold predictions and permutation importances; repeated spatial-CV
results with Wilcoxon/Holm tests; buffer, block-size, GroupKFold, balanced
sampling and control results; transfer confidence intervals; run logs and
manuscript figures (PNG + PDF). All random seeds are fixed (SEED = 42); the
full pipeline runs in about 70 minutes on a 16-core desktop without GPU.

Key findings: random validation inflated macro-F1 by 0.16-0.41 relative to
spatial validation; terrain attributes produced the only large incremental
gain (+0.079/+0.094; terrain-only features reached 0.199/0.279); spectral
indices and 3x3 texture contributed at most 0.03; holding out entire
geological units reduced macro-F1 to 0.091/0.109, with label-shuffled controls
at 0.145/0.160; cross-region transfer remained near chance.

## Keywords

lithological mapping; spatial cross-validation; benchmark; Sentinel-2;
reproducibility; model transferability; arid Australia

## License

Code: MIT. Derived data: CC BY 4.0.
