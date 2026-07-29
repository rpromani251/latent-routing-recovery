# Data dependencies

Two raw inputs, neither checked into git (see `.gitignore`).

## 1. Seattle Building Energy Benchmarking data

`data/raw/seattle_building_energy_benchmarking.csv`

Seattle's annual building-energy disclosure dataset ("Building Energy Benchmarking Data,
2015-Present") from the [Seattle Open Data portal](https://data.seattle.gov). The portal
updates this dataset yearly and its export URL isn't stable, so it's not hardcoded
anywhere in this repo:

```bash
python scripts/download_seattle_data.py --url "<export URL from the dataset's Export panel>"
```

or download it by hand from the portal and place it at the path above.

Expected columns (used by `src/detect_recover_interpret/model.py:load_buildings`): `OSEBuildingID`,
`Latitude`, `Longitude`, `SiteEUIWN(kBtu/sf)`, `PropertyGFATotal`, `YearBuilt`,
`NumberofFloors`, `NumberofBuildings`, `LargestPropertyUseType`, `Neighborhood`,
`DataYear`, and optionally `Demolished`.

## 2. King County tract demographics

`data/processed/king_county_tracts_demographics.geojson`

Census tract geometry (TIGER/Line) joined with ACS 2011-2015 B03002 demographics
(`pct_black_latino` per tract), used to define the gate `b(x) >= 0.25`. This is shared
infrastructure with the sibling
[`geospatial-xai-attacks`](../../geospatial-xai-attacks) repo, which derives it from raw
Census/TIGER inputs via `scripts/process_census.py`.

To get it:

```bash
# from a checkout of geospatial-xai-attacks with data/raw/ and data/processed/ populated
cp ../geospatial-xai-attacks/data/processed/king_county_tracts_demographics.geojson \
   data/processed/king_county_tracts_demographics.geojson
```

or regenerate it there first if that repo's `data/processed/` is empty (see its README's
"Reproduce" section).

`legacy/diag_model_class.py` has an additional external-data dependency of its own — see
its docstring.
