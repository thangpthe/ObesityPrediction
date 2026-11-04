# Raw data sources

## UCI obesity levels dataset

**Expected files**
- `estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip` — original downloaded archive.
- `ObesityDataSet_raw_and_data_sinthetic.csv` — extracted source CSV (keep the source spelling).

**Official source:** UCI Machine Learning Repository, *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*  
https://archive.ics.uci.edu/dataset/544/estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition

**DOI:** https://doi.org/10.24432/C5H31Z  
**Original article:** https://doi.org/10.1016/j.dib.2019.104344  
**License:** CC BY 4.0. Attribute Palechor and de la Hoz Manotas (2019) and the UCI Machine Learning Repository when redistributing or adapting the data.

**Important provenance note:** This file is a mixed raw-and-synthetic dataset. The original publication reports 485 collected records and 77% synthetic observations generated using Weka/SMOTE in the final 2,111-row dataset. The distributed CSV has no row-level original/synthetic indicator. Do not describe all 2,111 records as independently collected real-world observations.

**Integrity**
- CSV SHA-256: `28b6991062855d3d15642e38b1aa9d1b32fb7abbb5f5b096f1868cd706154b`
- ZIP SHA-256: `9cf2ee5e3c3139815c059217a832ff3c0a4ec07bc55b80245f1b94bb3254b297`

Do not edit source files in place. Put transformed data under `data/interim/` or `data/processed/` and regenerate it using version-controlled scripts.
