# WSTS Reproduction Notes

## 1. Baseline identification

- Repository: https://github.com/SebastianGer/WildfireSpreadTS
- Pinned commit: f62dc49f2c2a6d249c071533ae5340c0d01bfb95
- Dataset record: https://zenodo.org/records/8006177
- Dataset license: CC-BY-4.0

## 2. Dataset download and extraction

- Archive: WildfireSpreadTS.zip
- Download completed successfully.
- Published MD5 checksum was verified against the downloaded archive.
- Archive extraction completed.
- Verified archive TIFF count: 13,607.
- Verified extracted TIFF count: 13,607.
- Missing extracted TIFFs: 0.
- Unexpected extracted TIFFs: 0.
- Extracted years: 2018, 2019, 2020 and 2021.
- Raw data is stored in Google Drive and is not committed to Git.

An earlier inventory reported 14,218 files. This count could not
be reconciled with the archive. The verified archive and extracted
path comparison establishes 13,607 TIFF files as the authoritative
count for this reproduction record.

## 3. Storage observations

- Windows machine free space after cleanup: 205.27 GB.
- Colab runtime filesystem: 108 GB total, 89 GB available at the time inspected.
- Google Drive and Colab runtime storage are separate storage locations.

## 4. Official HDF5 conversion instructions

The baseline README documents:

    python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR

The README states that HDF5 files take approximately twice as much space but enable faster training.

The README also permits TIFF loading by setting:

    --data.load_from_hdf5=False

## 5. Conversion script inspection

Inspected:
src/preprocess/CreateHDF5Dataset.py

Verified behavior:
- Processes years 2018, 2019, 2020 and 2021.
- Uses FireSpreadDataset.
- Calls get_generator_for_hdf5().
- Writes one HDF5 file per fire under year directories.
- Stores dataset under the HDF5 key 'data'.
- Stores year, fire_name, img_dates and lnglat as attributes.
- Skips an HDF5 file if it already exists.
- Sets HDF5_USE_FILE_LOCKING to FALSE.

## 6. Conversion decision

HDF5 conversion was inspected but not executed during this Phase 1 inspection.
The official TIFF loader was used successfully for sample inspection.
No HDF5 conversion duration, output size or conversion errors are claimed.

## 7. Local baseline modification

The pinned baseline was locally modified in:
src/dataloader/FireSpreadDataset.py

The compatibility patch replaces the unavailable T_co import in the current PyTorch environment with a TypeVar definition.
The patch is recorded separately in:
docs/wsts_baseline_local_patch.diff

Pinned upstream commit remains:
f62dc49f2c2a6d249c071533ae5340c0d01bfb95

## 8. Sample inspection

- Event: fire_21458798
- Year: 2018
- Input TIFF example: 2018-01-06.tif
- Next-day target TIFF: 2018-01-07.tif
- Original raster dimensions: 304 x 247
- Raw bands: 23
- Raw dtype: float32
- CRS: EPSG:32610
- Resolution: 375 m x 375 m
- Loader input shape: (1, 40, 288, 224)
- Loader input dtype: torch.float32
- Loader target shape: (288, 224)
- Loader target dtype: torch.int64
- Verified target positive pixels for the inspected pair: 10

## 9. Remaining Phase 1 coordination and verification

- Review the 23-channel data dictionary against the dataset creation source.
- Complete the temporal leakage and operational availability audit.
  Unknown values remain explicitly marked in docs/leakage_audit.csv.
- Coordinate event footprints, projected coordinates and grid conventions
  with Role 4.
- Provide event metadata, event years and AOI information to Role 6
  for the HLS historical coverage feasibility audit.
- Confirm whether daily raster grids remain consistent within each event.
- Treat the sample manifest as one verified loader sample, not a
  dataset-wide sample validation.


## Archive and Extraction Integrity Verification

The original `WildfireSpreadTS.zip` archive was compared against
the extracted dataset directory.

- Archive TIFF count: 13,607
- Extracted TIFF count: 13,607
- Missing extracted TIFFs: 0
- TIFFs not present in archive: 0
- Unique archive paths: 13,607
- Unique extracted paths: 13,607

Path-level comparison was performed after normalizing paths
to their year-level dataset structure.

All four years (2018–2021) had zero missing TIFF files.

An earlier inventory recorded 14,218 files. This count could not
be reconciled with the original archive. The archive-level and
path-level verification establishes 13,607 TIFF files as the
verified count.

No additional extraction or download was required.
