# LIDC-IDRI Data Loading Utilities

This repository provides utilities for working with the **LIDC-IDRI (Lung Image Database Consortium / Image Database Resource Initiative)** dataset. It supports loading imaging series (CT/DX/CR) and parsing XML annotation files containing radiologists’ markings. Each case contains one or more thoracic CT scans.

The dataset is available from [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). It contains thoracic CT scans in DICOM format and associated XML annotations produced by four experienced radiologists. Annotations include nodules (≥ 3mm) and non-nodules (< 3mm). Each patient may have multiple reading sessions corresponding to different radiologists.

A typical directory structure looks like:

data/raw/chest/manifest-XXXXXXXXXXXX/
└── LIDC-IDRI-xxxx/
└── xx-xx-xxxx-NA-NA-xxxxx/
└── xxxxxxx.xxxxxx-NA-xxxxx/
├── *.dcm
└── *.xml


DICOM files contain CT slices, while XML files contain the corresponding annotations.

The main functions provided are:

- **`load_data_chest(subject_id, modality, choose="largest", index=None)`**  
  Loads a patient’s imaging series.  
  Parameters:  
  - `subject_id`: patient ID (e.g. `1`, `"0001"`, `"LIDC-IDRI-0001"`)  
  - `modality`: `"CT"`, `"DX"`, `"CR"`  
  - `choose`: selection rule (`"largest"`, `"first"`, `"index"`)  
  - `index`: index to use if `choose="index"`  
  Returns: `(volume, meta_dict, voxel_spacing)`

- **`load_xml_info(subject_id, modality="CT")`**  
  Parses XML annotations for a given patient. Returns an `XmlAnnotations` object containing both nodules and non-nodules.

Annotations are returned in dataclasses:

```python
@dataclass
class Roi:
    z: float
    sop_uid: str
    inclusion: Optional[bool]
    contour: List[Tuple[float, float]]

@dataclass
class NoduleAnn:
    nodule_id: str
    diameter_mm: Optional[float]
    malignancy: Optional[int]
    characteristics: dict
    rois: List[Roi]
    xml_file: str

@dataclass
class NonNoduleAnn:
    non_nodule_id: str
    sop_uid: str
    x: float
    y: float
    z: float
    xml_file: str

@dataclass
class XmlAnnotations:
    nodules: List[NoduleAnn]
    non_nodules: List[NonNoduleAnn]

usage examples:

from data_load_chest import load_data_chest, load_xml_info

# Load CT series
vol, meta, spacing = load_data_chest("1", "CT")

# Load annotations
anns = load_xml_info("1", "CT")

print("Number of nodules:", len(anns.nodules))
print("Number of non-nodules:", len(anns.non_nodules))

# Inspect one nodule
if anns.nodules:
    n0 = anns.nodules[0]
    print("\n=== Nodule Example ===")
    print("Nodule ID:", n0.nodule_id)
    print("Malignancy:", n0.malignancy)
    print("Diameter (mm):", n0.diameter_mm)
    print("Characteristics:", n0.characteristics)
    print("Number of ROIs:", len(n0.rois))
    if n0.rois:
        roi0 = n0.rois[0]
        print("ROI z:", roi0.z, "SOP UID:", roi0.sop_uid, "Points:", len(roi0.contour))

# Inspect one non-nodule
if anns.non_nodules:
    m0 = anns.non_nodules[0]
    print("\n=== Non-Nodule Example ===")
    print("Non-nodule ID:", m0.non_nodule_id)
    print("Coordinates (x,y,z):", (m0.x, m0.y, m0.z))
    print("SOP UID:", m0.sop_uid)


Sample output:

Number of nodules: 13
Number of non-nodules: 13

=== Nodule Example ===
Nodule ID: Nodule 001
Malignancy: 5
Diameter (mm): 8.2
Characteristics: {'subtlety': 3, 'margin': 2, ...}
Number of ROIs: 6
ROI z: -125.0 SOP UID: 1.3.6.1.4.1.14519... Points: 19

=== Non-Nodule Example ===
Non-nodule ID: NN_1
Coordinates (x,y,z): (396.0, 216.0, -185.0)
SOP UID: 1.3.6.1.4.1.14519...
