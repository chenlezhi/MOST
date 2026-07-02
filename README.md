# MOST

This is the official implementation for our paper:"MOST: Modularity-guided Domain Identification in Spatial Transcriptomics Data"
---

## Experiments

Please replace the following `{dataset_name}` with the actual dataset name.

### Setup

```bash
conda create -n MOST python=3.10
conda activate MOST
```

### Data

**To generate DLPFC data:**

1.  Set the `datasets` variable in `DLPFC_generate_data.py` to your target slice number (e.g., `151507`)
2.  Execute the generation script:

```bash
python {dataset_name}_generate_data.py
```

#### For All Other Datasets

Direct execution (no configuration needed):

```bash
python {dataset_name}_generate_data.py
```

### Train

```bash
python {dataset_name}_test.py
```


## Datasets

You can download the benchmark datasets from the links below:

+ DorsoLateral PreFrontal Cortex (DLPFC): [LieberInstitute/spatialLIBD: Code for the spatialLIBD R/Bioconductor package and shiny app](https://github.com/LieberInstitute/spatialLIBD)
+ Human Breast Cancer (HBC):  [Human Breast Cancer (Block A Section 1) - 10x Genomics](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0)
+ Mouse Anterior Brain (MAB):  [Mouse Brain Serial Section 1 (Sagittal-Anterior) - 10x Genomics](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0)

