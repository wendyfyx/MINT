# Macrostructure-Informed Normative Tractometry (MINT)

## Introduction
> Paper: [*Microstructural Mapping of Neural Pathways in Alzheimer’s Disease using Macrostructure-Informed Normative Tractometry*](https://doi.org/10.1101/2024.04.25.591183) (published in Alzheimer's \& Dementia)

MINT is a normative tractometry framework for mapping microstructural and macrostructural abnormalities in brain white matter tracts (see [twitter thread](https://x.com/wendyfyx/status/1785538128620769397)).

## Requirement
1. Create environment with `environment.yml` (conda) or `requirements.txt` (pip) file
2. Download the [HCP-842 atlas](https://linkinghub.elsevier.com/retrieve/pii/S1053811918304324)
    ```
    dipy_fetch bundle_atlas_hcp842
    ```

## Preparing Input Data
MINT only supports analyzing bundles registered to the MNI space, to establish correspondence across subjects during model training. To run MINT, DTI must be in the same space as tractograms. If they are both in the native space, you can follow these steps used in BUAN with DIPY CLI:
1. `dipy_slr` to register your tractograms to the HCP-842 atlas (`~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk`)
2. `dipy_recobundles` to extract bundles from the registered tractogram in the MNI space (`/rec_bundles`). Specify the `--out_recognized_labels` to save the streamline labels.
3. `dipy_labelsbundles` to extract the same bundles from tractograms in the native space (`/org_bundles`), using label files generated from step 2. These bundles will be used for bundle profiling of DTI measures.

## Tutorials
Example PPMI data from the Bundle Analytics (BUAN) demo in DIPY can be downloaded [here](https://nih.figshare.com/articles/dataset/BUAN_Demo_-_DIPY_Processed_Parkinson_s_Progression_Markers_Initiative_PPMI_Data_Derivatives/14183108) to be used with the provided bash scripts in `/scripts`.
1. **`step1_prep_h5_data.sh`**  
    Create .h5 file of concatenated streamline coordinates and bundle profiles of DTI metrics, for a given collection of subjects. Each bundle is stored as an numpy array of size Nx128x7 (N streamlines x 128 points per streamline x 7 features, including x, y, z, FA, MD, RD, AxD), with the key **subj/bundle** (e.g. s0001/AF_L). The `data.fetcher.DataFetcher` class can be used to access data from .h5 files. 
2. **`step2_make_json_configs.sh`**  
    Creating train/test split, and json configs for model training. Each key in the JSON config is in the format **h5_path/subj/bundle**, and the value is the number of streamlines for the given bundle. Subjects from multiple .h5 files may be used in one config if you wish to train on multiple datasets.
3. **`step3_pretrain_vae.sh`**  
    Pretrain model from scratch given JSON config file created in step 2. Model settings provided in this script is the same as the model from the paper. This step is optional if you wish to use the provided pretrained model.
4. **`step4_finetune_vae.sh`**  
    Fine-tune model given pretrained model and JSON config file. The model from the paper pretrained on TractoInferno is available in `/assets/tractoinferno_pretrain_24000steps.ckpt`. Currently, all VAE parameters are modifiable during fine-tuning and we do not freeze any component.
5. **`step5_inference.sh`**  
    Run inference to save along-tract metrics (MAE, mean of original data, mean of reconstructed data). Specify `output_z_path` and `output_recon_path` to also save the latent embeddings and reconstruction. 


## Cite our work 
```
@article{feng_microstructural_2024,
	title = {Microstructural mapping of neural pathways in {Alzheimer}'s disease using macrostructure‐informed normative tractometry},
	copyright = {All rights reserved},
	issn = {1552-5260, 1552-5279},
	url = {https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.14371},
	doi = {10.1002/alz.14371},
	language = {en},
	urldate = {2025-01-07},
	journal = {Alzheimer's \& Dementia},
	author = {Feng, Yixue and Chandio, Bramsh Q. and Villalon‐Reina, Julio E. and Thomopoulos, Sophia I. and Nir, Talia M. and Benavidez, Sebastian and Laltoo, Emily and Chattopadhyay, Tamoghna and Joshi, Himanshu and Venkatasubramanian, Ganesan and John, John P. and Jahanshad, Neda and Reid, Robert I. and Jack, Clifford R. and Weiner, Michael W. and Thompson, Paul M. and {for the Alzheimer's Disease Neuroimaging Initiative}},
	month = dec,
	year = {2024},
	pages = {alz.14371},

```
