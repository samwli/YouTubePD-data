# YouTubePD-data
Construct and preprocess video dataset for Parkinson's disease (PD) from YouTube. Please set up the instructions in `PDRegionExtraction` for extracting PD-informative facial regions--bounding boxes are interpolated by modifying `MMPose` framework. The region extraction was run on a single V100 GPU. The resulting bounding box coordinates are provided, so this step may be skipped. Select the datasheet you want to download from, and construct and preprocess the dataset using `prepare_data.py`.

The region-level binary annotations are provided in a dictionary format in `region_video_annotations.pkl`, to be loaded into the model.

Examples of the processed videos can be found in the `processed_examples` folder. Two example videos are provided, with and without visualized bboxes for each.
