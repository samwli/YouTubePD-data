# YouTubePD-data
Construct and preprocess video dataset for Parkinson's disease (PD) from YouTube. Please set up the instructions in `PDRegionExtraction` for extracting PD-informative facial regions--we interpolate bounding boxes by modifying `MMPose` functionality. Select the datasheet you want to download from, and construct and preprocess the dataset using `prepare_data.py`. The region-level binary annotations are provided in a dictionary format in `region_video_annotations.pkl`, to be loaded into the model.

Examples of intermediate and final keypoint interpolation results can be seen in `demo_results`.
