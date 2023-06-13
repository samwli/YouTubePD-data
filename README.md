# YouTubePD-data
Construct and preprocess video dataset for Parkinson's disease (PD) from YouTube. Please set up the instructions in `PDRegionExtraction` for extracting PD-informative facial regions--bounding boxes are interpolated by modifying `MMPose` framework. The region extraction was run on a single V100 GPU. The resulting bounding box coordinates are provided, so this step may be skipped. In the `data_sheets` folder, two datasheets are provided. The first datasheet downloads a balanced PD positive/negative dataset, and the second datasheet downlaods only PD-negative videos to expand the dataset. Select the datasheet you want to download from, and construct and preprocess the dataset using `prepare_data.py`.

The region-level binary annotations are provided in a dictionary format in `region_video_annotations.pkl`, to be loaded into the model. Each video is a key in the dictionary; the corresponding value is an array of strings. Each string is in the format `annotated_frame_num/total_frame_num, region_index, severity, confidence`. The training code directly uses the file in this format to load in region annotations.

The train and test data splits, along with the video level annotations, are also provided as csv files.

Examples of the processed videos can be found in the `processed_examples` folder. Example videos of two patients are provided, with and without visualized bboxes for each.
