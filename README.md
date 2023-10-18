# YouTubePD-data
Construct and preprocess video dataset for Parkinson's disease (PD) from YouTube. Please see `instructions.md` in `PDRegionExtraction` for extracting PD-informative facial regions--bounding boxes are interpolated by modifying [MMPose](https://github.com/open-mmlab/mmpose) framework. The region extraction can be run on a single V100 GPU. The resulting bounding box coordinates on the YouTube-PD dataset are pre-extracted and included. In the `data_sheets` folder, two data sheets are provided. The first data sheet downloads a balanced PD positive/negative dataset, and the second data sheet downloads only PD-negative videos to expand the dataset. Select the data sheet you want to download from, and construct and preprocess the dataset using `prepare_data.py`.

The dataset also includes region-level annotations in a dictionary format (`region_video_annotations.pkl`) parsed from the data sheets to be loaded into the model. Each video is a key in the dictionary; the corresponding value is an array of strings. Each string is in the format `annotated_frame_num/total_frame_num, region_index, severity, confidence`. The training code directly uses the file in this format to load in region annotations.

The train and test data splits, along with the video level annotations, are included as csv files.

Examples of the processed videos can be found in the `processed_examples` folder. Example videos of two patients are provided, with and without visualized bboxes for each.

## Acknowledgement

Code is largely based on [MMPose](https://github.com/open-mmlab/mmpose).
