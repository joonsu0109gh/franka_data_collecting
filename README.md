# Franky Data Collecting with Xbox Controller

This repository provides tools for collecting data with the FR3 robot using an Xbox controller, specifically for OpenVLA execution.

## Usage

### 1. Collect Data

``` shell
python collect_incremental.py
```

### 2. Convert to `.npy` Format for RLDS Transformation

``` shell
python transform_my_data_to_rlds.py --data-path {absolute path to data, e.g., ./my_robot_data/0000}
```

### 3. Move Transformed Data

Move the transformed data to
`./panda_datacollect/rlds_dataset_builder/my_dataset` and rename the
directory to `data`.

### 4. Navigate to Dataset Directory

``` shell
cd rlds_dataset_builder/my_dataset
```

### 5. Transform to RLDS Format

``` shell
conda activate rlds_env
tfds build --overwrite
```

### 6. Location of Saved Dataset

The transformed dataset will be stored at:

    ~/tensorflow_datasets/<name_of_your_dataset>

### 7. Fine-tune OpenVLA Model

You can now use the dataset to fine-tune the OpenVLA model.

### 8. Debug Action Data

Control with calculated action data for debugging:

``` shell
python debug_action_rerun.py --episode-path {e.g., ~/my_robot_data/0004_mydata/train/episode_3.npy}
```

------------------------------------------------------------------------

### References

-   [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder)