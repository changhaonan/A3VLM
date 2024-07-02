# Data generation

A3VLM generates the instruction-following data from `PartNet-Mobility`. The first step is download the raw `PartNet-Mobility` data from [https://sapien.ucsd.edu/downloads](https://sapien.ucsd.edu/downloads).


## Fix the URDF from PartNet-Mobility

The raw URDF provided by PartNet-Mobility can not be loaded by some URDF loaders. Thus, we need to fix the URDF files first.

```python
python urdf_fixer.py --data_dir=${PartNet-Mobility_path}
```

## Render Images

Then we can render the images from URDF. More options can be found at `render_robot_pyrender.py`.

```python
python render_robot_pyrender.py --data_name="all" --data_dir=${PartNet-Mobility_path} --output_dir=${output_path}
```

You can control how many files to generate using the variable `--num_poses` and `num_joint_value`. `num_joint_value` decide how many different configurations do we generate. And `num_poses` decide for each different configuration, how many different images from different angles do we generate. For fast sanity checking, you can set `num_poses=1` and `num_joint_value=1`.

## Generate 3D annotations

After we finish the rendering results, we then generate the 3d annotation results.
```python
python point_render.py --data_name="all" --data_dir=${PartNet-Mobility_path} --output_dir=${output_path}
```

## Generate labels

Finally, we can generate the instruction-following data.

```python
python partnet_label.py --data_name="all" --data_dir=${PartNet-Mobility_path} --output_dir=${output_path}  --vqa_tasks_folder=${--vqa_tasks_folder}
```

## Check the results

You can browse the output folder to check the annotation result. For a specific object, for example 149, the annotation visualization is saved in `${output_path}/149/raw_images/visual_images`.

For the following how to use these data, please check the model part of this repo.

# Trouble-shooting

1. I get error when running `render_robot_pyrender.py`, such as `Error in 7138: string is not a file: partnet_mobility/7138/textured_objs/new-8.obj`. This is normal, as there are some parts missing in the original PartNet mobility dataset. There can be also be `Error in 102278: ('Shader compile failure (0): b\'0(337) : error C1503: undefined variable "uv_0"\\n\'',`, for some texture in the original PartNet mobility dataset can not be correctly loaded into pyrender. These errors only account for a small proportion of data. Around 35 objects have these issues in all 2347 objects.

2. I get `Skip 10090 since not all files exist` when I am running `point_render.py`. This is because there are some errors happened in last step.
