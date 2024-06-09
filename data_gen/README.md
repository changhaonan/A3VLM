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

## Generate labels

Finally, we can generate the instruction-following data.

```python
python partnet_label.py --data_name="all" --data_dir=${PartNet-Mobility_path} --output_dir=${output_path}
```