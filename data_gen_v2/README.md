# Data generation V2

## Multi-Scan

1. First run pre-process.
```
python pre_multiscan.py
```

2. Run generation.
```
python multiscan_label.py
```

## PartNet

1. Run pre-process.
```
python pre_partnet_render.py
```

2. Run render.
```
python partnet_render.py
```

3. Do SD augmentation (Optional)
```
python data_augment_sd.py
```