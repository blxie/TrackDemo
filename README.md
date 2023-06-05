# TrackDemo
track demo


## STEP 1

- initialize

```python
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

- ckpt dirs

```bash
mkdir model_zoo
cp /data/ckpt/... .
mv .pth.tar tracking_engine_1_online_22k.pth.tar
```