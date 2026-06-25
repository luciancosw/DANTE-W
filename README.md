# DANTE-W: Diffuse Albedo Neural Texturing in the Wild

Step1: caching g-buffers by
```bash
python agents/warping.py
```

Step2: training by
```bash
CUDA_VISIBLE_DEVICES=0 python agents/pl_train.py
```

Step3: inference by
```bash
python agents/test.py
```