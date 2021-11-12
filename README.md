# TMLS 2021: Supercharging Experimentation Pipelines
Learn how to speed up your deep learning experiments using PyTorch Lightning and DVC!

The MNIST dataset is used for this example.

## Installation & Setup
```
pip install -U pip==21.2.4
pip install poetry==1.1.11
poetry install --no-interaction
```

Begin demo at:
```
git checkout -b <local-branch-name> start-demo
```

## Docker usage
The latest codebase is packaged and runnable within a Docker image.
```
docker build -t mnist .
```

You can use the container to do one of two use cases:

### Re-training on new data
1. Firstly, update the data for any of `data/MNIST/<set>_images.pt` or `data/MNIST/<set>_labels.pt`
where `<set>` is `train`, `val` or `test`. The image file contains grayscale images of shape `(N, 1, H, W)`
and the label file are also PyTorch tensors of shape `(N,)`.
2. Ensure that a `config.yaml` exists in the repo root. A copy can be obtained from `example_configs/`.
3. Update `config.yaml` as necessary.
4. Reproduce the experiments with DVC.
    ```
    docker run -it \
        -v $(pwd):/opt \
        --entrypoint dvc mnist repro
    ```
    A new model checkpoint is created in `checkpoints/checkpoint.ckpt`,
    and model metrics are available in a newly created file `test_metrics.json`.

### Inference
1. Supply training images in `data/MNIST/predict_images.pt`. It should be a PyTorch tensor of shape `(N, 1, H, W)` (grayscale images).
2. Ensure that you have a `predict_config.yaml` in the repo root. A copy can be obtained from `example_configs/`.
3. Update `predict_config.yaml` if necessary.
4. Perform model inference.
    ```
    docker run -it \
        -v $(pwd):/opt \
        --entrypoint python mnist -m predict_script
    ```
    This loads a model checkpoint specified in `predict_config.yaml` and saves the results to `predictions.pt` in the repo root.
