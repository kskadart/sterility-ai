# Sterility Detection Service

A computer vision-based service for detecting instrument sterility using a CV model.

## Demo Assets

Sample images to test the service can be found at: [Yandex Disk](https://disk.360.yandex.ru/d/4dmDJqBSVoxnwQ)

## Installation

```bash
uv pip install -e .
```

## Scripts

| Script | Description |
|--------|-------------|
| `streamlit_app.py` | Web application for interactive sterility detection |
| `remove_duplicates.py` | Utility to find and remove duplicate images from dataset |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `model_training.ipynb` | Train and evaluate the sterility detection model |
| `cleanup_dataset.ipynb` | Dataset preprocessing and cleanup utilities |

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - for non-commercial use only.
