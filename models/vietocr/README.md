# VietOCR local assets

FoodHealthQA loads VietOCR from the local configuration in this directory.

- `vgg_transformer.json` is tracked with the source code.
- `vgg_transformer.pth` is the pretrained model (about 152 MB) and is ignored by Git.
- If the model is missing, the application downloads it once from the official
  VietOCR URL, validates its size and SHA-256, then stores it here permanently.
- Set `VIETOCR_WEIGHTS_PATH` to use a different persistent model location.

After the model has been downloaded, VietOCR initialization and inference work
without network access.
