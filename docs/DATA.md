# Data Layout

The repository does not include the solar dataset. Keep data local or in private object storage, then point the CLI at it with `--dataset-root`.

Supported layout:

```text
Solar Dataset/
  Solar Dataset/
    training/
      low_res/
      high_res/
    validation/
      low_res/
      high_res/
```

Also supported:

```text
dataset-root/
  train/
    low_res/
    high_res/
  val/
    low_res/
    high_res/
```

LR and HR filenames must share the same stem. The loader refuses to pair images by sorted order when names do not match, because that can silently corrupt supervision.

Common image extensions are supported: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.bmp`.
