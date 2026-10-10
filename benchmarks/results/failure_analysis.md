# Failure analysis

Model `ft_y11n_ep3.pt`, input 1280 px, confidence 0.25, 548 VisDrone val images, 38,759 labelled objects. Found = any predicted box with IoU >= 0.5 (class-agnostic).

**Overall recall: 49.5%**

## By object size (in model-input pixels)

| size     |   objects | recall   |
|:---------|----------:|:---------|
| < 8 px   |      2210 | 9.8%     |
| 8-16 px  |      9720 | 26.3%    |
| 16-32 px |     15154 | 49.4%    |
| 32-64 px |      8743 | 72.8%    |
| >= 64 px |      2932 | 86.9%    |

## By occlusion (VisDrone annotation)

| occlusion       |   objects | recall   |
|:----------------|----------:|:---------|
| none            |     16810 | 63.1%    |
| partial (1-50%) |     18863 | 41.7%    |
| heavy (50-100%) |      3086 | 22.6%    |

## By class

| class       |   objects | recall   |
|:------------|----------:|:---------|
| person      |     13969 | 33.9%    |
| two-wheeler |      7750 | 23.4%    |
| vehicle     |     17040 | 74.1%    |

## By lighting (mean image brightness < 70 of 255 = dusk/night; checked visually)

| light      |   objects | recall   |
|:-----------|----------:|:---------|
| daylight   |     37612 | 49.5%    |
| dusk/night |      1147 | 48.7%    |


| light      |   images |    gt |   preds |   false_pos | precision   |
|:-----------|---------:|------:|--------:|------------:|:------------|
| daylight   |      531 | 37612 |   26531 |        7920 | 70.1%       |
| dusk/night |       17 |  1147 |    1266 |         707 | 44.2%       |

## Example failure images

Green = found, red = missed, yellow = false positive.

- `docs/figures/failures/0000153_01801_d_0000001.jpg` — recall 13% of 55 objects, 4 false positives, brightness 145
- `docs/figures/failures/0000021_00000_d_0000001.jpg` — recall 13% of 31 objects, 3 false positives, brightness 97
- `docs/figures/failures/0000153_01201_d_0000001.jpg` — recall 15% of 59 objects, 7 false positives, brightness 136
- `docs/figures/failures/0000273_05601_d_0000466.jpg` — recall 16% of 32 objects, 3 false positives, brightness 156
- `docs/figures/failures/0000055_00714_d_0000110.jpg` — recall 66% of 85 objects, 101 false positives, brightness 26
