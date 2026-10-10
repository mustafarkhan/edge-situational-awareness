
## seadronessee / test: 1200 images, 6825 boxes

| class | boxes | share |
|---|---|---|
| swimmer | 4221 | 61.8% |
| boat | 1361 | 19.9% |
| life_saving_appliances | 521 | 7.6% |
| buoy | 431 | 6.3% |
| jetski | 291 | 4.3% |

Objects per image: mean 5.7, max 16, images with none: 0

sqrt(area) in original pixels: median 20.4, p10 10.5

| model input | median object size | < 8 px | < 16 px | < 32 px (COCO 'small') |
|---|---|---|---|---|
| 640 | 7.1 px | 59.3% | 87.3% | 96.7% |
| 1280 | 14.1 px | 13.4% | 59.3% | 87.3% |

figure -> /home/claude/edge-situational-awareness/docs/figures/seadronessee_test_stats.png
summary -> /home/claude/edge-situational-awareness/benchmarks/results/seadronessee_test_stats.json
