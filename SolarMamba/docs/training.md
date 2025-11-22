# Training Instructions

## Prerequisites
- Python 3.8+
- PyTorch
- pvlib
- pandas
- matplotlib

## Configuration
Edit `SolarMamba/config.yaml` to adjust hyperparameters:
```yaml
training:
  epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 0.01
```

## Running Training
Execute the run script:
```bash
cd SolarMamba
./run.sh
```
Or run manually:
```bash
cd SolarMamba
python train.py
```

## Loss Function
- **Type:** MSE Loss
- **Target:** Clear Sky Index ($k^*$)
- **Reasoning:** $k^*$ is a normalized measure of cloudiness (0-1.2), making it a more stable target than raw GHI (0-1000+).

## Optimization
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Weight Decay:** 0.01
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2)

## Metrics
- **Loss:** MSE on $k^*$.
- **RMSE:** Root Mean Square Error on reconstructed GHI ($W/m^2$).
