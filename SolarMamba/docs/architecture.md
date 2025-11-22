# SolarMamba Architecture

## Overview
SolarMamba is a hybrid physical-deep learning model designed for solar irradiance forecasting. It combines a visual encoder (MambaVision) with a temporal encoder (Pyramid TCN) using a gated fusion mechanism.

## Diagram

```mermaid
graph TD
    %% --- STYLING ---
    classDef phys fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef deep fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black;
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef output fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:black;

    %% --- INPUTS ---
    Img[("ASI Image\n(512x512)")]:::deep
    Meta[("PSA Metadata\n(Lat/Lon/Time)")]:::phys
    Hist[("Met History\n(GHI, Temp, Press)")]:::phys

    %% --- PHYSICS ENGINE ---
    subgraph "Physics Engine (pvlib)"
        Calc[Calculate Solar Position\n(SZA, Azimuth)]:::phys
        CS[Calculate Clear Sky GHI\n(Ineichen Model)]:::phys
        Norm[Normalize Target:\nk* = GHI / GHI_cs]:::phys
    end

    %% --- ENCODERS ---
    subgraph "Visual Encoder (MambaVision)"
        S1[Stage 1: Texture]:::deep
        S2[Stage 2: Shape]:::deep
        S3[Stage 3: Global]:::deep
        S4[Stage 4: Semantic]:::deep
    end

    subgraph "Temporal Encoder (Pyramid TCN)"
        T1[Branch 1: Noise]:::phys
        T2[Branch 2: Daily]:::phys
        T3[Branch 3: System]:::phys
        T4[Branch 4: Trend]:::phys
    end

    %% --- GATED FUSION ---
    subgraph "Ladder Fusion (Sigmoid Gating)"
        F1((Gate)):::fusion
        F2((Gate)):::fusion
        F3((Gate)):::fusion
        F4((Gate)):::fusion
    end

    %% --- FLOW ---
    Meta --> Calc & CS
    Hist --> Norm
    CS --> Norm
    
    %% Image Path
    Img --> S1 --> S2 --> S3 --> S4
    
    %% Time Path (Physics Augmented)
    Calc --> T1 & T2 & T3 & T4
    Norm --> T1 & T2 & T3 & T4

    %% Cross Connections (Gating)
    S1 & T1 --> F1
    S2 & T2 --> F2
    S3 & T3 --> F3
    S4 & T4 --> F4

    %% RECONSTRUCTION
    F1 & F2 & F3 & F4 --> Concat[Aggregator] --> PredK([Predict k*]):::output
    PredK & CS --> Recon[Reconstruct GHI:\nGHI = k* × GHI_cs]:::output
    Recon --> Final([Final Forecast]):::output
```

## Components

### 1. Visual Encoder
- **Backbone:** MambaVision-B
- **Features:** Extracted from Stages 1, 2, 3, and 4.

### 2. Temporal Encoder
- **Architecture:** Pyramid TCN
- **Branches:** 4 parallel branches with kernel sizes 3, 5, 7, 9.
- **Input:** 7 channels (k*, Temp, Pressure, SZA, Azimuth, sin_hour, cos_hour).

### 3. Ladder Fusion
- **Mechanism:** Residual Gating.
- **Logic:** The temporal vector is projected and passed through a sigmoid to create a gate. This gate modulates the visual features.
- **Formula:** $V_{out} = V_{in} \cdot \sigma(T_{proj}) + V_{in}$

### 4. Head
- **Structure:** Global Average Pooling -> Concatenation -> MLP -> Sigmoid -> Scale (1.2).
- **Output:** Clear Sky Index ($k^*$).
