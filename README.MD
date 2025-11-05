# DAS Project - Meeting Notes

## Project Overview
This project focuses on anomaly detection and instance segmentation for identifying objects in data.

### Phase 1: Anomaly Detection (Priority)
- Implement instantaneous anomaly detection with instance segmentation
- **This is the core focus** - implement as quickly as possible
- Investigate parallelization opportunities for performance optimization
- Computer Vision approach is acceptable

### Phase 2: Clustering
- Apply clustering model to detected anomalies
- Consider HDBSCAN for clustering algorithm
- Assumption: Objects within the same cluster are the same object (semi-manual identification)

### Phase 3: Identification (Optional)
- Run identification model directly after anomaly detection
- Implement self-identification if feasible

## Summary
1. Detect as many objects as possible
2. Cluster the detected objects
3. Implement self-identification (if possible)

## Notes
- Simple classification can be performed after initial detection
- Identification model can run as step 2 immediately after anomaly detectionÆR OBS PÅ AT DATA IKKE KOMMER MED OP. EN RODET OMGANG AF Resumé

## FRA GUSTAV

Jeg har prøvet at replikere i CAE.ipynb den model eller approachd e bruger i Label-Free Anomaly Detection. jeg skal bruge noget ground truth data, altså data hvor ingen skibe findes for virkelig at kunne få det op at virke. Jeg kan beskrive resultater senere.

Planen er også at implementere en GRAPH DFNN ved nærmere lejlighed.