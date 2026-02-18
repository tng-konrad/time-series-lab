# Time Series Lab
Experiments with time series - barebone implementations of ideas from papers

## Agentic Time Series Forecasting (ATSF)

Paper: [Beyond Model-Centric Prediction—Agentic Time Series Forecasting](https://arxiv.org/abs/2602.01776)

Notebook: [agentic-forecasting-simple.ipynb](agentic-forecasting-simple.ipynb)

### Overview

The notebook demonstrates a simplified setting where an AI agent manages the forecasting workflow:
1.  **Perception**: The agent is provided with a synthetic time series containing a structural break.
2.  **Planning & Action**: The agent selects an appropriate statistical forecasting tool (SES or Holt-Winters) and parameters based on its analysis of the data.
3.  **Reflection**: After an initial forecast performance check, the agent reflects on the failure (caused by the structural break) and refines its plan—typically by shortening the training window to the post-break data.

This "minimal setting" implementation validates the core two-cycle agentic workflow.

### Potential Extensions

- **Expanded Toolkit**: Adding more complex models (ARIMA, Transformers, boosted trees and whatnot) to the agent's available tools.
- **Multimodal Perception**: Providing the agent with visual plots of the series to improve its analysis of trends and breaks.
- **Long-term Memory**: Allowing the agent to learn from multiple forecasting tasks over time.


## Multi-layer Stack Ensembles for Time Series Forecasting

Paper: [Multi-layer Stack Ensembles for Time Series Forecasting](https://arxiv.org/abs/2511.15350)

Notebook: [mls-ensembles.ipynb](mls-ensembles.ipynb)

### Overview


## Multi-layer Stack Ensembles for Time Series Forecasting

Paper: [Multi-layer Stack Ensembles for Time Series Forecasting](https://arxiv.org/abs/2511.15350)

Notebook: [mls-ensembles.ipynb](mls-ensembles.ipynb)

The notebook implements a hierarchical stacking framework that learns optimal combinations of forecasters:

1. **Level 1 (Base Models)**: Trains a diverse set of "weak" learners, including Seasonal Naive, Linear Regression, and Multi-Layer Perceptrons (MLP). It uses time-series cross-validation to generate Out-of-Fold (OOF) predictions, ensuring the stacker sees "unseen" data patterns.
2. **Level 2 (Stackers)**: Learns to combine L1 predictions using different architectural approaches, such as a Linear Stacker (constrained weighted average) and an MLP Stacker (non-linear combination).
3. **Level 3 (Aggregator)**: Employs a Greedy Ensemble selection algorithm (Caruana et al., 2004) to find the best weighted combination of the Level 2 stackers, further reducing the Mean Absolute Error (MAE).
4. **Refinement**: Retrains all models on the full dataset to maximize performance for final inference.

### Potential Extensions

- **Model Diversity**: Incorporating more sophisticated base models like Gradient Boosted Trees (XGBoost/LightGBM) or specialized Transformer architectures.
- **Meta-Feature Engineering**: Adding exogenous variables (e.g., calendar features, lagged statistics) as inputs to the Level 2 stackers to help them decide which base models to trust in different contexts.
- **Dynamic Weighting**: Implementing attention mechanisms in the stackers to allow weights to shift based on the recent volatility of the time series.