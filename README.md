# Time-series-lab
Experiments with time series


## Agentic Time Series Forecasting (ATSF)

Paper: [Beyond Model-Centric Prediction—Agentic Time Series Forecasting](https://arxiv.org/abs/2602.01776)

Notebook: [agentic-forecasting-simple.ipynb](agentic-forecasting-simple.ipynb)

This repository contains a minimal implementation of the ideas presented in the paper. 

### Overview

The notebook demonstrates a simplified setting where an AI agent manages the forecasting workflow:
1.  **Perception**: The agent is provided with a synthetic time series containing a structural break.
2.  **Planning & Action**: The agent selects an appropriate statistical forecasting tool (SES or Holt-Winters) and parameters based on its analysis of the data.
3.  **Reflection**: After an initial forecast performance check, the agent reflects on the failure (caused by the structural break) and refines its plan—typically by shortening the training window to the post-break data.

This "minimal setting" implementation validates the core two-cycle agentic workflow.

### Potential Extensions

While this *is* fully functional, many extensions are possible:
- **Expanded Toolkit**: Adding more complex models (ARIMA, Transformers, boosted trees and whatnot) to the agent's available tools.
- **Multimodal Perception**: Providing the agent with visual plots of the series to improve its analysis of trends and breaks.
- **Long-term Memory**: Allowing the agent to learn from multiple forecasting tasks over time.


## Multi-layer Stack Ensembles for Time Series Forecasting

Paper: [Multi-layer Stack Ensembles for Time Series Forecasting](https://arxiv.org/abs/2511.15350)


