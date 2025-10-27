### **Project Description**
**Bayesian Framework for Wind Power Forecasting with Uncertainty Estimation**

**Purpose**

The purpose of this thesis is to develop a model for forecasting the total wind-power production for each Norwegian bidding zone, including quantified uncertainty estimates, two days ahead at an hourly resolution for 24 hours. Reliable forecasts are essential for Statnett, the Norwegian transmission system operator responsible for maintaining balance and stability in the national power grid. Accurate and uncertainty-aware forecasts support better planning, decision-making, and efficient energy management.

**Background**

Earlier this year, Statnett and SINTEF organized the “WindAI – Forecasting wind power production” competition, aiming to develop more accurate and robust models for forecasting wind-power production in Norway. The competition produced a variety of strong approaches, demonstrating valuable design choices such as feature engineering techniques and handling of installed capacity changes.
This thesis will integrate selected components and methodological ideas from these approaches within a coherent Bayesian framework. The Bayesian perspective enables explicit treatment of uncertainty and robustness, both of which are critical for operational forecasting.

**Goal**

The goal is to design and implement a **probabilistic wind-power forecasting model** that:
* Predicts wind-power production exactly two days ahead at an hourly resolution for 24 hours,
* Provides well-calibrated uncertainty estimates,
* Is robust to changes such as wind-park capacity adjustments and the commissioning of new wind parks, and
* Is feasible for deployment at Statnett.

**Work Plan**

Step 1: Preparation and Exploration
* **1a Familiarization:** Review the WindAI competition models and datasets to understand their structure, features, and performance characteristics. Conduct a brief literature review on wind-power forecasting methods.
* **1b Bayesian framework setup:** Develop a simple baseline model within a Bayesian framework and study relevant literature on probabilistic forecasting and uncertainty estimation.
* **1c Evaluation metrics:** Identify suitable metrics to assess model performance, including measures of accuracy, uncertainty calibration, and robustness to changes in input data.

Step 2: **Model Design**
* Select and integrate relevant components or ideas from the *WindAI* competition models (e.g., use of unavailability messages, capacity factor modeling, and feature representations). Design an overall model architecture within the Bayesian framework.

Step 3: **Training and Evaluation**
* Train and compare the developed models, combining insights from Step 1 and Step 2. Evaluate performance in terms of both predictive accuracy and uncertainty estimation quality.

*Note: Steps 2 and 3 will follow an iterative process*

