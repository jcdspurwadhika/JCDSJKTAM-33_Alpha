# JCDSJKTAM-33 Alpha
# E-Commerce Churn Analysis and Prediction.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://ecom-churn-predict.streamlit.app/)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-5BC0DE?style=flat-square&logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/rafi.priyahutama/viz/E-CommerceChurnAnalysisDashboard_17702170327400/Dashboard1-Overview?publish=yes)
![Task](https://img.shields.io/badge/Task-Data_Analysis_and_Modelling-34495E?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Library-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Model](https://img.shields.io/badge/Model-XGBoost-A569BD?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-2ECC71?style=flat-square)
## Business Understanding

### Background and Market Context

India's e-commerce sector has transitioned from aggressive user acquisition to a retention-focused phase. With over 800 million active internet users and widespread smartphone adoption, the market is projected to surpass USD 200 billion by 2026. However, growth from Tier-1 and Tier-2 cities has plateaued, shifting the strategic priority toward extracting profitability from existing customers rather than chasing new sign-ups.

Customer acquisition in India's competitive landscape has become increasingly expensive. Customer Acquisition Cost (CAC) for electronics buyers ranges between ₹800–₹1,500 (USD 10–20), spiking above ₹2,500 (USD 30+) in premium segments. With razor-thin net margins of 0.5%–3% after logistics and overheads, the first transaction rarely generates profit. **Customers only become truly profitable through repeat purchases over time.**

Adding to this pressure, consumer behavior is shifting due to competition from Quick-commerce players (Blinkit, Zepto, Swiggy) and deep-discount marketplaces. Churn is no longer simply about inactivity—it increasingly represents **migration of Customer Lifetime Value (CLV) to competitors** offering marginally better pricing or service.

### About Flopkart

Flopkart is one of India's largest e-commerce platforms, offering a diverse product range spanning electronics, fashion, appliances, and groceries. Despite strong logistical infrastructure and brand recognition, the company faces persistent challenges: retaining high-value customers, managing escalating acquisition costs, and defending market share against Q-commerce disruptors and regional competitors.

---

## Defining Churn

The churn definition in this project was established through rigorous evaluation of behavioral, transactional, and experiential variables, combined with contextual understanding of Indian e-commerce dynamics.

**Key observations from the dataset:**
- Customers labeled as churned exhibit `DaySinceLastOrder = 0`, non-zero `OrderCount`, and measurable engagement—contradicting traditional inactivity-based definitions
- Indian e-commerce lacks subscription models (unlike Netflix/Spotify) or formal account termination triggers
- Variables like `SatisfactionScore`, `Complain`, and `CashbackAmount` indicate that customer experience and incentive dependency are central to the labeling logic

| Traditional Churn | This Dataset |
|---|---|
| Binary: canceled or not | Spectrum: customer can be active but flagged |
| Backward-looking ("User has churned") | Forward-looking ("User is at risk") |
| Triggered by explicit action | Inferred from behavioral signals |
| Measures transaction | Measures relationship health |

**Conclusion:** Churn represents a **business-defined, algorithm-assisted customer label** indicating:
- **Churn = 1:** Customer is unlikely to sustain long-term engagement
- **Churn = 0:** Customer will most likely maintain long-term engagement

Churn does **NOT** mean account deletion, app uninstallation, or that the customer will never return. The flag serves as a retention risk indicator based on combined behavioral trends and historical performance.

---

## Problem Statement

Customer churn presents a critical resource allocation challenge for e-commerce platforms. While churn reflects declining engagement and diminishing customer value, the core threat lies in **limited marketing and retention resources.** Interventions such as promotions, discounts, and personalized outreach require significant investment, making indiscriminate efforts economically unsustainable.

In non-subscription contexts like Flopkart, churn is rarely triggered by a single explicit action. Instead, it emerges from patterns across behavioral, engagement, and transactional signals. Churn classification therefore serves as a **decision-enabling mechanism to prioritize which customers warrant proactive intervention.** Failing to identify at-risk customers carries **substantial economic consequences in the form of lost future customer value.**

This project analyzes key behavioral features associated with churn and develops a predictive model to identify high-risk customers. By enabling targeted interventions, the model supports more accurate and efficient retention efforts.

**Target Variable:**
- `0` : Customer does not churn
- `1` : Customer churns

**Stakeholder:** Flopkart's Marketing and Strategy Team  
**Project Role:** Flopkart's Data Science Team

---

## Objectives

1. Develop a churn classification model optimized toward **minimizing False Negatives** (avoiding missed high-risk churners)
2. Identify and quantify the key behavioral and engagement factors associated with elevated churn risk
3. Deliver a robust, deployable churn risk model that supports marketing and strategy decision-making

---

## Evaluation Metrics

### Cost-Benefit Framework

Monetary costs were assigned to classification outcomes based on Indian e-commerce market realities (2024–2025), focusing on the Electronics category.

**Cost of False Negative (Missed Churner):**
| Component | Value |
|-----------|-------|
| Average Order Value (AOV) | ₹5,000 (~USD 55) |
| Gross Margin per Order (10–15%) | ₹750 (~USD 7.50) |
| Annual Gross Margin (2 purchases/year) | ₹1,500 (~USD 15) |
| Lost Margin (3-year lifetime) | ₹4,500 (~USD 45) |
| Replacement Cost (CAC) | ₹1,000–1,500 (~USD 13) |
| **Total FN Cost** | **~USD 60** |

**Cost of False Positive (Unnecessary Intervention):**
| Component | Value |
|-----------|-------|
| Communication (SMS/WhatsApp) | ~USD 0.05 |
| Incentive (5–10% discount on AOV) | ~USD 4.50 |
| **Total FP Cost** | **~USD 5** |

**Asymmetry Ratio:** `60:5 = 12:1`  
Missing one true churner costs as much as running 12 unnecessary retention actions.

**Precision Floor:** `5/60 = 8.4%`  
Minimum precision required for a retention campaign to break even.

### Selected Metrics

| Metric | Purpose |
|--------|---------|
| **F2-Score** | Primary metric—prioritizes recall to minimize missed churners |
| **PR AUC** | Supporting metric—evaluates ranking ability in imbalanced data |
| **Confusion Matrix** | Breakdown of prediction outcomes |

---

## Dataset Overview

| Feature | Description |
|---------|-------------|
| CustomerID | Unique identifier |
| Churn | Target label (1 = At-risk, 0 = Stable) |
| Tenure | Months with platform |
| PreferredLoginDevice | Primary access device |
| CityTier | Urban classification (1 = Metro, 2 = Mid-tier, 3 = Small city) |
| WarehouseToHome | Delivery distance (km) |
| PreferredPaymentMode | Common payment method |
| Gender | Customer gender |
| HourSpendOnApp | Daily app usage |
| NumberOfDeviceRegistered | Linked devices count |
| PreferedOrderCat | Most purchased category |
| SatisfactionScore | Feedback rating (1–5) |
| MaritalStatus | Relationship status |
| NumberOfAddress | Saved delivery locations |
| Complain | Recent complaint filed (1 = Yes) |
| OrderAmountHikeFromlastYear | YoY spending change (%) |
| CouponUsed | Coupons redeemed |
| OrderCount | Total transactions |
| DaySinceLastOrder | Days since last purchase |
| CashbackAmount | Average cashback earned |

---

## Technology Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Tableau
- Jupyter Notebook

---

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter Notebook
4. Execute cells sequentially

---

## Key Findings

- **82% of churned customers have tenure ≤ 3 months**—the early period is critical for retention
- **Complaint history** is the second strongest churn indicator after tenure
- Lower satisfaction scores and reduced app engagement signal disengagement
- F2-score evaluation is essential given the 12:1 cost asymmetry between missed churners and false alarms

---

## Recommendations

### Business Recommendations: New Customer Retention Program

**Target:** All customers with tenure ≤ 3 months

| Initiative | Description | Objective |
|------------|-------------|-----------|
| **Automated Welcome Series** | Personalized touchpoints at day 1, 7, 14, 30, 60, 90 | Maintain early engagement |
| **Early Purchase Protection** | Extended return windows + proactive delivery tracking | Build trust with new customers |
| **Milestone Rewards** | Dual-track incentives for tenure duration and order count | Gamification and psychological stickiness |
| **Priority Support Queue** | Route new customers to high-performing agents | Prevent negative experiences that trigger complaints |

### Machine Learning Recommendations

| Recommendation | Rationale |
|----------------|-----------|
| **Periodic Retraining (6-month cycle)** | Prevent model drift as customer behavior evolves |
| **Enhanced Feature Engineering** | Add monetary indicators aligned with RFM framework |
| **Batch Prediction Support** | Enable CSV upload in Streamlit for mass scoring |

**Proposed Batch Workflow:**
```
Input:  CSV (customer data without churn flag)
           ↓
Output: CSV (original data + churn prediction + probability score)
```

---

## Disclaimer

This analysis serves educational and demonstration purposes. Production implementation would require additional validation, continuous monitoring, and adaptation to specific operational constraints.
