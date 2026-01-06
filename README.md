### Case Study: Detecting Anomalies in Hospital Resource Usage (MVP)
Motivation

Hospitals operate close to capacity, and unexpected surges in admissions, bed occupancy, or ICU usage can overwhelm staff and infrastructure. Early detection of anomalous demand patterns enables proactive staffing, resource reallocation, and escalation planning.

Data (MVP Scope)

Publicly available hospital utilization summaries from Canadian Institute for Health Information (CIHI), including:

Monthly inpatient admissions

Average length of stay

Bed occupancy rates

ICU utilization (where available)

Only datasets ingestible using the current MVP of the Data Acquisition Package are used—no manual cleaning or enrichment beyond standardized normalization.

ML Task

Unsupervised anomaly detection, treating unusual utilization patterns as deviations from historical norms rather than labeled “events.”

Models explored:

Isolation Forest for fast, interpretable anomaly scoring

Autoencoder (optional extension) to model normal utilization patterns and flag high reconstruction error

Anomalies are detected at the regional and hospital-group level across time.

Output

Time-series plots with anomaly overlays

Ranked anomaly windows with severity scores

Short narrative explaining detected spikes and potential operational interpretations

Key Skill Signal

Practical unsupervised learning in a real public-sector context

Time-series reasoning without labeled outcomes

Clear separation of data acquisition, modeling, and interpretation
