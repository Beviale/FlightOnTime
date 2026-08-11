# **Risk Classification — FlightOnTime**

**Framework:** Regulation (EU) 2024/1689 (AI Act), as amended by Regulation (EU) 2026/1744 (Digital Omnibus on AI)
**Assessed:** 11 August 2026 ·
**Outcome:** **Not high-risk.** Minimal-risk tier, with transparency duties under Article 50(1).

---

## **1. What the system does**

FlightOnTime predicts whether a scheduled commercial flight will arrive 15+ minutes late (the BTS `ArrDel15` convention). Inputs are the flight's scheduled characteristics — carrier, route, timing, calendar features — plus the weather forecast at origin and destination. Output is a calibrated probability plus a binary label derived from a tuned operating threshold.

**Users:** travellers and applications acting on their behalf, deciding how much buffer to leave for a connection or which of two itineraries to book.

**What it is not used for.** Classification under the AI Act follows *intended purpose*, so these boundaries are what the assessment below rests on:

* not a safety component of any aircraft, avionics, or air traffic management system;
* no interface to ATC, dispatch, crew scheduling, or any operational aviation system;
* not used to grant, price, or deny any service — no compensation decisions, insurance pricing, or credit assessment;
* no biometric data, no emotion inference, no profiling of individuals — the unit of analysis is a **flight**, not a passenger;
* generates no synthetic audio, image, video, or text.

---

## **2. Classification**

### **2.1 Prohibited practices — Article 5: not engaged**

No manipulation or exploitation of vulnerabilities, no social scoring, no criminal-risk prediction, no biometric identification or categorisation, no emotion inference in work or education settings. The Omnibus also added a prohibition on AI-generated non-consensual intimate imagery — likewise not engaged.

### **2.2 Embedded in a regulated product — Article 6(1) + Annex I: not engaged**

Annex I Section B covers civil aviation (Regulation (EU) 2018/1139) — the route by which AI inside certified aeronautical products or ATM systems becomes high-risk.

FlightOnTime is neither an aeronautical product nor a component of one. It reads public BTS flight statistics and public weather forecasts, and its output reaches a traveller. No certified product is involved, so no third-party conformity assessment is triggered.

### **2.3 Annex III areas — Article 6(2): not engaged**

Seven of the eight areas are plainly irrelevant: biometrics, education, employment, essential services and benefits, law enforcement, migration and border control, justice and democratic processes. FlightOnTime performs no function in any of them.

**Point 2 (critical infrastructure) is the only close call, and it fails on two independent grounds.**

The provision covers <cite>AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity</cite>.

**Ground 1 — air transport is not in the list.** The enumeration is closed: critical *digital* infrastructure, *road* traffic, and four utilities. Aviation appears nowhere.

**Ground 2 — it is not a "safety component".** It defines these as <cite>systems used to directly protect the physical integrity of critical infrastructure or the health and safety of persons and property</cite>, whose failure <cite>might directly lead to risks to the physical integrity of critical infrastructure and thus to risks to health and safety of persons and property</cite>. The examples given are water-pressure monitoring and fire-alarm control in data centres.

FlightOnTime protects nothing physical. A wrong prediction means a traveller leaves at a suboptimal time — inconvenience and misallocated personal time, not a physical-safety risk.

### **2.4 Article 6(3) — noted, not relied on**

Since no Annex III area is engaged, the derogation is unnecessary. Recorded only to show the conclusion holds even under a broader reading of point 2: the system <cite>performs a narrow procedural task</cite> — one probability for one flight, on a fixed feature set, feeding no downstream determination about a person. The profiling trigger, which would <cite>always</cite> force high-risk status, is not met: no feature describes an individual.

---

## **3. Obligations that do apply**

Not being high-risk does not mean no obligations. Both of the following are **in force now**.

### **Article 50(1) — transparency · applies since 2 August 2026**

Article 50 is <cite>not tied to the risk-based classification and applies regardless of whether a system qualifies as high-risk, prohibited, or minimal-risk</cite>.

**Required:** users interacting with the system must be told they are dealing with AI, unless it is obvious to a reasonably observant person.

**Implemented:** an explicit statement in the web interface and in the API documentation, alongside the model's stated accuracy and limitations.

### **Article 4 — AI literacy · applies since 2 February 2025**

Staff operating the system must understand what it does and does not do. The Omnibus <cite>softened this to a duty to support the development of AI literacy among staff rather than to guarantee a specific level</cite>.

**Implemented:** model cards, dataset cards, developer guide, documented limitations and failure modes.

---

## **4. Voluntary alignment (Article 95)**

Article 95 encourages applying high-risk practices voluntarily. FlightOnTime already does, as engineering practice rather than legal obligation — which also means a change of intended purpose would not find the project starting from zero:

| Chapter III requirement | Already in place |
|---|---|
| Art. 10 — data governance | Versioned pipeline (DVC), automated validation (Great Expectations, Deepchecks), documented provenance |
| Art. 11 — technical documentation | Model and dataset cards, ML Canvas, reproducible pipeline definition |
| Art. 12 — record-keeping | Experiment tracking and model registry (MLflow), logged predictions |
| Art. 15 — accuracy and robustness | Walk-forward temporal validation, probability calibration, behavioural testing, drift monitoring |

---

## **5. Conclusion and triggers for reassessment**

**FlightOnTime sits in the minimal-risk tier**, subject to the Article 50(1) transparency duty in its user-facing interface and the Article 4 AI literacy duty. It is not high-risk under either Article 6(1) or Article 6(2).

Because classification tracks intended purpose, this must be reassessed if any of the following happens:

* output is supplied to an airline, airport, or air navigation provider **for operational decisions** rather than passenger information;
* the system is integrated into a product certified under Regulation (EU) 2018/1139;
* output is used to determine **access to, eligibility for, or pricing of** a service, e.g. compensation or insurance;
* passenger-level features are introduced;
* the Commission amends the law to cover air transport.

---

## **6. Sources**

* Regulation (EU) 2024/1689 (AI Act)
* Regulation (EU) 2026/1744 (Digital Omnibus on AI)
* Regulation (EU) 2018/1139 (civil aviation)
* Directive (EU) 2022/2557 (critical entities) 
* EU AI Act Explorer (Future of Life Institute) — https://artificialintelligenceact.eu