# **Risk Classification - FlightOnTime**

**Framework:** Regulation (EU) 2024/1689 (AI Act), as amended by Regulation (EU) 2026/1744 (Digital Omnibus on AI)
**Assessed:** 11 August 2026
**Outcome:** **Not high-risk.** Minimal-risk tier, with transparency duties under Article 50(1).

---

## **1. What the system does**

FlightOnTime predicts whether a scheduled commercial flight will arrive 15+ minutes late (the BTS `ArrDel15` convention). Inputs are the flight's scheduled characteristics - carrier, route, timing, calendar features - plus the weather forecast at origin and destination. Output is a calibrated probability plus a binary label derived from a tuned operating threshold.

**Users:** travellers and applications acting on their behalf, deciding how much buffer to leave for a connection or which of two itineraries to book.

**What it is not used for.** Classification under the AI Act follows *intended purpose*, so these boundaries are what the assessment below rests on:

* not a safety component of any aircraft, avionics, or air traffic management system;
* no interface to ATC, dispatch, crew scheduling, or any operational aviation system;
* not used to grant, price, or deny any service - no compensation decisions, insurance pricing, or credit assessment;
* no biometric data, no emotion inference, no profiling of individuals - the unit of analysis is a **flight**, not a passenger;
* generates no synthetic audio, image, video, or text.

---

## **2. Classification**

### **2.1 Prohibited practices - Article 5: not engaged**

No manipulation or exploitation of vulnerabilities, no social scoring, no criminal-risk prediction, no biometric identification or categorisation, no emotion inference in work or education settings. The Omnibus also added a prohibition on AI-generated non-consensual intimate imagery - likewise not engaged.

### **2.2 Embedded in a regulated product - Article 6(1) + Annex I: not engaged**

Annex I lists other EU product-safety laws. Article 6(1) makes an AI system high-risk when it is a safety component of a product covered by one of them - the AI Act plugs into existing sectoral regimes rather than replacing them.

Section B contains two aviation entries - one on security, one on safety - and neither reaches FlightOnTime:

* **Item 13 - Reg. (EC) 300/2008, aviation *security*.** Screening, access control, protection against unlawful interference. FlightOnTime performs no security function at all.
* **Item 20 - Reg. (EU) 2018/1139, aviation *safety* (EASA).** Listed only as regards **unmanned aircraft** and their engines, propellers, parts and remote-control equipment. It does not reach manned commercial aviation.

FlightOnTime is not a product covered by either, nor a safety component of one. It reads public BTS statistics and public weather forecasts, and its output goes to a traveller. No regulated product, so no third-party conformity assessment.

### **2.3 Annex III areas - Article 6(2): not engaged**

Seven of the eight areas are plainly irrelevant: biometrics, education, employment, essential services and benefits, law enforcement, migration and border control, justice and democratic processes. FlightOnTime does nothing in any of them.

**Point 2 (critical infrastructure) is the only close call, and it fails twice over.**

The provision covers AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity.

**First, air transport isn't on the list.** It's a closed enumeration: critical *digital* infrastructure, *road* traffic, four utilities.

**Second, it isn't a "safety component".** Recital 55 defines those as systems that directly protect the physical integrity of critical infrastructure, or the health and safety of persons and property - the examples given are water-pressure monitoring and fire-alarm control in data centres.

FlightOnTime protects nothing physical. It operates purely on information. When the model makes a mistake, the cost lands on a traveller's schedule.

---

## **3. Obligations that do apply**

Not being high-risk doesn't mean no obligations. Both of these are **live now**.

### **Article 50(1) - transparency**

Article 50 isn't tied to the risk tiers at all - it applies whether a system is high-risk, prohibited, or minimal-risk.

**Required:** users interacting with the system must be told they are dealing with AI, unless that is obvious to a reasonably observant person.

**Implementation:** an explicit statement in the web interface and in the API docs, alongside the model's stated accuracy and limitations.


### **Article 4 - AI literacy**

Staff operating the system need to understand what it does and doesn't do. The Omnibus softened this into a duty to *support* AI literacy among staff, rather than guarantee a specific level.

**Implementation:** model cards, dataset cards, developer guide, documented limitations and failure modes.

---

## **4. Voluntary alignment (Article 95)**

Article 95 encourages applying high-risk practices voluntarily. We already do, as engineering practice rather than legal obligation - which also means a change of intended purpose wouldn't leave us starting from zero:

| Chapter III requirement | Already in place |
|---|---|
| Art. 10 - data governance | Versioned pipeline (DVC), automated validation (Great Expectations, Deepchecks), documented provenance |
| Art. 11 - technical documentation | Model and dataset cards, ML Canvas, reproducible pipeline definition |
| Art. 12 - record-keeping | Experiment tracking and model registry (MLflow), logged predictions |
| Art. 15 - accuracy and robustness | Walk-forward temporal validation, probability calibration, behavioural testing, drift monitoring |

---

## **5. Conclusion**

**FlightOnTime sits in the minimal-risk tier**, subject to the Article 50(1) transparency duty in its user-facing interface and the Article 4 AI literacy duty. Not high-risk under either Article 6(1) or Article 6(2).

Classification tracks intended purpose, so revisit this if any of the following happens:

* output goes to an airline, airport, or air navigation provider **for operational decisions** rather than passenger information;
* the system gets integrated into an Annex I product - in aviation that means unmanned aircraft or a security function;
* output starts determining **access to, eligibility for, or pricing of** a service, e.g. compensation or insurance;
* passenger-level features are introduced;
* the Commission amends Annex III to cover air transport.

---

## **6. Sources**

* Regulation (EU) 2024/1689 (AI Act)
* Regulation (EU) 2026/1744 (Digital Omnibus on AI)
* Regulation (EC) No 300/2008 (aviation security)
* Regulation (EU) 2018/1139 (aviation safety, EASA)
* Directive (EU) 2022/2557 (critical entities)
* EU AI Act Explorer (Future of Life Institute) - https://artificialintelligenceact.eu
