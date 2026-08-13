# **Risk Classification - FlightOnTime**

**Framework:** Regulation (EU) 2024/1689 (AI Act), as amended by Regulation (EU) 2026/1744 (Digital Omnibus on AI)
**Assessed:** 11 August 2026
**Outcome:** **Not high-risk.** Minimal-risk tier, with transparency duties under Article 50(1).

---

## **1. What the system does**

FlightOnTime predicts whether a US scheduled commercial flight will arrive 15+ minutes late (the BTS `ArrDel15` convention). The inputs are the flight's scheduled characteristics (e.g., carrier, route, timing, calendar features) plus the weather forecast at the origin and destination. The output is a calibrated probability accompanied by a binary label derived from a tuned operating threshold.

**Users:**
* travellers, deciding how much buffer to leave for a connection or which of two itineraries to book;
* airport-side applications, monitoring gate assignments and turnaround planning;
* airline analysts, monitoring which routes carry persistently elevated delay risk and, through explainability, why.

**What it is not used for.** Classification under the AI Act follows *intended purpose*, so these boundaries are what the assessment below rests on:

* it is not a safety component of any aircraft, avionics, or air traffic management system;
* it is not an interface to ATC, dispatch, crew scheduling, or any operational aviation system;
* it is not used to grant, price, or deny any service - no compensation decisions, insurance pricing, or credit assessment;
* it does not process biometric data, infer emotions, or profile individuals - the unit of analysis is a **flight**, not a passenger;
* it does not generate any synthetic audio, image, video, or text.

---

## **2. Classification**

### **2.1 Prohibited practices - Article 5: Not applicable**

FlightOnTime does not involve manipulation or exploitation of vulnerabilities, social scoring, criminal-risk prediction, biometric identification or categorisation, or emotion inference in workplace or educational settings. The Omnibus also added a prohibition on AI-generated non-consensual intimate imagery, which is likewise not applicable here.

### **2.2 Embedded in a regulated product - Article 6(1) + Annex I: Not applicable**

Article 6(1) classifies an AI system as high-risk if it is either a product itself or a safety component of a product covered by the EU harmonisation legislation listed in Annex I, and is required to undergo a third-party conformity assessment prior to being placed on the market or put into service.

Section B of Annex I contains two aviation entries - one on security, one on safety - and neither reaches FlightOnTime:

* **Item 13 - Reg. (EC) 300/2008, aviation *security*.** Screening, access control, protection against unlawful interference. FlightOnTime performs no security function.
* **Item 20 - Reg. (EU) 2018/1139, aviation *safety* (EASA).** Listed only as regards **unmanned aircraft** and their engines, propellers, parts and remote-control equipment. It does not reach commercial aviation.

FlightOnTime is not a product covered by Annex I, nor a safety component of one. It ingests public BTS statistics and public weather forecasts, and its output is delivered via an API. It does not involve any regulated product, and it does not require a third-party conformity assessment.

### **2.3 Annex III areas - Article 6(2): Not applicable**

Seven of the eight areas are clearly not applicable: biometrics, education, employment, essential services and benefits, law enforcement, migration and border control, and justice and democratic processes. FlightOnTime does not operate in any of these areas.

**Point 2 (critical infrastructure) is the only close call, but it fails on two separate grounds.**

The provision covers AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity.

**First, air transport is not on the list.** It is a closed enumeration: critical *digital* infrastructure, *road* traffic, and four utilities.

**Second, it is not a "safety component".** Recital 55 defines these as systems that directly protect the physical integrity of critical infrastructure, or the health and safety of persons and property - the examples given are water-pressure monitoring and fire-alarm control in data centres.

FlightOnTime protects nothing physical. It operates purely on information. If the model makes an error, the impact is strictly confined to a traveller's schedule.

---

## **3. Obligations that do apply**

Falling outside the high-risk category does not imply a total absence of legal obligations. Both of the following provisions are currently in force.

### **Article 50(1) - transparency**

Article 50 is not tied to the risk tiers - it applies whether a system is prohibited, high-risk, or minimal-risk.

**Required:** users interacting with the system must be told they are dealing with AI, unless that is obvious to a reasonably observant person.

**Implementation:** an explicit statement in the web interface and in the API documentation, alongside the model's stated accuracy and limitations.


### **Article 4 - AI Literacy**

Personnel operating or managing the system must understand its capabilities, intended purpose, and operational scope. The Omnibus reframed this requirement into a duty to support and promote AI literacy among relevant staff.

**Implementation:** Provision of detailed model cards, dataset cards, developer guides, and explicit documentation covering system limitations and known failure modes.

---

## **4. Voluntary alignment (Article 95)**

Article 95 encourages the voluntary application of requirements designed for high-risk AI systems. FlightOnTime already adopts these practices as standard engineering rigor rather than out of a legal obligation. Consequently, should its intended purpose change in the future, the system would not be starting from scratch:

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
* the system gets integrated into an Annex I product;
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
