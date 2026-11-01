# skillet: Video Recording Script (about 8–9 minutes)

Every number below is what the live app shows for these inputs, so it can be said with confidence.

Live app: https://skillet.streamlit.app/

---

## Before you record

- **Wake the app:** open https://skillet.streamlit.app/ about 5 minutes early.
- **Screen:** use a full-screen browser, close other tabs, and zoom to 90–100% so the left rail and the plan panel both fit.
- **Reset:** reload the page right before you start, so it shows the defaults (Developer Full Stack, 5 years, United States, AI ticked, empty stack).
- **Adding a skill:** type its name, then **click the exact skill** in the list. When you type "Python", the first row is "Select 2 matches", which would also add MicroPython, so don't press Enter.

---

## The script

### 0:00–0:45 · Introduction *(show the top of the page)*

**Say:**

> "Hello, I'm Vibhaansh Bhardwaj, 24BCE1401. This is my Business Analytics project, **skillet**: a recommendation model that tells a software developer **which five skills to learn next**.
>
> Developers and companies usually choose what to learn on instinct or from trend articles. I wanted to answer it with data, using the **2025 Stack Overflow Developer Survey**: **19,210 working developers, 122 skills and 156 countries**, with pay in US dollars."

### 0:45–1:45 · The inputs and the cold start *(point at the left rail, then the plan panel)*

**Do:** point at Role, Experience, Country, the AI checkbox, then the six skill boxes.

**Say:**

> "On the left are the user's inputs: role, years of experience, country, whether they use AI tools, and their current skills.
>
> With no skills entered, the model compares me with the **731 developers** who share my profile: full-stack, US, 0 to 15 years, AI users. It shows the highest-value skills in that group: **Docker, AWS, Terraform, Kubernetes and Snowflake**.
>
> The big number, **+82%**, is the pay association of learning all five together, with a 95% interval of +72% to +92%."

### 1:45–3:30 · Adding a stack: the main result *(the most important part)*

**Do:** add **Python, JavaScript, SQL** (Languages), **PostgreSQL** (Databases) and **Docker** (Cloud platforms).

**Say while adding:**

> "Now I'll enter a typical stack: Python, JavaScript, SQL, PostgreSQL and Docker."

**Then point at the sky chart:**

> "This chart places all 122 skills by how often developers use them **together**. My stack is drawn as the white constellation, and the five recommendations are the amber-ringed stars right next to it."

**Point at the plan panel:**

> "The model now finds my **300 nearest peers**: the developers among those 731 whose skills are most similar to mine, measured by **Jaccard similarity**. The recommendations are **TypeScript, VS Code, npm, React and AWS**.
>
> Take TypeScript: **67% of my peers know it**, and it's associated with **+6.7% pay** at the same experience and country.
>
> Estimated together, the five are worth **+24%**, interval +20% to +28%, with 3 of 5 significant. Their separate effects add up to about 41%, but skills that come as a bundle share one premium, so I fit them **in one joint regression** to avoid double-counting."

**Point at the three facts:**

> "My peers' median pay is **\$130,000**. They earn **2% less** than the average developer with the same experience in the same country. My persona is **Lean generalist stacks**."

### 3:30–4:30 · Profile changes *(shows every input matters)*

**Do:** change **Country → India**.

**Say:**

> "If I change the country to India, the median pay drops to **\$15,000** and the plan changes, because the model compares me only with developers in my own country. The plan is now worth **+14%**."

**Do:** change the country back to the **United States**, then move **experience to 25**.

**Say:**

> "With 25 years of experience, the median rises to **\$160,000**. These small tags show how each skill moved: **AWS moved up 3**, and **OpenAI GPT is new**. Every input changes the result."

### 4:30–5:30 · The evidence table *(scroll down)*

**Do:** move experience back to **5** and scroll to **"The five, in detail"**.

**Say:**

> "This table gives the evidence for each recommendation: the salary effect from its own regression, the **95% confidence interval**, the **p-value**, how many survey users know it, and the shares of my peers who **know** it and **want** to learn it.
>
> The score combines three things: **relevance** (do peers use it), **value** (the salary coefficient) and **demand** (do peers want it). Skills with a negative pay effect are never recommended."

### 5:30–6:15 · Persona and the ROI chart *(scroll)*

**Point at "Who you are in the data":**

> "Personas come from **PCA and K-means clustering** on skills. I found three: .NET, lean generalist and modern JavaScript. My stack falls in **Lean generalist**, a cluster of 8,868 developers."

**Scroll to "Skill ROI landscape":**

> "These are the 20 skills with the highest pay effect in the whole survey: **Scala +35%, Snowflake +33%, Terraform +31%**. Each dot is the estimate and each line its 95% interval. Brighter means more significant."

### 6:15–7:15 · The calculator *(scroll)*

**Do:** in "What if I also learned these?", add **Kubernetes, Terraform and Snowflake**.

**Say:**

> "The calculator values any set of skills together. Kubernetes, Terraform and Snowflake together are worth **+73.7%**, with an interval of +64% to +84%. Separately they'd add up to about 92%, but the joint model doesn't double-count overlapping skills. This chart shows each skill's contribution when estimated together."

### 7:15–7:50 · A rare profile *(scroll to top)*

**Do:** change **Country → Nepal**.

**Say:**

> "For a rare profile like Nepal, fewer than 50 developers match, so the model **relaxes the filters**. It matches on role only, and the page says exactly which filters were dropped."

### 7:50–8:45 · The models and validation *(open "How these recommendations are scored" at the bottom)*

**Say:**

> "Under the hood:
>
> - **Regression with country fixed effects** measures each skill's pay value.
> - **KNN with Jaccard similarity** finds similar developers.
> - **A joint regression** values the plan.
>
> I also used **hypothesis tests**, which showed AI users earn **9.9% more** at the same country and experience, even though the raw comparison says the opposite (Simpson's paradox). I compared **logistic regression, a decision tree and Naive Bayes** for job satisfaction, with an AUC of about 0.57, and used **PCA, K-means and factor analysis** for the personas.
>
> On **500 held-out developers**, the recommender found a hidden real skill **78% of the time**, against 72% for recommending popular skills and 5% for recommending the highest-paying ones."

### 8:45–9:15 · Close

**Say:**

> "One limitation: these are **associations, not causal effects**. The data is self-reported survey data. The app is live at **skillet.streamlit.app**, and the full method and results are in my reports. Thank you."

---

## Quick checklist of what to click

1. Defaults: **+82%**
2. Add Python, JavaScript, SQL, PostgreSQL, Docker: **+24%**, TypeScript first
3. Country India: **\$15,000**, **+14%**
4. Back to US, experience 25: **\$160,000**, tags
5. Experience back to 5, scroll to the evidence table
6. Persona, then the ROI chart
7. Calculator: Kubernetes + Terraform + Snowflake: **+73.7%**
8. Scroll up, Country Nepal: relaxation note
9. Open the method notes, then close

**If you go over time,** cut step 8 (Nepal) and shorten the models part. That gets you to about 7½ minutes.
