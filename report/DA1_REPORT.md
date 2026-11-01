# skillet — DA 1: Problem / Idea of the Model

---

## DA 1: Problem / Idea of the Model

### 1. Title of the project

**skillet: an evidence-based recommendation model that tells a software developer which five skills to learn next, using salary regressions, peer similarity and peer demand from the 2025 Stack Overflow Developer Survey.**

### 2. Introduction and background of the problem

Software skills change quickly. A developer can choose from hundreds of languages, databases, cloud platforms, frameworks and AI tools. Companies spend large learning-and-development (L&D) budgets helping their engineers keep up. Despite that, the decision of *what* to learn next is usually made on instinct, on what colleagues happen to use, or on "top skills of the year" articles.

Those sources have three weaknesses:

1. **They ignore the individual.** A front-end developer and a data engineer should not get the same advice.
2. **They ignore evidence.** "Popular" is not the same as "valuable", and "valuable" in one country is not the same in another.
3. **They give no measure of certainty.** A claim that a skill "pays more" is rarely accompanied by a sample size, a confidence interval or a control for experience and location.

At the same time, large public datasets now record what developers use, what they want to learn, and what they earn. The Stack Overflow Developer Survey collects exactly this from tens of thousands of working developers every year. That makes a data-driven recommendation possible.

### 3. Problem statement

> Given a developer's role, years of experience, country, use of AI tools and current skill set, recommend the **five skills they should learn next**. Back each recommendation with statistical evidence of its relevance to developers like them, the demand for it among those developers, and its association with pay, holding experience and country fixed. Estimate the combined value of the five-skill plan with a valid confidence interval.

### 4. Objectives of the model

1. **Personalised recommendation.** Rank every skill the developer does not yet have, using the behaviour of developers with a similar profile and stack.
2. **Evidence for every recommendation.** For each skill, report the pay association (%), its 95% confidence interval and p-value, the share of similar developers who know it and the share who want to learn it.
3. **Plan valuation.** Estimate the combined pay association of the five recommended skills without double-counting skills that usually come together.
4. **Population insight.** Identify developer personas (segments) and test common beliefs, such as whether AI-tool users earn more or are more satisfied.
5. **Validation.** Show on held-out respondents that the recommendations beat simple baselines.
6. **A usable product.** Deliver the model as an interactive web application.

### 5. Proposed dataset and its source

| Item | Detail |
|---|---|
| Dataset | **Stack Overflow Developer Survey 2025** (public, anonymised) |
| Source | Stack Overflow Insights, https://survey.stackoverflow.co/ (download of the full results file, `survey_results_public.csv`) |
| Size | About 49,000 responses and 170 questions; about 19,000 working professional developers after filtering |
| Key fields | `Employment`, `MainBranch`, `DevType` (role), `WorkExp`, `Country`, `ConvertedCompYearly` (annual pay in USD), `JobSat` (0–10), `AISelect` (AI-tool use), `RemoteWork`, `OrgSize`, and the multi-select *HaveWorkedWith* / *WantToWorkWith* questions for languages, databases, platforms, web frameworks, developer environments and AI models |
| Supporting data | The 2022, 2023 and 2024 surveys, for a multi-year trend of skill pay premiums |

### 6. Proposed methodology and statistical techniques

The model combines three building blocks:

| Block | Technique (syllabus module) | Purpose |
|---|---|---|
| **Value** | Simple and multiple linear regression of log pay, with country fixed effects (Module 3) | Pay association of each skill, at the same experience and country |
| **Fit** | K-nearest neighbours with Jaccard similarity (Module 4) | Find developers with a similar profile and stack |
| **Demand** | Shares of similar developers who *want* each skill | Measure demand among similar developers |

Supporting analyses:

- **Data preparation and visualisation (Modules 1–2):** cleaning, outlier treatment, histograms, box plots, scatter plots, heat maps.
- **Hypothesis testing (Module 2):** Welch's t-tests and chi-square tests with Cramér's V.
- **Classification (Module 4):** logistic regression, decision tree and Naive Bayes for job satisfaction, evaluated by ROC/AUC.
- **Dimensionality reduction and clustering (Module 5):** PCA, K-means, hierarchical clustering and factor analysis to find developer personas.
- **Further analysis:** one-way ANOVA across personas, a skill co-occurrence network, and a held-out evaluation of the recommender with bootstrap confidence intervals.

**Proposed pipeline:**

1. Clean the survey and build a binary skill matrix (developer × skill).
2. Estimate each skill's salary effect with regression.
3. For a given user, filter developers with the same profile, find the most similar stacks, and score each candidate skill on relevance, value and demand.
4. Value the top five together with a joint regression.
5. Present the results in a web app.

### 7. Expected recommendations and outcomes

- **For a developer:** a ranked list of five skills, each with its pay association and confidence interval, and with how many similar developers know it and want it. Plus the combined value of the plan.
- **For L&D managers:** developer personas that can serve as training tracks, and a ranking of skills by pay premium and reliability.
- **For HR and analytics teams:** evidence on questions such as the pay gap for AI-tool users, controlled for country and experience.
- **Expected findings, to be tested:**
  - cloud and data-platform skills carry the highest pay premiums;
  - personalised recommendations recover a developer's real next skills better than popularity or salary rankings alone;
  - job satisfaction is only weakly explained by skills.
