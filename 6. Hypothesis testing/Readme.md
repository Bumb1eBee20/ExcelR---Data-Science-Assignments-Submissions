# 6. HYPOTHESIS TESTING

# Hypothesis Testing - Bombay Hospitality Ltd.

## Background
Bombay hospitality Ltd. operates a franchise model for producing exotic Norwegian dinners throughout New England. The operating cost for a franchise in a week (W) is given by the equation W = $1,000 + $5X, where X represents the number of units produced in a week. Recent feedback from restaurant owners suggests that this cost model may no longer be accurate, as their observed weekly operating costs are higher.

---

## Objective
To investigate the restaurant owners' claim about the increase in weekly operating costs using hypothesis testing.

---

## Data Provided
* **Theoretical weekly operating cost model:** W = $1,000 + $5X
* **Sample size (n):** 25 restaurants
* **Sample mean weekly cost ($\bar{x}$):** Rs. 3,050
* **Units produced (X):** Follows a normal distribution with mean ($\mu_X$) of 600 units and standard deviation ($\sigma_X$) of 25 units

---

## Assignment Tasks

1. **State the Hypotheses statement:**
   Formulate the null ($H_0$) and alternative ($H_1$) hypotheses based on the restaurant owners' claim.

2. **Calculate the Test Statistic:**
   Use the z-statistic formula:
   $$z = \frac{\bar{x} - \mu}{\frac{\sigma}{\sqrt{n}}}$$
   * $\bar{x}$ = sample mean weekly cost (Rs. 3,050)
   * $\mu$ = theoretical mean weekly cost according to the cost model ($W = \$1,000 + \$5X$ for $X = 600$ units)
   * $\sigma = 5 \times 25$
   * $n = 25$

3. **Determine the Probability and Compare:**
   Evaluate using an alpha level of 5% ($\alpha = 0.05$).

4. **Make a Decision:**
   Compare the test statistic with the critical value to decide whether to reject the null hypothesis.

5. **Conclusion:**
   Based on the decision, conclude whether there is strong evidence to support the claim that weekly operating costs are higher than the model suggests.

---

## Submission Guidelines
* Prepare a Python file detailing each step of your hypothesis testing process.
* Include calculations for the test statistic and the critical value.
* Provide a clear conclusion based on your analysis.





# Estimation and Confidence Intervals

## Background
In quality control processes, especially when dealing with high-value items, destructive sampling is a necessary but costly method to ensure product quality. The test to determine whether an item meets the quality standards destroys the item, leading to the requirement of small sample sizes due to cost constraints.

---

## Scenario
A manufacturer of print-heads for personal computers is interested in estimating the mean durability of their print-heads in terms of the number of characters printed before failure. To assess this, the manufacturer conducts a study on a small sample of print-heads due to the destructive nature of the testing process.

---

## Data
A total of 15 print-heads were randomly selected and tested until failure. The durability of each print-head (in millions of characters) was recorded as follows:

`1.13, 1.55, 1.43, 0.92, 1.25, 1.36, 1.32, 0.85, 1.07, 1.48, 1.20, 1.33, 1.18, 1.22, 1.29`

---

## Assignment Tasks

### a. Build 99% Confidence Interval Using Sample Standard Deviation
Assuming the sample is representative of the population, construct a 99% confidence interval for the mean number of characters printed before the print-head fails using the sample standard deviation. Explain the steps you take and the rationale behind using the t-distribution for this task.

### b. Build 99% Confidence Interval Using Known Population Standard Deviation
If it were known that the population standard deviation is 0.2 million characters, construct a 99% confidence interval for the mean number of characters printed before failure.
