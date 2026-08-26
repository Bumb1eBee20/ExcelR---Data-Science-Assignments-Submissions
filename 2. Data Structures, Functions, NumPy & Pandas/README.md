# 2. Data Structures, Functions, NumPy & Pandas

# Python Assignment: Data Structures, Functions, NumPy & Pandas

This repository contains assignment questions covering core Python data structures, control flow, NumPy operations, and Pandas DataFrames.

---

## 📋 Table of Contents
- [Instructions](#instructions)
- [Section A: Data Structures & Control Structures](#section-a-data-structures--control-structures)
- [Section B: NumPy](#section-b-numpy)
- [Section C: Exploring Pandas](#section-c-exploring-pandas)

---

## Instructions
* **Attempt all questions.**[cite: 2]
* **Submit the completed assignment in `.ipynb` format.**[cite: 2]
* **Use appropriate function definitions and comments for clarity.**[cite: 2]

---

## Section A: Data Structures & Control Structures

1. **List Operations**
   Create a list of 5 integers. Perform and print the result of the following operations: `append`, `extend`, `insert`, `remove`, `pop`, `clear`, `index`, `count`, `sort`, and `reverse`[cite: 2].

2. **Tuple Immutability**
   Create a tuple that stores 3 student names[cite: 2]. Try changing the second name in the tuple[cite: 2]. What happens? Explain why[cite: 2].

3. **Set Uniqueness**
   Create a set of integers with some duplicate values[cite: 2]. Print the set and explain the output[cite: 2].

4. **Dictionary Manipulation**
   Create a dictionary with the keys: `'name'`, `'age'`, and `'city'`[cite: 2]. Update the age and add a new key `'email'`[cite: 2]. Print the final dictionary[cite: 2].

5. **Voting Eligibility Script**
   Write a script that checks if a person is eligible to vote (age ≥ 18)[cite: 2]. Take age as a variable and print the appropriate message[cite: 2].

6. **Grading System**
   Given a `marks` variable, print the corresponding grade[cite: 2]:
   * **90 and above:** `'A'`[cite: 2]
   * **75–89:** `'B'`[cite: 2]
   * **50–74:** `'C'`[cite: 2]
   * **Below 50:** `'Fail'`[cite: 2]

7. **Nested Number Classifier**
   Given a number, check if it's positive, and if it is also even[cite: 2]. If not positive, print if it's zero or negative[cite: 2].

---

## Section B: NumPy

1. **Array Creation**
   Create[cite: 2]:
   * A scalar using `np.array(5)`[cite: 2]
   * A 1D array with values 1 to 5[cite: 2]
   * A 2D array (2x3) with values from 10 to 60 in steps of 10[cite: 2]

2. **Random Matrix Generation**
   Generate a 4x4 NumPy array of random integers between 0 and 100 using `np.random.randint()`[cite: 2].

3. **NumPy to DataFrame Conversion**
   Create a 2D NumPy array of shape (3x3)[cite: 2]. Convert it into a Pandas DataFrame and add column names: `'A'`, `'B'`, `'C'`[cite: 2].

---

## Section C: Exploring Pandas

Create a small DataFrame manually with 10 rows and columns: `'Name'`, `'Age'`, `'City'`, and `'Salary'`[cite: 2]. Then perform the following operations:
* Use `.info()` and `.describe()`[cite: 2]
* Select `'Name'` and `'City'` columns[cite: 2]
* Drop the `'City'` column[cite: 2]
* Fill any missing values in the `'Salary'` column with the mean[cite: 2]
* Remove any duplicate rows[cite: 2]
