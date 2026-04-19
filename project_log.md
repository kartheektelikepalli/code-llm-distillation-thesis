# 📘 Code LLM Distillation Project — Design Log & Learnings

## 🧠 Overview

This project aims to distill a larger code generation model (CodeLlama 7B) into a smaller, efficient 1B model using supervised fine-tuning (LoRA). The goal is to create a lightweight, deployable coding assistant with improved performance over baseline small models.

---

## 🚀 Project Pipeline (High-Level)
Teacher Model (7B)
↓
Inference on MBPP
↓
Teacher Dataset (964 samples)
↓
Validation / Refinement Attempts
↓
Final Dataset (612 samples)
↓
Student Training (LoRA)


---

## 📊 Phase P0–P3 Summary

### 🔹 Dataset: MBPP
- Benchmark dataset for Python code generation
- Contains prompts + hidden test cases
- Evaluation metric: pass@1

---

### 🔹 Teacher Model
- Model: CodeLlama 7B (quantized via GGUF)
- Runtime: `llama.cpp`
- Interface: HTTP server

---

### 🔹 Inference Setup

Key characteristics:
- Batch inference over ~964 prompts
- Output stored in `.parquet`
- Validation via MBPP test cases

---

### 🔹 Initial Results
Pass@1: 625 / 964 = 64.83%


---

## 🧪 Prompt Engineering Efforts

We iteratively refined prompts to improve code generation quality:

### Techniques Tried:
- Explicit instruction formatting
- Function-only output enforcement
- Removing explanations
- Encouraging edge-case handling

### Observations:
- Prompt tuning had **marginal gains**
- Model still produced:
  - Markdown wrappers
  - Extra explanations
  - Embedded test cases

---

## ⚙️ Infrastructure Setup

### 🔹 llama.cpp
- Used quantized GGUF model
- Ran local inference server

Challenges:
- Server crashes / downtime
- Incorrect binary path (`server` vs `llama-server`)
- Need for rebuild in some cases

---

### 🔹 Parallel Processing

Implemented:
- `ThreadPoolExecutor`

Goals:
- Speed up inference and validation

Challenges:
- Race conditions avoided
- Debugging harder due to async execution
- Logging became noisy

---

## 📦 Data Storage

Used:
- `.parquet` format

Benefits:
- Efficient
- Structured
- Easy integration with pandas

Issues encountered:
- File appeared missing via `ls`
- Binary format unreadable via `cat`
- Required pandas for inspection

---

## 🔍 Validation Phase (Major Exploration)

### 🎯 Goal
Improve dataset quality beyond MBPP pass@1

---

## ❌ Approach 1: LLM-Generated Test Cases

### Idea:
- Use LLM to generate additional test inputs
- Validate code against them

### Implementation:
- Prompt model for JSON inputs
- Execute generated code on inputs

---

### Problems:
1. **Invalid Input Types**
   - Strings expected → integers passed
   - Lists expected → scalars passed

2. **Distribution Mismatch**
   - Generated inputs did not match problem constraints

3. **False Negatives**
   - Correct code failed due to bad inputs

---

### Conclusion:
❌ Rejected — unreliable validation signal

---

## ❌ Approach 2: Consistency Checking

### Idea:
- Run function twice on same input
- Compare outputs

---

### Problems:
- Not meaningful for deterministic functions
- Did not detect logical correctness
- Added unnecessary filtering

---

### Conclusion:
❌ Rejected — low value

---

## ❌ Approach 3: Strict Execution Filtering

### Idea:
Keep only samples that:
- Execute successfully
- Produce consistent outputs
- Pass all generated tests

---

### Problems:
- Over-filtering
- Pass rate dropped to **0%**
- Execution pipeline fragile:
  - Markdown issues
  - Function extraction issues
  - Input mismatch

---

### Key Debugging Issues:

#### 1. Markdown Wrapping
```python
```python
def func():


→ Required cleaning

---

#### 2. Function Extraction Failures
- Multiple functions
- Test code mixed with solution

---

#### 3. Execution Environment
- Missing builtins initially
- Caused silent failures

---

#### 4. Input Shape Bug
```python
func(*inp)   # WRONG
func(inp)    # CORRECT

### 5. Timeout Constraints
Too strict → valid code rejected

Final Outcome:

❌ Entire approach abandoned

Key Insight (Critical Turning Point)

MBPP test cases are already the best validation signal available

Attempting to:

re-validate
augment validation

👉 Introduced noise instead of improving quality

✅ Final Strategy (Adopted)
Step 1: Use Ground Truth
df = df[df["passed"] == True]

Step 2: Clean Code Only
Remove markdown
Extract first function
Discard malformed outputs

Step 3: Build Final Dataset

Result:

Final dataset size: 612

Final Dataset Quality
Metric	Value
Total samples	964
Passed	625
Final cleaned	612
Retention	~98%

Lessons Learned
1. Over-validation is harmful

Trying to exceed benchmark validation:
→ degraded dataset quality

2. Execution pipelines are fragile

Small issues caused total failure:

input shape
environment
parsing
3. Benchmark trust is critical

MBPP already provides:

robust validation
standardized evaluation
4. Simplicity wins

Final working pipeline was:

filter + clean
5. Engineering vs Research tradeoff

We explored:

multiple validation strategies

But:
→ simplest approach was most effective

Current Status

✅ Teacher dataset ready (612 samples)
✅ Clean code extracted
⏭️ Ready for distillation (P4)

Next Phase: P4
LoRA training
Student model (1B)
Evaluate improvement over baseline

Future Work (Optional)
Error-aware distillation
Instruction tuning expansion
RAG integration (later stage)
VSCode plugin deployment

