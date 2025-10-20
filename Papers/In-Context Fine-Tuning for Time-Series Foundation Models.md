[PDF Link](https://openreview.net/pdf?id=uxzgGLWPj2)
**# Zero-shot forecasting:

Zero-shot forecasting is a machine learning approach where a model makes predictions for a time series or sequence without having been explicitly trained on that specific task or dataset. Instead, it leverages pre-trained knowledge from related tasks or domains to generalize to new forecasting problems.

Key Characteristics:

1. No Task-Specific Training – Unlike traditional forecasting models (e.g., ARIMA, Prophet, or LSTM trained on a specific dataset), zero-shot forecasting does not require fine-tuning on the target data.

2. Leverages Pre-trained Models – Uses large-scale pre-trained models (e.g., transformers, foundation models) that have learned general patterns from diverse datasets.

3. Generalisation Across Domains – Can forecast for unseen time series by understanding underlying structures (e.g., trends, seasonality) from prior training.

  

How It Works:

- A model is pre-trained on a wide variety of time-series data (e.g., retail sales, weather, stock prices).

- When given a new, unseen time series, the model generates forecasts without additional training, relying on its learned representations.

  

Examples of Zero-Shot Forecasting Models:

- Large Language Models (LLMs) like GPT-4 can be prompted to generate forecasts based on given historical data.

- Time-Specific Foundation Models like Google’s TimesFM (a zero-shot time-series forecasting model trained on 100B+ real-world time points).

  

Advantages:

✅ No need for model retraining on new data.  

✅ Useful when historical data is scarce.  

✅ Can adapt quickly to new forecasting tasks.  

  

Limitations:

⚠️ May underperform compared to models fine-tuned on domain-specific data.  

⚠️ Relies heavily on the diversity and quality of pre-training data.  

  

Use Cases:

- Retail: Predicting sales for a new product with no prior sales data.

- Finance: Forecasting stock prices for newly listed companies.

- Energy: Estimating electricity demand in a new region.

  

Comparison with Other Methods:

| Method           | Requires Training on Target Data?       | Example Models |

|------------------|----------------------------------|----------------    |

| Traditional Forecasting | Yes (e.g., ARIMA, Prophet)    | SARIMA, ETS |

| Few-Shot Forecasting | Minimal training data needed  | NeuralProphet, N-BEATS |

| Zero-Shot Forecasting | No training needed                 | TimesFM, LLMs (GPT-4) |

  
  

# Time series forecasting and time series foundation models:

## Time Series Data  

Time series data is a sequence of data points collected or recorded at specific time intervals. Each data point is associated with a timestamp, allowing analysis of trends, patterns, and dependencies over time.  

  

 Key Characteristics of Time Series Data:  

- Temporal Ordering – Data points are recorded sequentially (e.g., hourly, daily, monthly).  

- Trends & Seasonality – Many time series exhibit long-term trends (upward/downward movement) and seasonal patterns (repeating cycles).  

- Irregular Noise – Random fluctuations that are not part of the systematic pattern.  

  

 Examples of Time Series Data:  

- Economic: Stock prices, GDP growth, unemployment rates.  

- Business: Daily sales, website traffic, and inventory levels.  

- Scientific: Weather measurements (temperature, rainfall), seismic activity.  

- Industrial: Sensor readings from machines (vibration, temperature).  

  
  

## Time Series Foundation Models  

Time series foundation models are large-scale machine learning models pre-trained on vast amounts of time series data, designed to generalize across multiple forecasting tasks without task-specific fine-tuning. They are inspired by large language models (LLMs), such as GPT, but adapted for sequential numerical data.  

  

 Key Features:  

1. Pre-trained on Diverse Datasets – Trained on billions of time steps from different domains (e.g., finance, weather, IoT).  

2. Zero/Few-Shot Forecasting – Can make predictions on unseen time series without retraining.  

3. Transfer Learning – Knowledge from one domain (e.g., retail sales) can help in another (e.g., energy demand).  

  

 Examples of Time Series Foundation Models:  

- TimesFM (Google) – A 200M-parameter model trained on 100B+ time steps for zero-shot forecasting.  

- Lag-Llama (Open-source) – A foundation model for probabilistic time series forecasting.  

- Moirai (Amazon) – A unified model for forecasting across different frequencies (hourly, weekly, etc.).  

  

 How They Differ from Traditional Models:  

| Aspect          | Traditional Models (ARIMA, Prophet) | Time Series Foundation Models |  

|----------------------|----------------------------------------|-----------------------------------|  

| Training Data    | Needs task-specific data               | Pre-trained on massive datasets   |  

| Adaptability     | Requires retraining for new tasks      | Zero/few-shot forecasting         |  

| Scalability      | Limited to single datasets             | Generalizes across domains        |  

  

 Applications:  

- Demand Forecasting – Predicting sales for new products.  

- Anomaly Detection – Identifying unusual patterns in sensor data.  

- Energy Forecasting – Estimating electricity load without historical data.  

# ARIMA:

ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model used for time series forecasting. It works by analyzing past data to predict future values, assuming that historical patterns (trends, seasonality) will continue.  

  

How ARIMA Works (3 Key Parts)  

ARIMA combines three components:  

  

1. AutoRegressive (AR) – "Learn from Past Values"  

- Predicts future values based on past values (like a regression model).  

- Example: If sales were high last month, they’ll likely be high next month.  

-Parameter (p): Number of past values used (e.g., AR(2) uses the last 2 data points).  

  

2. Integrated (I) – "Make Data Stationary"  

- Removes trends/seasonality by differencing (subtracting past values).  

- Example: If sales increase by 10 units each month, differencing converts it to a flat line.

-Parameter (d): Number of times differencing is applied.  

  

3. Moving Average (MA) – "Learn from Past Errors"  

- Predicts future values based on past forecast errors (unlike AR, which uses past values).  

- Example: If the model underpredicted sales last month, it adjusts future predictions upward.  

-Parameter (q): Number of past errors considered.  

  

ARIMA in Action  

Step 1: Make the data stationary (remove trends/seasonality) using differencing (I).  

Step 2: Fit an AR model (past values) and an MA model (past errors).  

Step 3: Combine them to forecast future values.  

  

Example: Predicting Monthly Sales  

-AR(1): Uses last month’s sales.  

-I(1): Subtracts last month’s sales to remove trend.  

-MA(1): Adjusts for last month’s prediction error.  

-Final Forecast: `AR + MA` applied to differenced data.  

  

When to Use ARIMA?  

✅Non-seasonal data (for seasonal data, use SARIMA).  

✅Small datasets (unlike deep learning, which needs big data).  

✅Linear patterns (fails with complex nonlinear trends).  

  
  

Limitations  

⚠️ Manual tuning of p, d, q is required (use ACF/PACF plots or auto-ARIMA).  

⚠️ Assumes data is stationary (real-world data often isn’t).  

⚠️ Struggles with unpredictable events (e.g., COVID disrupting trends).  

  
  

ARIMA vs. Modern Models  

| Model  | Pros | Cons |  

|--------|------|------|  

|ARIMA | Simple, interpretable | Manual tuning, linear assumptions |  

|Deep Learning (LSTM) | Handles complex patterns | Needs large data, hard to interpret |  

|Foundation Models (TimesFM) | Zero-shot forecasting | Less control, compute-heavy |  

  

# Inference Time:

Inference time refers to the phase when a trained machine learning model makes predictions on new, unseen data. Unlike training (where the model learns from data), inference is about applying the learned knowledge to generate outputs.

  

Key Points:

- Happens after model training.
    
- Involves feeding input data (e.g., past time series values) and getting predictions (e.g., future forecasts).
    
- Speed matters: Low inference time = faster predictions (critical for real-time apps).
    

  

# Inference:

 Inference in Machine Learning: A Deep Dive with Diverse Examples  

Inference is the stage where a trained model applies its learned knowledge to make predictions on new, unseen data. Unlike training (which adjusts model weights), inference is about execution—using the model to generate outputs.  

  

1. How Inference Works  

 Key Steps:  

- Input Data → Fed to the model (e.g., past time series values, an image, text prompt).  
    
- Forward Pass → Model computes output using learned parameters (no weight updates).  
    
- Prediction → Returns results (e.g., forecasted sales, classified image, generated text).  
    

  

Technical Flow:  

```

Input → Model (Forward Pass) → Output  

          (No Backpropagation)

``

  

 2. Diverse Examples of Inference  

  

 Example 1: Time Series Forecasting  

- Model: ARIMA / TimesFM (foundation model).  

- Input: `[Day 1: 100, Day 2: 110, Day 3: 120]`  

- Inference Task: Predict Day 4.  

- Output: `130` (if trend = +10/day).  

  

Inference-Time Adaptation:  

- Provide context examples during inference (no fine-tuning):  

  ```python

  input_context = """

  [2023-01-01: 50 → 2023-01-02: 55]

  [2023-01-03: 60 → 2023-01-04: 65]

  [2023-01-05: 70 → 2023-01-06: ?]

  """

  model.predict(input_context)   Output: 75

  ``

  

 Example 2: Computer Vision (Image Classification)  

- Model: ResNet (pre-trained on ImageNet).  

- Input: A new image of a cat.  

- Inference Task: Classify the image.  

- Output: `"Cat"` (with 95% confidence).  

  

Inference-Time Trick:  

- Use test-time augmentation (TTA) to improve accuracy:  

  - Predict on multiple augmented versions of the image (flipped, rotated) and average results.

  

 Example 3: Natural Language Processing (Text Generation)  

- Model: GPT-4.  

- Input: `"Translate to French: 'Hello'"`  

- Inference Task: Generate French translation.  

- Output: `"Bonjour"`.  

  

Inference-Time Optimization:  

- Few-shot prompting: Provide examples at inference to guide output:  

  ```python

  prompt = """

  English: "Hi" → French: "Salut"

  English: "Goodbye" → French: "Au revoir"

  English: "Hello" → French: ?

  """

  model.generate(prompt)   Output: "Bonjour"

  ```

  

 Example 4: Reinforcement Learning (Game AI)  

- Model: AlphaGo.  

- Input: Current board state in Go.  

- Inference Task: Choose next move.  

- Output: Places stone at position `(D, 4)`.  

  

Inference-Time Adjustment:  

- Temperature sampling: Adjust randomness of moves:  

  - Low temp = deterministic (best move).  

  - High temp = exploratory (risky moves).

  

 3. Inference vs. Training  

| Aspect          | Training Phase               | Inference Phase               |  

|-----------------|------------------------------|--------------------------------|  

| Goal        | Learn from data (update weights) | Apply learned knowledge (no weight updates) |  

| Compute Cost| High (GPU/TPU needed)        | Lower (can run on edge devices) |  

| Latency     | Slow (hours/days)            | Fast (milliseconds)            |  

| Examples    | Fine-tuning GPT-3 on emails  | GPT-3 generating email replies | 

  

 4. Optimizing Inference  

 A. Model Compression  

- Quantization: Reduce precision (32-bit → 8-bit) for faster inference.  

- Pruning: Remove unused neurons to shrink model size.  

  

 B. Hardware Acceleration  

- GPUs/TPUs: Parallel processing for batched inputs.  

- Edge AI: Deploy lightweight models on phones (e.g., TensorFlow Lite).  

  

 C. Caching & Batching  

- Cache frequent queries (e.g., weather forecasts).  

- Batch inputs for efficiency (e.g., process 100 images at once).

  

 5. Real-World Applications  

| Industry       | Inference Use Case                         | Model Example         |  

|---------------|--------------------------------------------|-----------------------|  

| Healthcare | Diagnosing X-rays (pneumonia detection)    | DenseNet-121          |  

| Finance   | Fraud detection (real-time transactions)   | Random Forest         |  

| Retail    | Demand forecasting (next-week sales)       | ARIMA / TimesFM       |  

| Autonomous Cars | Object detection (pedestrians, traffic signs) | YOLOv9        |  

  

 Key Takeaways  

1. Inference = Using a trained model to make predictions (no learning).  

2. Inference-time tricks (few-shot prompts, TTA) can mimic fine-tuning.  

3. Speed & efficiency matter (quantization, batching, hardware acceleration).  

4. Works across domains—time series, NLP, CV, robotics.  

# Why extra context helps in zero shot forecasting?

Simple Explanation: Why Extra Context Helps in Zero-Shot Forecasting  

  

Imagine you’re trying to predict tomorrow’s traffic on a highway you’ve never seen before. You’re given only last week’s traffic data for that highway. But:  

- The model wasn’t trained on this specific highway’s patterns.  

- Last week’s data might be noisy (e.g., a construction event messed up the usual flow).  

  

Problem: The model doesn’t have enough "experience" to generalize accurately.  

  

Solution: Add More "Examples" to the Prompt  

Instead of just showing last week’s data, give the model additional context:  

1. Traffic data from other highways (similar roads, different locations).  

2. Traffic data from other weeks (to capture trends, like rush hour patterns).  

  

Why This Works:  

- The model starts to "notice" common patterns across highways (e.g., "Ah, traffic always peaks at 8 AM!").  

- It can compare the new highway’s data to past examples and adjust predictions.

  

Real-World Analogy  

Think of it like teaching someone to predict pizza delivery times:  

- Weak Prompt (Fails):  

  "Here’s how long one pizza took last Thursday. Guess tomorrow’s time!"  

  → They’ll probably be wrong (too little info).  

  

- Strong Prompt (Works):  

  "Here’s last week’s pizza times, plus data from 10 other restaurants. Now guess!"  

  → They’ll notice patterns (e.g., "Fridays are always slower") and predict better. 

  

Key Idea  

Zero-shot models don’t learn from your data—they reason from context. More (relevant) examples = better reasoning!  

# Main contribution of the paper:

1. Introduces the study of in-context fine-tuning for time series foundation models, and propose the use of prompts that oth include usual history and related time-series examples in-context as well. Training is decoder-only and can adapt to varying history and horizon lengths. Resulting model can learn to borrow patterns from related examples.
    
2. Evaluates the benefits of in-context fine-tuning using our foundation model, and show that in-context fine tuning can lead to etter zero-shot performance on popular forecasting enchmarks as compared to other methods.
    

# Related Work

Related works can be broadly divided into three categories.

1. Prompting LLMs to directly predict the future of a numerical series encoded as text.
    
2. Fine-tuning pretrained LLMS on time-series data with adapter layers
    
3. Pretraining transformer ased models from scratch on huge volumes of time-series data.
    

# Strong Zero-Shot Performance Achieved by Stacked Transformer Models in Decoder-Only Mode for Time-Series Forecasting

  

This statement highlights a breakthrough in time-series forecasting, where a decoder-only transformer architecture (similar to GPT for text) achieves high accuracy without task-specific training (zero-shot). Here’s what it means:

  

1. Key Components Explained  

(A) Stacked Transformer Models  

- Transformers: Neural networks that process sequential data (like time-series) using self-attention to weigh important patterns.  

- Stacked: Multiple transformer layers are stacked deep (e.g., 12+ layers) to learn complex temporal relationships.  

  

(B) Decoder-Only Mode  

- Borrowed from LLMs like GPT, this architecture:  

  - Uses masked self-attention (each step only sees past data, not future).  

  - Autoregressively predicts future values step-by-step (like generating text).  

  

(C) Zero-Shot Performance  

- The model generalizes to new time-series tasks without fine-tuning.  

- Example: Predict electricity demand for a new city without training on its data.

  

2. Why This Works for Time-Series  

- Self-Attention: Captures long-range dependencies (e.g., weekly/yearly cycles).  

- Autoregressive Decoding: Predicts future values iteratively (like "predict next day → feed it back → predict next-next day").  

- Pretraining on Diverse Data: The model learns universal time-series patterns (trends, seasonality) from massive datasets.

  

3. Example Workflow  

4. Input: Past 30 days of sales data (`y₁:₃₀`).  

5. Decoder-Only Prediction:  

   - Step 1: Predict `y₃₁` using `y₁:₃₀`.  

   - Step 2: Predict `y₃₂` using `y₁:₃₁` (now includes its own prediction).  

   - Repeat for horizon `H`.

  

4. Advantages Over Traditional Models  

| Feature | Traditional Models (ARIMA) | Decoder-Only Transformer |  

|---------|----------------------------|--------------------------|  

| Training | Needs per-task training | Pretrained once, zero-shot |  

| Horizon Flexibility | Fixed `H` | Any `H` (autoregressive) |  

| Pattern Capture | Linear trends | Complex, nonlinear trends |  

  

5. Real-World Applications  

- Retail: Predict demand for new products.  

- Energy: Forecast grid load for unseen households.  

- Finance: Zero-shot stock price predictions.  

  

Key Takeaway  

Decoder-only transformers (like GPT for time-series) leverage pretraining + autoregressive decoding to achieve strong zero-shot forecasting, eliminating the need for model retraining.  

  

(Think of it as "ChatGPT for time-series data.")  

  
  

# Decoder Only Transformer vs Encoder Only Transformer vs Transformer having both encoder & decoder:

  

## 1. Decoder-Only Transformer  

Structure:  

- Uses only the decoder stack from the original Transformer.  

- Masked self-attention: Each token (or time-step) can only attend to past tokens (causal masking).  

  

Causal Masking = Causal masking in transformers ensures that each token (or time-step) can only attend to previous tokens in the sequence, preventing access to future information. This is critical for:  

1. Autoregressive generation (e.g., GPT, time-series forecasting).  

2. Preserving temporal causality (no "cheating" by peeking ahead).  

  

Mechanism:  

- Applies a lower-triangular mask to the attention scores.  

- Mask value `-∞` (or a large negative number) zeros future positions in softmax.  

  

Example:  

For input `[y₁, y₂, y₃]`, `y₂` can only attend to `y₁` and `y₂`, not `y₃`.

  

Key Point:  

Enables sequential prediction (e.g., next-word or next-time-step forecasts).

Key Features:  

- Autoregressive: Predicts outputs step-by-step (e.g., GPT for text, TimesFM for time series).  

- Zero-shot friendly: Pretrained on diverse data, generalizes to new tasks without fine-tuning.  

  

Use Cases:  

- Time-series forecasting (e.g., predict next `H` steps iteratively).  

- Text generation (GPT).  

  

Example:  

```python

 Pseudocode for decoder-only forecasting

for t in range(horizon_H):

    y_L+t = model(y_1:L + y_L+1:L+t-1)   Uses past predictions

```

  

## 2. Encoder-Only Transformer  

Structure:  

- Uses only the encoder stack (no decoder).  

- Bidirectional self-attention: Each token attends to all tokens in the input (no masking).  

  

Key Features:  

- Non-autoregressive: Processes entire input at once.  

- Feature extraction: Outputs embeddings for classification/regression.  

  

Use Cases:  

- Time-series classification (e.g., anomaly detection).  

- BERT-style models for NLP (contextual embeddings).  

  

Example:  

```python

 Pseudocode for encoder-only tasks

embeddings = encoder(y_1:L)   Processes full history

forecast = linear_layer(embeddings)   Direct prediction

```

  

3. Full Encoder-Decoder Transformer  

Structure:  

- Combines both encoder and decoder (original Transformer design).  

- The encoder processes input, decoder generates output step-by-step.  

  

Key Features:  

- Handles paired sequences: Input (e.g., past data) → Output (e.g., future forecasts).  

- Cross-attention: Decoder attends to the encoder’s outputs.  

  

Use Cases:  

- Seq2Seq tasks (e.g., machine translation).  

- Time-series imputation (corrupt input → clean output).  

  

Example:  

```python

 Pseudocode for encoder-decoder forecasting

encoder_output = encoder(y_1:L)  

for t in range(horizon_H):

    y_L+t = decoder(encoder_output, y_L+1:L+t-1)   Uses encoder context

```

  

---

  

Comparison Table  

| Feature               | Decoder-Only           | Encoder-Only          | Encoder-Decoder       |  

|-----------------------|------------------------|-----------------------|-----------------------|  

| Attention            | Masked (causal)        | Bidirectional         | Encoder: Bidirectional    <br>                                                                    Decoder: Masked + Cross-attention |  

| Prediction Mode   | Autoregressive         | One-shot              |  Autoregressive        |  

| Flexibility       | Zero-shot possible     | Needs fine-tuning     | Needs paired data     |  

| Time-Series Use   | Forecasting (GPT-style) | Classification/Embeddings | Missing value imputation |  

| NLP Example       | GPT                    | BERT                            | T5, BART              |  

  

Key Takeaways  

1. Decoder-only: Best for generative tasks (forecasting, text).  

2. Encoder-only: Best for analysis tasks (classification, embeddings).  

3. Encoder-decoder: Best for transformation tasks (translation, imputation).  

  

Time-Series Insight:  

- Decoder-only models (like TimesFM) mimic LLMs—great for zero-shot forecasting.  

- Encoder-decoder models are heavier but useful for complex mappings (e.g., noisy → clean data).  

  

# TimesFM:

A time series model made by Google published in 2024.

  

# Overall Model Simple explanation:

Let me break this down into simple, digestible parts with clear examples. We'll go step-by-step through the model's workflow.

  

 Part 1: Input Preparation (Patch Creation)

  

What's Happening:

1. You have several time series examples (like different stores' sales data)

2. Each series is divided into smaller chunks called "patches"

  

Example:

Imagine we have:

- Store A sales: [10, 15, 8, 12, 9, 14] (6 days)

- Store B sales: [20, 18] (2 days)

With patch length p=3:

  

Store A becomes:

- Patch 1: [10, 15, 8] (days 1-3)

- Patch 2: [12, 9, 14] (days 4-6)

  

Store B becomes:

- Patch 1: [20, 18, PAD] (days 1-2 + padding)

  

Why Patches?

- Like reading a book page-by-page instead of all at once

- Helps handle long sequences efficiently

  

Key Terms:

- p = patch size (like 3 days)

- ȳ⁽ⁱ⁾ⱼ = j-th patch of i-th example

- m̃ = mask (marks padded values)

  

Part 2: Turning Patches into Tokens (Like Words for Time Series)

  

Now that we have our patches (small chunks of time series), we need to convert them into something the transformer can understand - we call these tokens.

  

Step-by-Step Process:

1. Handle Missing Data with Masks:

   - Each patch has a mask (`m̃`) that marks which values are real (0) vs. padded (1).  

   - For Store B's patch `[20, 18, PAD]`, the mask would be `[0, 0, 1]` (PAD is masked).  

  

2. Embed Patches into Tokens:

   - Pass each patch through an MLP (neural network) to create a token:  

     ```

     t⁽ⁱ⁾ⱼ = MLP(ȳ⁽ⁱ⁾ⱼ ⊙ (1 − m̃⁽ⁱ⁾ⱼ))

     ```

     - `⊙` = element-wise multiplication (zeros out padded values).  

     - Example: For `[20, 18, PAD]` and mask `[0, 0, 1]`, the MLP only processes `[20, 18, 0]`.  

  

3. Add Separator Tokens (`σ`):  

   - A special token (like a "full stop" in a sentence) marks the end of each example.  

   - After all patches of Store A, we add `σ`.  

  

Intuitive Example:

- Store A Patches: `[10,15,8]` → Token `t¹₁`  

                      `[12,9,14]` → Token `t¹₂`  

                      Separator → `σ`  

- Store B Patches: `[20,18,PAD]` → Token `t²₁`  

                      Separator → `σ`  

  

Final Token Sequence: `[t¹₁, t¹₂, σ, t²₁, σ]`  

  

(Think of this like converting a book into sentences, then words, and adding periods.)

Key Concepts:  

- Tokens: Numerical representations of patches (like word embeddings in NLP).  

- Masks: Ensure padding doesn’t affect predictions.  

- Separators: Help the model distinguish between examples.  

  

Part 3: Processing Tokens with the Transformer (Like Predicting the Next Word in a Sentence)  

  

Now that we have our tokens, we feed them into a decoder-only transformer (similar to GPT for text). Here’s how it works:

  

1. Autoregressive Prediction (Step-by-Step Forecasting)  

The model predicts one patch at a time, using past patches as context (like predicting the next word in a sentence).  

  

Example: Predicting Store A’s Sales  

- Input Tokens: `[t¹₁ (days 1-3), t¹₂ (days 4-6), σ]`  

- Prediction Steps:  

  1. Takes `t¹₁` → Predicts patch for days 4-6 (`ŷ⁽¹⁾₂`).  

  2. Takes `t¹₁ + t¹₂` → Predicts next patch (e.g., days 7-9).  

  3. Separator `σ` signals "end of example."  

  

*(Just like how GPT predicts the next word based on previous words!)*  

  

2. Transformer’s Secret Sauce: Self-Attention  

- The model weighs the importance of past tokens when making predictions.  

  - Example: To predict `t¹₂`, it might focus more on `t¹₁` (previous patch) and less on `σ` (separator).  

- Uses causal masking to prevent "cheating" (can’t see future patches).  

  

Visualization of Attention  

```

Patch 1 (t¹₁): [10, 15, 8]  

Patch 2 (t¹₂): [12, 9, 14] → Predicted using attention over t¹₁  

Separator (σ) → "Stop predicting for Store A"

```

  

3. Output Prediction  

- Each transformer output (`o⁽ⁱ⁾ⱼ`) is passed through a Residual Layer to generate the final forecast:  

  ```

  ŷ⁽ⁱ⁾_{pj+1:pj+h} = OutputResidualLayer(o⁽ⁱ⁾ⱼ)

  ```  

  - Predicts the next `h` time steps (e.g., next 3 days).  

  

Example Output  

- For `t¹₁` (days 1-3), the model might output: `[11, 10, 9]` (prediction for days 4-6). 

  

4. Loss Function (Training Objective)  

- Compares predictions (`ŷ`) to actual values (`y`) using mean squared error (MSE).  

- Goal: Minimize the difference between predicted and real patches.  

  

Why This Matters  

- Zero-shot Learning: The model generalizes to new stores/time series without retraining.  

- Efficiency: Processes long sequences in chunks (patches).  

- Flexibility: Handles variable-length inputs via masking.  

  

Analogous to GPT  

| Time-Series Model (TimesFM) | GPT (Text) |  

|-----------------------------|------------|  

| Processes patches of sales data | Processes words/sentences |  

| Predicts next patch (days 4-6) | Predicts next word |  

| Uses separators (`σ`) between stores | Uses periods between sentences |  

  

---

  

Final Summary  

1. Input: Raw time series → Split into patches → Convert to tokens.  

2. Processing: Tokens fed to transformer (autoregressive + masked attention).  

3. Output: Predicts next patch → Repeats until separator.  

  

*(This is how TimesFM achieves zero-shot forecasting!)***