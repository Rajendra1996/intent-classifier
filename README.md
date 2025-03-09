# 📌 Intent Classifier - AI-powered Intent Recognition

## 🚀 Overview
Intent Classifier is a lightweight and efficient library for classifying user queries into predefined intents. It supports both **TF-IDF (Local Model)** and **OpenAI Embeddings** for intent matching. The library allows users to dynamically add or remove intents, cache frequent queries, and choose between algorithms for better flexibility.

## 📦 Installation

```sh
npm install intent-classifier
```

## 🔧 Usage

### **1️⃣ Basic Example - TF-IDF Based Matching**

```javascript
const IntentClassifier = require("intent-classifier");

const classifier = new IntentClassifier({
    useOpenAI: false, // Use TF-IDF-based classification
    dataset: [
        { text: "How do I book a flight?", intent: "book_flight" },
        { text: "I need to reserve a ticket", intent: "book_flight" },
        { text: "What's the weather like?", intent: "weather_query" }
    ]
});

async function run() {
    const result = await classifier.classify("How do I book a flight?");
    console.log(result); // { intent: 'book_flight', confidence: '98.42%' }
}

run();
```

### **2️⃣ OpenAI Embeddings-Based Matching**

```javascript
const classifier = new IntentClassifier({
    apiKey: "your-openai-api-key", // Required for OpenAI embeddings
    useOpenAI: true,
    dataset: [
        { text: "How do I reset my password?", intent: "password_reset" },
        { text: "I want to create an account", intent: "account_creation" }
    ]
});

async function run() {
    await classifier.generateEmbeddings(); // Generate embeddings for dataset
    const result = await classifier.classify("How do I reset my password?");
    console.log(result); // { intent: 'password_reset', confidence: '99.21%' }
}

run();
```

## ⚙️ **Configuration Options**
| Option           | Type     | Default | Description |
|-----------------|----------|---------|-------------|
| `useOpenAI`     | Boolean  | `false` | Use OpenAI embeddings for classification |
| `apiKey`        | String   | `null`  | OpenAI API key (Required if `useOpenAI` is `true`) |
| `threshold`     | Number   | `0.50`  | Confidence threshold for intent classification |
| `batchSize`     | Number   | `50`    | Batch size for OpenAI embeddings |
| `embeddingsFile`| String   | `./embeddings.json` | Path to store OpenAI-generated embeddings |

## 📌 **Methods**
### 🔍 `classify(query: string): Promise<{ intent: string, confidence: string }>`
Classifies a user query into an intent.

### 🔍 `addIntent(text: string, intent: string): Promise<void>`
Dynamically adds a new intent to the dataset and updates embeddings if using OpenAI.

### 🔍 `removeIntent(intent: string): Promise<void>`
Removes an intent from the dataset and updates embeddings if using OpenAI.

### 🔍 `generateEmbeddings(): Promise<void>`
Generates and saves embeddings for the dataset (Only for OpenAI mode).

## 🚀 **Additional Features**
- **Garbage Query Detection**: Uses heuristics and OpenAI to filter out nonsensical queries.
- **Jaccard Similarity Filtering**: Prevents force-matching by checking word similarity.
- **Query Caching**: Avoids redundant API calls for repeated queries.

## 🛠 **Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📝 **License**
This project is licensed under the MIT License.

