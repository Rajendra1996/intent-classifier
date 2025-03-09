const fs = require("fs");
const OpenAI = require("openai");
const natural = require("natural");
const { TfIdf } = natural;
require("dotenv").config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002";
const DATASET_FILE = "./intentDataset.json";
const EMBEDDINGS_FILE = "./embeddings.json";

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

class IntentClassifier {
    constructor(config = {}) {
        this.useOpenAI = config.useOpenAI !== undefined ? config.useOpenAI : true;
        this.threshold = config.threshold || 0.50;
        this.batchSize = config.batchSize || 50;
        this.cache = new Map(); // simple cache for repeated queries
        this.embeddingsFile = config.embeddingsFile || "./embeddings.json";
        this.dataset = Array.isArray(config.dataset) ? config.dataset : this.loadDataset();
        this.embeddings = this.loadEmbeddings();
        this.tfidf = new TfIdf();
        this.dataset.forEach(({ text }) => this.tfidf.addDocument(text));
    }

    loadDataset() {
        return fs.existsSync(DATASET_FILE) ? JSON.parse(fs.readFileSync(DATASET_FILE)) : [];
    }

    loadEmbeddings() {
        return fs.existsSync(EMBEDDINGS_FILE) ? JSON.parse(fs.readFileSync(EMBEDDINGS_FILE)) : [];
    }

    async getBatchEmbeddings(textArray) {
        try {
            console.log("🔍 Sending batch request to OpenAI...");
            const response = await openai.embeddings.create({
                model: OPENAI_EMBEDDING_MODEL,
                input: textArray,
            });

            if (!response.data || response.data.length === 0) {
                throw new Error("No embeddings returned by OpenAI.");
            }

            return response.data.map((entry) => entry.embedding);
        } catch (error) {
            console.error("❌ OpenAI Embedding Error:", error);
            throw error;
        }
    }

    async generateEmbeddings() {
        let embeddingsData = [];
        console.log("🔍 Starting batch embedding generation...");

        for (let i = 0; i < this.dataset.length; i += this.batchSize) {
            console.log(`⏳ Processing batch ${i + 1} to ${i + this.batchSize}...`);
            const batch = this.dataset.slice(i, i + this.batchSize).map(item => item.text);

            try {
                const batchEmbeddings = await this.getBatchEmbeddings(batch);
                batchEmbeddings.forEach((embedding, index) => {
                    embeddingsData.push({
                        text: this.dataset[i + index].text,
                        intent: this.dataset[i + index].intent,
                        vector: Array.from(embedding),
                    });
                });
                console.log(`✅ Batch ${i + 1} to ${i + this.batchSize} processed.`);
            } catch (err) {
                console.error(`❌ Error processing batch ${i + 1} to ${i + this.batchSize}:`, err);
            }
        }

        if (embeddingsData.length > 0) {
            console.log("📁 Saving embeddings...");
            fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(embeddingsData, null, 2));
            console.log("✅ Embeddings successfully saved.");
            this.embeddings = embeddingsData; // update in-memory embeddings
        } else {
            console.error("❌ No embeddings generated. OpenAI might not be responding.");
        }
    }

    async isValidQueryOpenAI(query) {
        try {
            const response = await openai.chat.completions.create({
                model: "gpt-4-turbo",
                messages: [
                    {
                        role: "system",
                        content: "You are a query classifier. Evaluate the following query for its coherence and meaning. Return exactly 'valid' if the query is a coherent, meaningful question or statement. Return exactly 'invalid' if the query is nonsensical, random characters, or does not contain any meaningful words. Do not include any additional text."
                    },
                    {
                        role: "user",
                        content: `Query: "${query}"\n\nIs this query valid or invalid?`
                    }
                ],
                max_tokens: 5,
                temperature: 0
            });

            const result = response.choices[0].message.content.trim().toLowerCase();
            return result === "valid";
        } catch (error) {
            console.error("⚠️ OpenAI Query Validation Error:", error);
            return false;
        }
    }

    async addIntent(text, intent) {
        if (!text || !intent) throw new Error("Both text and intent are required.");
        if (this.dataset.some(entry => entry.text === text)) {
            console.log(`⚠️ Intent already exists for: "${text}"`);
            return;
        }

        this.dataset.push({ text, intent });
        this.tfidf.addDocument(text);

        if (this.useOpenAI) {
            const embedding = await this.getBatchEmbeddings([text]);
            this.embeddings.push({ text, intent, vector: Array.from(embedding[0]) });
            fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(this.embeddings, null, 2));
        }

        console.log(`✅ Added intent: "${intent}" for text: "${text}"`);
    }

    async removeIntent(intentToRemove) {
        console.log(`🗑 Removing intent: "${intentToRemove}"...`);

        // Remove from dataset
        this.dataset = this.dataset.filter(({ intent }) => intent !== intentToRemove);

        // Rebuild the TF-IDF model
        this.tfidf = new natural.TfIdf();
        this.dataset.forEach(({ text }) => this.tfidf.addDocument(text));

        // Remove from embeddings (if OpenAI is used)
        if (this.useOpenAI) {
            this.embeddings = this.embeddings.filter(({ intent }) => intent !== intentToRemove);
            fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(this.embeddings, null, 2));
        }

        console.log(`✅ Intent "${intentToRemove}" removed successfully.`);
    }

    classifyWithTFIDF(query) {
        let bestMatch = { intent: "unknown", similarity: 0 };
        let queryVector = [];
        this.tfidf.tfidfs(query, (i, measure) => queryVector.push(measure));

        this.dataset.forEach(({ text, intent }) => {
            let docVector = [];
            this.tfidf.tfidfs(text, (i, measure) => docVector.push(measure));
            let similarity = this.cosineSimilarity(queryVector, docVector);
            if (similarity > bestMatch.similarity) {
                bestMatch = { intent, similarity };
            }
        });

        const confidence = (bestMatch.similarity * 100).toFixed(2) + "%";
        return bestMatch.similarity >= this.threshold
            ? { intent: bestMatch.intent, confidence }
            : { intent: "unknown", confidence };
    }

    async classifyWithOpenAI(query) {
        // Heuristic check: if the query has no vowels, consider it garbage.
        if (!/[aeiou]/i.test(query)) {
            console.log("❌ Query heuristic failed: no vowels detected in query:", query);
            return { intent: "unknown", confidence: "0%" };
        }

        if (!(await this.isValidQueryOpenAI(query))) {
            console.log(`❌ OpenAI detected garbage input: "${query}"`);
            return { intent: "unknown", confidence: "Garbage Detected" };
        }
    
        const queryEmbedding = await this.getBatchEmbeddings([query]);
        let bestMatch = { intent: "unknown", similarity: 0 };
    
        for (let { text, intent, vector } of this.embeddings) {
            let similarity = this.cosineSimilarity(queryEmbedding[0], vector);
            if (similarity > bestMatch.similarity) {
                bestMatch = { intent, similarity };
            }
        }
    
        const confidence = (bestMatch.similarity * 100).toFixed(2) + "%";
        console.log(`🔍 [OpenAI] Cosine Similarity for "${query}": ${bestMatch.similarity}`);
    
        if (bestMatch.similarity < this.threshold) {
            console.log(`❌ Low confidence (${confidence}). Returning 'unknown'.`);
            return { intent: "unknown", confidence };
        }
    
        // Additional check: compute Jaccard similarity between the query and the best match's text.
        const bestEntry = this.embeddings.find(e => e.intent === bestMatch.intent);
        if (bestEntry) {
            const bestText = bestEntry.text;
            const queryWords = new Set(query.toLowerCase().split(/\W+/).filter(Boolean));
            const bestTextWords = new Set(bestText.toLowerCase().split(/\W+/).filter(Boolean));
            const intersection = new Set([...queryWords].filter(x => bestTextWords.has(x)));
            const union = new Set([...queryWords, ...bestTextWords]);
            const jaccard = union.size > 0 ? intersection.size / union.size : 0;
            console.log(`🔍 Jaccard similarity between query and best match: ${jaccard.toFixed(2)}`);
            if (jaccard < 0.2) {
                console.log(`❌ Low Jaccard similarity (${jaccard.toFixed(2)}). Returning 'unknown'.`);
                return { intent: "unknown", confidence };
            }
        }
    
        return { intent: bestMatch.intent, confidence };
    }
    
    async classify(query) {
        if (this.cache.has(query)) {
            return this.cache.get(query);
        }
        const result = this.useOpenAI 
            ? await this.classifyWithOpenAI(query)
            : this.classifyWithTFIDF(query);
        this.cache.set(query, result);
        return result;
    }

    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] ** 2;
            normB += vecB[i] ** 2;
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

module.exports = IntentClassifier;
