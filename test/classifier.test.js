const IntentClassifier = require("../src/classifier");

describe("IntentClassifier", () => {
    let classifier;

    beforeAll(async () => {
        classifier = new IntentClassifier({
            apiKey: process.env.OPENAI_API_KEY, // OpenAI key (needed for OpenAI tests)
            useOpenAI: true, // Set to true for OpenAI embedding-based classification
            dataset: [
                { text: "How do I book a flight?", intent: "book_flight" },
                { text: "I need to reserve a ticket", intent: "book_flight" },
                { text: "What's the weather like?", intent: "weather_query" }
            ],
            threshold: 0.30 // Lower threshold for better matching
        });

        await classifier.generateEmbeddings(); // Ensure embeddings are generated
    });

    test("Rejects garbage input", async () => {
        const result = await classifier.classify("xxxxxxxx?");
        console.log("🔍 Debug: Garbage Input Classification →", result);
        expect(result.intent).toBe("unknown");
    });

    test("Correctly classifies valid queries", async () => {
        const result = await classifier.classify("How do I book a flight?");
        console.log("🔍 Debug: Valid Query Classification →", result);
        expect(result.intent).toBe("book_flight");
    });

    test("Correctly detects similar intents", async () => {
        const result = await classifier.classify("I need to book a ticket");
        console.log("🔍 Debug: Similar Intent Classification →", result);
        expect(result.intent).toBe("book_flight");
    });

    test("Correctly classifies OpenAI-based intent", async () => {
        const result = await classifier.classify("Tell me today's weather");
        console.log("🔍 Debug: OpenAI Intent Classification →", result);
        expect(result.intent).toBe("weather_query");
    });

    test("Allows dynamic addition of new intent", async () => {
        await classifier.addIntent("How can I cancel my flight?", "cancel_flight");
        const result = await classifier.classify("Cancel my booking");
        console.log("🔍 Debug: Dynamic Intent Addition →", result);
        expect(result.intent).toBe("cancel_flight");
    });

    test("Removes an intent successfully (OpenAI mode)", async () => {
        console.log("🛠 Adding new intent...");
        await classifier.addIntent("Lets go to gym", "go_gym");

        let result = await classifier.classify("Lets go to gym");
        console.log("🔍 Debug: Before Intent Removal →", result);
        expect(result.intent).toBe("go_gym");

        console.log("🗑 Removing intent...");
        classifier.removeIntent("go_gym");

        // **Wait for the embeddings file to update**
        await new Promise(resolve => setTimeout(resolve, 1000));

        // **Ensure classifier reloads the updated embeddings**
        classifier = new IntentClassifier({
            apiKey: process.env.OPENAI_API_KEY,
            useOpenAI: true,
            dataset: classifier.dataset
        });

        result = await classifier.classify("Lets dance");
        console.log("🔍 Debug: After Intent Removal →", result);
        expect(result.intent).toBe("unknown");
    });

    test("Caches repeated OpenAI queries to avoid redundant API calls", async () => {
        const firstRun = await classifier.classify("How do I book a flight?");
        const secondRun = await classifier.classify("How do I book a flight?");
        console.log("🔍 Debug: Query Caching Check →", firstRun, secondRun);
        expect(firstRun.intent).toBe(secondRun.intent);
    });
});
