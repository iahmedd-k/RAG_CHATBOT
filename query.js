/************************************************************
 * STEP 1: LOAD ENVIRONMENT VARIABLES (.env)
 ************************************************************/
import * as dotenv from "dotenv";
dotenv.config();


import readlineSync from "readline-sync";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";  // Official Gemini SDK


/************************************************************
 * STEP 2: CREATE EMBEDDING MODEL
 * This converts text → numbers (vectors) so Pinecone can search.
 ************************************************************/
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "models/text-embedding-004", // Google’s official embedding model
});


/************************************************************
 * STEP 3: CONNECT TO PINECONE
 ************************************************************/
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

console.log("\n📊 Pinecone Index Stats:");
console.log(JSON.stringify(await pineconeIndex.describeIndexStats(), null, 2));


/************************************************************
 * STEP 4: CHAT MODEL (OFFICIAL GOOGLE SDK)
 * This is the LLM that will answer final user queries.
 ************************************************************/
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const chatModel = genAI.getGenerativeModel({
    model: "gemini-2.0-flash",  // Fast + supported
});


/************************************************************
 * STEP 5: SEARCH DOCUMENTS IN PINECONE
 * RAG STEP 1 = Retrieve relevant chunks using embeddings
 ************************************************************/
async function searchDocuments(query) {
    try {
        console.log("➡️ Generating embedding for query:", query);

        // Convert user question to vector
        const queryVector = await embeddings.embedQuery(query);

        console.log("➡️ Searching Pinecone...");

        // Search inside your namespace
        const searchResults = await pineconeIndex
            .namespace("TEST-namespace")
            .query({
                topK: 5,
                vector: queryVector,
                includeMetadata: true,
            });

        // If no documents found
        if (!searchResults.matches?.length) {
            console.log(" No matching documents found.");
            return "";
        }

        console.log(`\n✔ Found ${searchResults.matches.length} matching documents.\n`);

        // Build context string from retrieved chunks
        return searchResults.matches
            .map((m, i) => `[${i + 1}] ${m.metadata.text || m.metadata.pageContent}`)
            .join("\n\n---\n\n");

    } catch (err) {
        console.error("Error searching Pinecone:", err.message);
        return "";
    }
}


/************************************************************
 * STEP 6: LLM ANSWERING USING RAG
 * RAG STEP 2 = Generate final answer using only retrieved text
 ************************************************************/
async function getAIResponse(userQuery, context) {
    const prompt = `
You are a helpful assistant.

Use ONLY the following document context to answer the user's question.
If the answer is not found in the context, say:
"I could not find the answer in the provided document."

--------------------
DOCUMENT CONTEXT:
${context || "NO CONTEXT FOUND"}
--------------------

USER QUESTION:
${userQuery}
`;

    try {
        // Send prompt to Gemini
        const response = await chatModel.generateContent(prompt);

        // Return the text output
        return response.response.text();

    } catch (err) {
        console.error(" Gemini Chat Error:", err.message);

        // fallback: show retrieved documents
        if (context) {
            return `Chat model unavailable — showing retrieved documents:\n\n${context}`;
        }

        return "Sorry, I could not process your question.";
    }
}


/************************************************************
 * STEP 7: COMPLETE CHAT PROCESS
 ************************************************************/
async function chatting(question) {
    console.log("\n Step 1: Searching documents...");
    const context = await searchDocuments(question);

    console.log("\n Step 2: Generating answer...");
    const answer = await getAIResponse(question, context);

    console.log("\nAssistant:", answer);
}


/************************************************************
 * STEP 8: MAIN INTERACTIVE CHAT LOOP
 ************************************************************/
async function main() {
    console.log(" Welcome to Your RAG Chatbot!");
    console.log("Ask questions about your documents.\nType 'exit' to quit.\n");

    while (true) {
        const q = readlineSync.question("You: ");

        if (q.toLowerCase() === "exit") {
            console.log("👋 Goodbye!");
            process.exit(0);
        }

        await chatting(q);
        console.log("");
    }
}

main();
