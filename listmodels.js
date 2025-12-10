// Check available models via REST API
import * as dotenv from "dotenv";
dotenv.config();

async function listAvailableModels() {
    const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GEMINI_API_KEY}`
    );
    const data = await response.json();
    console.log('Available models:', data.models.map(m => m.name).join('\n'));
}

listAvailableModels();