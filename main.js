
import * as dotenv from 'dotenv';
dotenv.config();

import { PineconeStore } from '@langchain/pinecone';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const PDF_PATH = './sample.pdf';
const pdfLoader = new PDFLoader(PDF_PATH);
const rawDocs = await pdfLoader.load();

console.log("Loaded Documents:", rawDocs.length);



/// chunking

import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

console.log("Chunked Documents Completed");

// initialize embeddings

const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });




  // dataabase pinecode


const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
console.log("Pinecone Index Initialized");

/// langchainpinecode vector store
await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    namespace: "TEST-namespace",
    maxConcurrency: 5,
  });

console.log("Documents Successfully Stored in Pinecone");

console.log("Documents Indexed in Pinecone");
