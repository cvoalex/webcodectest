import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';
import "dotenv/config";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.text());
app.use(express.json());

const port = process.env.PORT || 3000;
const apiKey = process.env.OPENAI_API_KEY;

// Serve static files from the current directory
app.use(express.static(__dirname));

const sessionConfig = JSON.stringify({
  session: {
    type: "realtime",
    model: "gpt-realtime",
    audio: {
      input: {
        format: "pcm16",
        turn_detection: { type: "semantic_vad", create_response: true }
      },
      output: {
        format: "pcm16", // Use PCM16 for easier processing
        voice: "alloy",
        speed: 1.0
      }
    },
    // Enhanced instructions for better lip sync
    instructions: `You are a helpful AI assistant with natural, expressive speech. 
    Speak clearly and at a moderate pace for optimal lip synchronization. 
    Use natural intonation and pauses. Keep responses conversational and engaging.
    When describing visual content, be vivid and descriptive.`
  },
});

// API route for ephemeral token generation
app.get("/token", async (req, res) => {
  try {
    console.log("🔑 Generating ephemeral token for WebRTC session");
    
    const response = await fetch(
      "https://api.openai.com/v1/realtime/client_secrets",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: sessionConfig,
      },
    );

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log("✅ Ephemeral token generated successfully");
    res.json(data);
  } catch (error) {
    console.error("❌ Token generation error:", error);
    res.status(500).json({ 
      error: "Failed to generate token",
      details: error.message 
    });
  }
});

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    services: {
      openai_api: !!apiKey,
      frame_generator: "external", // Python service
      grpc_service: "external"     // Your gRPC service
    }
  });
});

// API endpoint to get system configuration
app.get("/config", (req, res) => {
  res.json({
    audio: {
      sampleRate: 24000,
      chunkDuration: 40,
      bufferSize: 3000,
      inferenceFrames: 16,
      batchSize: 5
    },
    models: [
      "test_optimized_package_fixed_1",
      "test_optimized_package_fixed_2", 
      "test_optimized_package_fixed_3",
      "test_optimized_package_fixed_4",
      "test_optimized_package_fixed_5"
    ],
    endpoints: {
      frameGenerator: "ws://localhost:8080",
      grpcService: "localhost:50051"
    }
  });
});

// Serve the main application
app.get("/", (req, res) => {
  try {
    const htmlPath = path.join(__dirname, 'index.html');
    const html = fs.readFileSync(htmlPath, 'utf-8');
    res.set({ "Content-Type": "text/html" }).send(html);
  } catch (error) {
    console.error("❌ Error serving index.html:", error);
    res.status(500).send("Error loading application");
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error("❌ Express error:", error);
  res.status(500).json({
    error: "Internal server error",
    message: error.message
  });
});

// Graceful shutdown handling
process.on('SIGINT', () => {
  console.log('\n🛑 Received SIGINT. Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Received SIGTERM. Shutting down gracefully...');
  process.exit(0);
});

app.listen(port, () => {
  console.log(`🚀 Real-time Lip Sync Console server running on http://localhost:${port}`);
  console.log(`🔑 OpenAI API Key: ${apiKey ? '✅ Configured' : '❌ Missing'}`);
  console.log(`📡 Frame Generator: ws://localhost:8080`);
  console.log(`🎬 gRPC Service: localhost:50051`);
  console.log(`\n💡 Open http://localhost:${port} to start using the console`);
});

export default app;
