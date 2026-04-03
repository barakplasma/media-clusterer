# Photo Organizer 📸 ✨

A high-performance, **zero-server**, local AI photo organizer that runs entirely in your browser using **WebGPU** and **Transformers.js**.

---

## 🚀 Key Features

- **Semantic Search**: Find photos by description (e.g., "sunset at the beach", "golden retriever puppy") using multimodal AI.
- **Privacy First**: No photos are uploaded. No server involved. Everything happens locally on your device.
- **WebGPU Powered**: Blazing fast embeddings and projections using your GPU.
- **Infinite Canvas**: Explore thousands of photos in a fluid, 2D map powered by UMAP dimensionality reduction.
- **Cluster Visualization**: Visually similar photos are automatically grouped and colored using k-means clustering.
- **Smart Caching**: Embeddings are cached in IndexedDB for instant re-opening of previously processed folders.
- **Safari/iOS Compatible**: Robust fallbacks for mobile browsers without File System Access API or WebGPU.

## 🛠️ Technology Stack

- **Language**: TypeScript (Strict, zero `any`)
- **AI Engine**: [Transformers.js](https://huggingface.co/docs/transformers.js) (Nomic-Embed-Vision-v1.5 & Nomic-Embed-Text-v1.5)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Projections**: [UMAP-JS](https://github.com/PAIR-code/umap-js)
- **Storage**: IndexedDB & LocalStorage
- **Deployment**: Kubernetes + Caddy (Static Hosting)

## 📦 Project Structure

```text
index.html       — Main application entry & UI
src/
  app.ts         — Core application lifecycle & event handling
  db.ts          — IndexedDB storage management
  embeddings.ts  — AI model normalization & vector extraction
  similarity.ts  — Vector similarity search algorithms
  types.ts       — Shared TypeScript interfaces & types
public/
  umap-js.min.js — UMAP dimensionality reduction library
```

## 👨‍💻 Development

### Setup

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

### Production Build

```bash
# Build and type-check
npm run build

# Run tests (one-off)
npm test
```

## 📜 How it Works

1.  **Loading**: The app loads two specialized models: one for vision (images) and one for text (search queries).
2.  **Embedding**: Images are processed in small batches to generate 768-dimensional semantic vectors.
3.  **UMAP**: These high-dimensional vectors are projected into a 2D space while preserving local relationships.
4.  **Clustering**: k-means identifies clusters of similar images for color-coding.
5.  **Search**: Text queries are embedded into the same vector space, allowing for direct comparison with images via cosine similarity.

---

*Built with ❤️ for privacy and high-performance web computing.*
