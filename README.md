# Media Clusterer 📸 🎥 ✨

A high-performance, **zero-server**, local AI media management tool that runs entirely in your browser using **WebGPU**, **Transformers.js**, and **DruidJS**.

---

<img width="1891" height="1035" alt="demo of the media clusterer based on https://www.kaggle.com/datasets/heavensky/image-dataset-for-unsupervised-clustering?resource=download" src="https://github.com/user-attachments/assets/db8520e5-b937-4f1b-b28e-df0c52821c0a" />

---

## 🚀 Key Features

- **Semantic Search**: Find photos and videos by description (e.g., "surfing at sunset", "golden retriever puppy") using multimodal AI.
- **Video Support**: Supports **MP4** and **WEBM** files. Automatically extracts thumbnails for embedding and visualization.
- **Privacy First**: No media is uploaded. No server involved. Everything happens locally on your device.
- **WebGPU Powered**: Blazing fast embeddings and projections using your GPU.
- **Infinite Canvas**: Explore thousands of media files in a fluid, 2D map.
- **Advanced Projections**: Switch between **UMAP**, **t-SNE**, **PCA**, **Isomap**, **LLE**, **MDS**, **Sammon**, and **TriMap** in real-time.
- **Cluster Visualization**: Visually similar media are automatically grouped and colored using k-means clustering.
- **Smart Caching**: Embeddings are cached in IndexedDB for instant re-opening of previously processed folders.

## 🛠️ Technology Stack

- **Language**: TypeScript (Strict)
- **AI Engine**: [Transformers.js](https://huggingface.co/docs/transformers.js) (v4+)
- **Build Tool**: [Vite](https://vitejs.dev/) (v8+ with Rolldown)
- **Projections**: [DruidJS](https://github.com/saehm/DruidJS)
- **Deployment**: [Cloudflare Pages](https://pages.cloudflare.com/)

## 📦 Project Structure

```text
index.html       — Main application entry & UI
src/
  app.ts         — Core application logic & event handling
  db.ts          — IndexedDB storage management
  embeddings.ts  — AI model normalization & vector extraction
  similarity.ts  — Vector similarity search algorithms
  types.ts       — Shared TypeScript interfaces
```

## 👨‍💻 Development

### Setup

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

### Build & Deploy

```bash
# Production Build
npm run build

# Deploy to Cloudflare Pages (Manual)
npm run deploy
```

---

*Built with ❤️ for privacy and high-performance web computing.*
