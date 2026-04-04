/**
 * IndexedDB cache for embeddings
 */

import type { CacheKey } from './types';

let db: IDBDatabase | null = null;

const DB_NAME = 'photo-organizer-v1';
const STORE_NAME = 'embeddings';
const DB_VERSION = 1;

/**
 * Open IndexedDB database
 */
export async function openDB(): Promise<IDBDatabase> {
  if (db) return db;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = () => {
      db = request.result;
      resolve(db);
    };

    request.onerror = () => {
      reject(request.error);
    };
  });
}

/**
 * Batch get embeddings from cache
 */
export async function cacheGetBatch(keys: CacheKey[]): Promise<(Float32Array | null)[]> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const results: (Float32Array | null)[] = new Array(keys.length).fill(null);
    let count = 0;

    keys.forEach((key, i) => {
      const request = store.get(key);
      request.onsuccess = () => {
        results[i] = request.result ?? null;
        count++;
        if (count === keys.length) resolve(results);
      };
      request.onerror = () => {
        reject(request.error);
      };
    });

    if (keys.length === 0) resolve([]);
  });
}

/**
 * Put embedding in cache
 */
export async function cachePut(key: CacheKey, value: Float32Array): Promise<void> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    store.put(value, key);

    transaction.oncomplete = () => {
      resolve();
    };

    transaction.onerror = () => {
      reject(transaction.error);
    };
  });
}

/**
 * Batch put embeddings (20 per transaction for performance)
 */
export async function cachePutBatch(entries: [CacheKey, Float32Array][]): Promise<void> {
  const database = await openDB();

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    for (const [key, value] of entries) {
      store.put(value, key);
    }

    transaction.oncomplete = () => {
      resolve();
    };

    transaction.onerror = () => {
      reject(transaction.error);
    };
  });
}
