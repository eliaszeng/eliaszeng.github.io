const DB_NAME = 'TradingJournal';
const DB_VERSION = 1;
const STORE_NAME = 'entries';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('createdAt', 'createdAt', { unique: false });
        store.createIndex('stockCode', 'stockCode', { unique: false });
        store.createIndex('type', 'type', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(mode, fn) {
  return openDB().then(db => new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, mode);
    const store = transaction.objectStore(STORE_NAME);
    const result = fn(store);
    if (result && result.onsuccess !== undefined) {
      result.onsuccess = () => resolve(result.result);
      result.onerror = () => reject(result.error);
    } else {
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
    }
  }));
}

export function addEntry(entry) {
  const data = {
    ...entry,
    id: crypto.randomUUID(),
    createdAt: entry.createdAt || new Date().toISOString(),
  };
  return tx('readwrite', store => store.put(data)).then(() => data);
}

export function updateEntry(entry) {
  return tx('readwrite', store => store.put(entry)).then(() => entry);
}

export function deleteEntry(id) {
  return tx('readwrite', store => store.delete(id));
}

export function getEntry(id) {
  return tx('readonly', store => store.get(id));
}

export function getAllEntries() {
  return tx('readonly', store => {
    return new Promise((resolve) => {
      const results = [];
      const index = store.index('createdAt');
      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };
    });
  });
}

export async function exportJSON() {
  const entries = await getAllEntries();
  const blob = new Blob([JSON.stringify(entries, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trading-journal-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function importJSON(file) {
  const text = await file.text();
  const entries = JSON.parse(text);
  if (!Array.isArray(entries)) throw new Error('Invalid format');
  for (const entry of entries) {
    await tx('readwrite', store => store.put(entry));
  }
  return entries.length;
}

export function clearAll() {
  return tx('readwrite', store => store.clear());
}

export function getStockLinks(code) {
  if (!code) return {};
  const c = code.replace(/\D/g, '');
  const ths = `https://stockpage.10jqka.com.cn/${c}/`;
  const prefix = c.startsWith('6') ? 'sh'
    : c.startsWith('4') || c.startsWith('8') ? 'bj' : 'sz';
  const east = `https://quote.eastmoney.com/${prefix}${c}.html`;
  return { ths, east };
}
