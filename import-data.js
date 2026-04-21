
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { parse } from 'csv-parse/sync';
import { readFileSync } from 'fs';



const firebaseConfig = {
  apiKey: "AIzaSyDwsvaPmWwviTuJv2DPMvhpJq-nhtwhkiQ",
  authDomain: "hotels-analytics.firebaseapp.com",
  projectId: "hotels-analytics",
  storageBucket: "hotels-analytics.firebasestorage.app",
  messagingSenderId: "402846881119",
  appId: "1:402846881119:web:2a01c46ee2536d467112ea",
  measurementId: "G-CQ8SLTRGF2"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

async function importCSV(filePath, collectionName, transform) {
  const raw = readFileSync(filePath, 'utf8');
  const rows = parse(raw, { columns: true, skip_empty_lines: true });
  console.log(`Importing ${rows.length} rows into "${collectionName}"...`);
  for (const row of rows) {
    await addDoc(collection(db, collectionName), {
      ...transform(row),
      imported_at: serverTimestamp()
    });
  }
  console.log('Done.');
}

// Coerce numeric strings to numbers
function coerce(row) {
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    const n = Number(v);
    out[k] = v === '' ? null : isNaN(n) ? v : n;
  }
  return out;
}

await importCSV('Nexus_Booked_Events_2026_2027.csv',   'booked_events',   coerce);
await importCSV('Nexus_Incoming_RFPs_Ranked.csv',      'incoming_rfps',   coerce);
await importCSV('Nexus_Transient_Demand_2026_2027.csv','transient_demand', coerce);
