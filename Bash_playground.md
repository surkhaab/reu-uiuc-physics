#!/bin/bash

# ─── 1. CREATE FILES ───────────────────────────────────────
for i in {1..5}; do
  echo "I am file number $i" > "file_$i.txt"
  echo "Created file_$i.txt"
done

# ─── 2. PRINT CONTENTS OF EACH FILE ───────────────────────
echo ""
echo "=== File Contents ==="
for f in file_*.txt; do
  echo "$f: $(cat $f)"
done

# ─── 3. RENAME .txt TO .dat ────────────────────────────────
echo ""
echo "=== Renaming .txt to .dat ==="
for f in file_*.txt; do
  mv "$f" "${f%.txt}.dat"
  echo "Renamed $f → ${f%.txt}.dat"
done

# ─── 4. BATCH FIND & REPLACE INSIDE FILES ─────────────────
echo ""
echo "=== Replacing 'file' with 'document' in all .dat files ==="
sed -i 's/file/document/g' *.dat
grep "" *.dat   # print all file contents to confirm

# ─── 5. BACKUP ALL .dat FILES INTO A ZIP ──────────────────
echo ""
echo "=== Creating backup zip ==="
zip -r "backup_$(date +%F).zip" *.dat
echo "Backup created: backup_$(date +%F).zip"

# ─── 6. CLEANUP — delete the .dat files ───────────────────
echo ""
echo "=== Cleaning up .dat files ==="
for f in *.dat; do
  rm "$f"
  echo "Deleted $f"
done

echo ""
echo "Done! Only the zip remains."
