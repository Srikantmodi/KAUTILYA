# Dataset Notes — IO-VNBD & Team Drive Logs

SIH PS-26168 Intelligent Dead Reckoning
Owner: Member 6 (On-Device ML Serving, Edge Engine & Benchmark Harness)

Derived from: Master PRD §6.11, api-contracts.md §12

---

## Critical CSV Parsing Gotchas

### 1. Column Name Whitespace

CSV column names in the IO-VNBD dataset carry **inconsistent leading/trailing
whitespace**.  Always `.strip()` column names immediately after reading:

```python
df = pd.read_csv(path, ...)
df.columns = df.columns.str.strip()
```

Failure to do this causes silent `KeyError`s or, worse, silently reading the
wrong column when a partial match is attempted.

### 2. Character Encoding — S- (Phone) Files

S- (smartphone) files are encoded as **`latin-1`** (ISO 8859-1), **not** UTF-8.
Always try UTF-8 first, then fall back:

```python
def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    df.columns = df.columns.str.strip()
    return df
```

V- (vehicle reference) files are typically UTF-8 and do not require fallback.

### 3. Timestamp Units — Seconds vs. Milliseconds

| File type | Time column unit | Conversion needed |
|-----------|-----------------|-------------------|
| V- (vehicle) | **seconds** (float, e.g. `1637245123.456`) | Multiply by 1000 to get epoch ms |
| S- (phone) | **milliseconds** (int, e.g. `1637245123456`) | Already in epoch ms — no conversion |

**This is a costly, easy-to-miss unit mismatch.**  Any time-alignment logic
(merging phone + vehicle data, computing fix age, etc.) will produce garbage
if the units are not normalized first.

```python
# Normalize to epoch milliseconds
if is_vehicle_file:
    df['timestamp_ms'] = (df['time'] * 1000).astype(int)
else:
    df['timestamp_ms'] = df['time'].astype(int)
```

### 4. GPS Outage Index File

The IO-VNBD dataset paper references a GPS-outage index file whose exact
location within the dataset was **never definitively confirmed** by the team.

- **If located**: wire into `test-harness/real_outage_evaluator.py` for the
  strongest possible benchmark evidence (real, labeled outage windows from the
  dataset authors themselves).
- **If not located**: `test-harness/simulator/gnss_outage_simulator.py` is the
  documented fallback.  Any drift numbers produced using simulated outages
  **must be clearly labeled as "simulated"** in all reports and outputs — never
  presented as if they were measured against real outage events.

The file is expected to be at:
`ml-pipeline/data/raw/iovnbd/synchronised/categorised/gps_outages.csv`

---

## IO-VNBD Dataset Structure

```
ml-pipeline/data/raw/iovnbd/
├── synchronised/categorised/
│   ├── V-*.csv          # Vehicle reference (CAN-bus speed, GPS, etc.)
│   ├── S-*.csv          # Smartphone sensor data (accel, gyro, GPS)
│   └── gps_outages.csv  # GPS outage index (location unconfirmed)
└── unsynchronised/categorised/
    └── S-*.csv          # Additional phone-only sessions
```

- 94 total sessions, 8 drivers
- Cross-verified against the dataset's published paper
- All sessions from European roads (Belgium) — Indian road conditions will
  differ; team-collected drives in `ml-pipeline/data/raw/team_drives/` are
  needed for final validation

---

## Column Reference (IO-VNBD Synchronised)

### V- (Vehicle) Files
Typical columns (after `.strip()`):
`time`, `latitude`, `longitude`, `speed`, `heading`, `ax`, `ay`, `az`,
`gx`, `gy`, `gz`, `obd_speed`, ...

### S- (Phone) Files
Typical columns (after `.strip()`):
`time`, `latitude`, `longitude`, `speed`, `bearing`, `accuracy`,
`ax`, `ay`, `az`, `gx`, `gy`, `gz`, `mx`, `my`, `mz`, ...

---

## Team Drive Logs

Team-collected drive logs stored in `ml-pipeline/data/raw/team_drives/`
should follow a standardized CSV format:

```csv
timestamp_ms,ax,ay,az,gx,gy,gz,lat,lon,speed_mps,accuracy_m,satellites
```

- All timestamps in **epoch milliseconds**
- Accelerometer in **m/s²** (includes gravity)
- Gyroscope in **rad/s**
- GPS fields may be empty/NaN during GNSS outage periods
