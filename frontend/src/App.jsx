import { useState, useEffect, useCallback } from "react";
import {
  LayoutDashboard, ShieldAlert, Upload, Activity, Brain, Settings,
  TrendingUp, AlertTriangle, CheckCircle, Server, Database, BarChart3,
  Download, FileSearch, Trash2, Eye, Info, Smartphone,
  UserRound, CreditCard, Clock3, Coins, MapPin, ShieldCheck,
  ScanSearch, FileSpreadsheet, History, ChartNoAxesCombined, BrainCircuit
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, BarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, Legend
} from "recharts";

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const modelMetrics = [
  { name: "Precision", value: 99.83 },
  { name: "Recall", value: 99.92 },
  { name: "F1 Score", value: 99.87 },
];

const COLORS = ["#dc2626", "#16a34a", "#2563eb", "#f59e0b", "#8b5cf6", "#ec4899"];

/* ------------------------------------------------------------------ */
/*  CSV PARSER                                                         */
/* ------------------------------------------------------------------ */
function parseCsv(text) {
  const rows = []; let row = []; let value = ""; let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (inQuotes && text[i + 1] === '"') { value += '"'; i++; }
      else { inQuotes = !inQuotes; }
    } else if (c === "," && !inQuotes) {
      row.push(value); value = "";
    } else if ((c === "\n" || c === "\r") && !inQuotes) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(value);
      if (row.some(x => x !== "")) rows.push(row);
      row = []; value = "";
    } else {
      value += c;
    }
  }
  row.push(value);
  if (row.some(x => x !== "")) rows.push(row);
  const [headers = [], ...dataRows] = rows;
  return dataRows.map(cells => Object.fromEntries(headers.map((h, i) => [h, cells[i] ?? ""])));
}

function toNum(v) {
  if (v === "" || v === undefined || v === null) return 0;
  const n = Number(v);
  return isNaN(n) ? 0 : n;
}

/* ================================================================== */
/*  APP ROOT                                                           */
/* ================================================================== */
function App() {
  const [activePage, setActivePage] = useState("Dashboard");

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Prediction", icon: ScanSearch },
    { name: "CSV Analysis", icon: FileSpreadsheet },
    { name: "History", icon: History },
    { name: "Analytics", icon: ChartNoAxesCombined },
    { name: "Monitoring", icon: Activity },
    { name: "Model", icon: BrainCircuit },
    { name: "Settings", icon: Settings },
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandIcon"><ShieldAlert size={25} /></div>
          <div>
            <div className="brandName">FraudGuard</div>
            <div className="brandSub">MLOps Platform</div>
          </div>
        </div>
        <div className="menuTitle">MAIN MENU</div>
        <nav>
          {menuItems.map(item => {
            const Icon = item.icon;
            return (
              <button
                key={item.name}
                className={`navButton ${activePage === item.name ? "active" : ""}`}
                onClick={() => setActivePage(item.name)}
              >
                <Icon size={19} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebarBottom">
          <div className="systemBox">
            <div className="systemHeader"><Server size={17} /><span>System Status</span></div>
            <div className="systemStatus"><span className="onlineDot"></span>All systems operational</div>
          </div>
          <div className="version">FraudGuard v2.0</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{activePage}</h1>
            <p>
              {activePage === "Dashboard"
                ? "Overview of your fraud detection system"
                : `Manage your ${activePage.toLowerCase()} module`}
            </p>
          </div>
          <div className="apiStatus"><span className="onlineDot"></span><span>API Online</span></div>
        </header>
        <div className="content">
          {activePage === "Dashboard" && <Dashboard />}
          {activePage === "Prediction" && <Prediction />}
          {activePage === "CSV Analysis" && <CSVAnalysis />}
          {activePage === "History" && <HistoryPage />}
          {activePage === "Analytics" && <AnalyticsPage />}
          {activePage === "Monitoring" && <Monitoring />}
          {activePage === "Model" && <Model />}
          {activePage === "Settings" && <SettingsPage />}
        </div>
      </main>
    </div>
  );
}

function FieldGroup({ label, children, span, error }) {
  return (
    <div className={`formGroup ${error ? "fieldError" : ""}`} style={span ? { gridColumn: span } : undefined}>
      <label>{label}</label>
      {children}
      {error && <span className="fieldErrorMsg">⚠ {error}</span>}
    </div>
  );
}

/* ================================================================== */
/*  PREDICTION COMPONENT                                               */
/* ================================================================== */
function Prediction() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const defaultDate = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  const defaultTime = `${pad(now.getHours())}:${pad(now.getMinutes())}`;

  const [form, setForm] = useState({
    type: "TRANSFER",
    currency: "INR",
    previous_fraud_count: "0",
    date: defaultDate,
    time: defaultTime,
    amount: "1000",
    oldbalanceOrg: "5000",
    newbalanceOrig: "4000",
    oldbalanceDest: "1000",
    newbalanceDest: "2000",
    device_type: "Mobile",
    new_device: "No",
    new_location: "No",
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // General change handler for dropdowns/date/time
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  // Dedicated numeric handler to fix the "08000" bug:
  // Allows empty string on Backspace/Delete.
  // Replaces leading zero like "08000" -> "8000".
  const handleNumericChange = (e) => {
    const { name, value } = e.target;
    if (value === "") {
      setForm(prev => ({ ...prev, [name]: "" }));
      return;
    }
    // Only allow numbers, optional leading minus and optional decimal
    if (!/^-?\d*\.?\d*$/.test(value)) return;

    let clean = value;
    // Strip leading zeros before digits: "08000" -> "8000", while keeping "0" or "0.5"
    if (/^0\d+/.test(clean)) {
      clean = clean.replace(/^0+/, "");
      if (clean === "") clean = "0";
    }
    setForm(prev => ({ ...prev, [name]: clean }));
  };

  // Client-side validation: only check that required inputs are provided.
  // Logical error conditions (negative amount, zero amount, balance mismatches)
  // are evaluated by the backend rule-based fraud validation engine.
  const validate = useCallback(() => {
    const e = {};
    if (form.amount === "") {
      e.amount = "Amount is required.";
    }
    if (form.oldbalanceOrg === "") {
      e.oldbalanceOrg = "Old balance (origin) is required.";
    }
    if (!form.date) e.date = "Date is required.";
    if (!form.time) e.time = "Time is required.";

    setErrors(e);
    return Object.keys(e).length === 0;
  }, [form]);

  useEffect(() => { validate(); }, [validate]);

  const hasErrors = Object.keys(errors).length > 0;

  const handlePredict = async (ev) => {
    ev.preventDefault();
    if (!validate()) return;
    setLoading(true);
    setResult(null);

    const timeStr = `${form.date} ${form.time}:00`;
    const body = {
      type: form.type,
      amount: toNum(form.amount),
      oldbalanceOrg: toNum(form.oldbalanceOrg),
      newbalanceOrig: toNum(form.newbalanceOrig),
      oldbalanceDest: toNum(form.oldbalanceDest),
      newbalanceDest: toNum(form.newbalanceDest),
      Time: timeStr,
      currency: form.currency,
      device_type: form.device_type,
      new_device: form.device_type === "Mobile" ? form.new_device : null,
      new_location: form.new_location,
      previous_fraud_count: toNum(form.previous_fraud_count),
    };

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error("Prediction request failed");
      const data = await res.json();
      if (data.valid === false && data.errors) {
        const serverErrs = {};
        data.errors.forEach(err => { serverErrs[err.field] = err.message; });
        setErrors(serverErrs);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch {
      setResult({ error: "Could not connect to FastAPI. Make sure the backend is running on port 8000." });
    } finally {
      setLoading(false);
    }
  };

  const currencies = [
    ["INR", "Indian Rupee"], ["USD", "US Dollar"], ["EUR", "Euro"], ["GBP", "British Pound"],
    ["JPY", "Japanese Yen"], ["AUD", "Australian Dollar"], ["CAD", "Canadian Dollar"],
    ["SGD", "Singapore Dollar"], ["AED", "UAE Dirham"], ["CHF", "Swiss Franc"],
  ];

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>Transaction Prediction</h2>
          <p>Enter transaction details and let the XGBoost production model determine whether the transaction is fraudulent.</p>
        </div>
        <div className="modelBadge light">
          <Brain size={18} />
          <div><strong>XGBoost</strong><span>Production Model</span></div>
        </div>
      </div>

      <div className="predictionGrid">
        <div className="panel">
          <form onSubmit={handlePredict}>
            {/* 1. BASIC DETAILS */}
            <div className="sectionTitle"><UserRound size={16} /> Basic Details</div>
            <div className="formGrid">
              <FieldGroup label="Transaction Type" error={errors.type}>
                <select name="type" value={form.type} onChange={handleChange}>
                  <option value="CASH_IN">CASH_IN</option>
                  <option value="CASH_OUT">CASH_OUT</option>
                  <option value="DEBIT">DEBIT</option>
                  <option value="PAYMENT">PAYMENT</option>
                  <option value="TRANSFER">TRANSFER</option>
                </select>
              </FieldGroup>
              <FieldGroup label={<span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><Coins size={14} /> Currency</span>} error={errors.currency}>
                <select name="currency" value={form.currency} onChange={handleChange}>
                  {currencies.map(([code, name]) => <option key={code} value={code}>{code} - {name}</option>)}
                </select>
              </FieldGroup>
              <FieldGroup label={<span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><ShieldAlert size={14} /> Previous Fraud Count</span>} error={errors.previous_fraud_count}>
                <input
                  type="text"
                  inputMode="numeric"
                  name="previous_fraud_count"
                  value={form.previous_fraud_count}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
              <div className="formGroup" style={{ gridColumn: "1 / -1" }}>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><Clock3 size={14} /> Transaction Time</label>
                <div style={{ display: "flex", gap: "10px" }}>
                  <div className={`formGroup ${errors.date ? "fieldError" : ""}`} style={{ flex: 1, gap: 0 }}>
                    <input type="date" name="date" value={form.date} onChange={handleChange} />
                    {errors.date && <span className="fieldErrorMsg">⚠ {errors.date}</span>}
                  </div>
                  <div className={`formGroup ${errors.time ? "fieldError" : ""}`} style={{ flex: 1, gap: 0 }}>
                    <input type="time" name="time" value={form.time} onChange={handleChange} />
                    {errors.time && <span className="fieldErrorMsg">⚠ {errors.time}</span>}
                  </div>
                </div>
              </div>
            </div>

            {/* 2. TRANSACTION DETAILS */}
            <div className="sectionTitle" style={{ marginTop: 28 }}><CreditCard size={16} /> Transaction Details</div>
            <div className="formGrid">
              <FieldGroup label="Amount" error={errors.amount}>
                <input
                  type="text"
                  inputMode="decimal"
                  name="amount"
                  value={form.amount}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
              <FieldGroup label="Old Balance (Origin)" error={errors.oldbalanceOrg}>
                <input
                  type="text"
                  inputMode="decimal"
                  name="oldbalanceOrg"
                  value={form.oldbalanceOrg}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
              <FieldGroup label="New Balance (Origin)" error={errors.newbalanceOrig}>
                <input
                  type="text"
                  inputMode="decimal"
                  name="newbalanceOrig"
                  value={form.newbalanceOrig}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
              <FieldGroup label="Old Balance (Destination)" error={errors.oldbalanceDest}>
                <input
                  type="text"
                  inputMode="decimal"
                  name="oldbalanceDest"
                  value={form.oldbalanceDest}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
              <FieldGroup label="New Balance (Destination)" error={errors.newbalanceDest}>
                <input
                  type="text"
                  inputMode="decimal"
                  name="newbalanceDest"
                  value={form.newbalanceDest}
                  onChange={handleNumericChange}
                  placeholder="0"
                />
              </FieldGroup>
            </div>

            {/* 3. ADDITIONAL INFORMATION */}
            <div className="sectionTitle" style={{ marginTop: 28 }}><Info size={16} /> Additional Information</div>
            <div className="formGrid">
              <FieldGroup label={<span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><Smartphone size={14} /> Device Type</span>} error={errors.device_type}>
                <select name="device_type" value={form.device_type} onChange={handleChange}>
                  <option value="Mobile">Mobile</option>
                  <option value="Web">Web</option>
                  <option value="ATM">ATM</option>
                  <option value="POS">POS</option>
                  <option value="Other">Other</option>
                </select>
              </FieldGroup>

              {form.device_type === "Mobile" && (
                <FieldGroup label="New Device?" error={errors.new_device}>
                  <div className="segmentedControl">
                    <button
                      type="button"
                      className={form.new_device === "Yes" ? "active" : ""}
                      onClick={() => setForm(p => ({ ...p, new_device: "Yes" }))}
                    >
                      Yes
                    </button>
                    <button
                      type="button"
                      className={form.new_device === "No" ? "active" : ""}
                      onClick={() => setForm(p => ({ ...p, new_device: "No" }))}
                    >
                      No
                    </button>
                  </div>
                </FieldGroup>
              )}

              <FieldGroup label={<span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><MapPin size={14} /> New Location?</span>} error={errors.new_location}>
                <div className="segmentedControl">
                  <button
                    type="button"
                    className={form.new_location === "Yes" ? "active" : ""}
                    onClick={() => setForm(p => ({ ...p, new_location: "Yes" }))}
                  >
                    Yes
                  </button>
                  <button
                    type="button"
                    className={form.new_location === "No" ? "active" : ""}
                    onClick={() => setForm(p => ({ ...p, new_location: "No" }))}
                  >
                    No
                  </button>
                </div>
              </FieldGroup>
            </div>

            <button className="predictButton" type="submit" disabled={loading || hasErrors}>
              {loading ? "Analyzing Transaction..." : "Predict Transaction"}
            </button>
          </form>
        </div>

        {/* PREDICTION RESULT CARD */}
        <div className="predictionResult">
          {!result && (
            <div className="emptyResult">
              <div className="emptyIcon"><ShieldCheck size={35} /></div>
              <h3>Prediction Result</h3>
              <p>Submit a transaction to see the fraud detection result.</p>
            </div>
          )}
          {result && !result.error && result.prediction !== undefined && (
            <div className={`resultCard ${result.prediction === 1 ? "fraudResult" : "legitimateResult"}`}>
              {result.prediction === 1 ? <ShieldAlert size={42} /> : <ShieldCheck size={42} />}
              <div className="resultLabel">{result.result}</div>
              <div className="probability">{(result.fraud_probability * 100).toFixed(4)}%</div>
              <p>Fraud Probability</p>
              <div className="resultDetails">
                <div><span>Prediction</span><strong>{result.prediction}</strong></div>
                <div><span>Threshold</span><strong>{result.threshold}</strong></div>
              </div>
              {result.detection_method && (
                <div className="resultDetails" style={{ marginTop: 8 }}>
                  <div>
                    <span>Detection</span>
                    <strong style={{ fontSize: 12 }}>
                      {result.detection_method === "rule-based" ? "Rule-based fraud validation" : "XGBoost ML Model"}
                    </strong>
                  </div>
                </div>
              )}
              {result.rule_reason && (
                <div style={{ marginTop: 10, padding: "8px 12px", borderRadius: 8, background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.15)", fontSize: 12, color: "#dc2626", fontWeight: 600 }}>
                  Reason: {result.rule_reason}
                </div>
              )}
              {result.transaction_id && (
                <div style={{ marginTop: 14, fontSize: 12, opacity: 0.75, fontWeight: 600 }}>
                  ID: {result.transaction_id}
                </div>
              )}
            </div>
          )}
          {result?.error && (
            <div className="errorResult">
              <AlertTriangle size={30} />
              <h3>Connection Error</h3>
              <p>{result.error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  CSV ANALYSIS COMPONENT                                             */
/* ================================================================== */
function CSVAnalysis() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null);
    setResults([]);
    setSummary(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) { setError("Please select a CSV file first."); return; }
    setLoading(true); setError(""); setResults([]); setSummary(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API}/predict_csv`, {
        method: "POST",
        headers: { "Accept": "application/json" },
        body: formData
      });
      if (!response.ok) throw new Error("CSV prediction failed");

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        if (data.predictions) {
          setResults(data.predictions);
          setSummary({ total: data.total, fraud: data.fraud, legitimate: data.legitimate });
        } else if (Array.isArray(data)) {
          setResults(data);
        }
      } else {
        const predictionResults = parseCsv(await response.text());
        setResults(predictionResults);
        const fraudCount = predictionResults.filter(r => Number(r.prediction) === 1 || r.result === "FRAUD").length;
        setSummary({ total: predictionResults.length, fraud: fraudCount, legitimate: predictionResults.length - fraudCount });
      }
    } catch (err) {
      setError(err.message || "Could not connect to the CSV prediction API.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!results.length) return;
    const allKeys = Object.keys(results[0]);
    const csvContent = [
      allKeys.join(","),
      ...results.map(row => allKeys.map(k => `"${String(row[k] ?? "").replace(/"/g, '""')}"`).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "fraud_predictions.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>CSV Transaction Analysis</h2>
          <p>Upload a CSV file to analyze multiple transactions using the production XGBoost model.</p>
        </div>
        <div className="modelBadge light">
          <FileSpreadsheet size={18} />
          <div><strong>Batch Prediction</strong><span>CSV Analysis</span></div>
        </div>
      </div>

      <div className="panel uploadPanel">
        <div className="uploadArea">
          <div className="uploadIcon"><Upload size={32} /></div>
          <h3>Upload Transaction CSV</h3>
          <p>Select a CSV file containing transaction details.</p>
          <label className="fileButton">
            Choose CSV File
            <input type="file" accept=".csv" onChange={handleFileChange} />
          </label>
          {file && (
            <div className="selectedFile">
              <FileSearch size={17} />
              <span>{file.name}</span>
            </div>
          )}
          <button className="uploadButton" onClick={handleUpload} disabled={!file || loading}>
            {loading ? "Analyzing Transactions..." : "Analyze CSV"}
          </button>
          {error && (
            <div className="csvError">
              <AlertTriangle size={17} />
              {error}
            </div>
          )}
        </div>
      </div>

      {summary && (
        <div className="csvSummary">
          <CSVStat title="Total Transactions" value={summary.total} type="blue" />
          <CSVStat title="Fraud Detected" value={summary.fraud} type="red" />
          <CSVStat title="Legitimate" value={summary.legitimate} type="green" />
          <CSVStat
            title="Fraud Rate"
            value={summary.total ? `${((summary.fraud / summary.total) * 100).toFixed(2)}%` : "0%"}
            type="orange"
          />
        </div>
      )}

      {results.length > 0 && (
        <div className="panel resultsPanel">
          <div className="panelHeader">
            <div>
              <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}><FileSpreadsheet size={18} /> Prediction Results</h3>
              <p>Results returned by the FastAPI prediction service</p>
            </div>
            <button className="downloadButton" onClick={handleDownload}>
              <Download size={17} />Download CSV
            </button>
          </div>
          <div className="tableWrapper">
            <table className="resultsTable">
              <thead>
                <tr>
                  <th>Step</th><th>Type</th><th>Amount</th><th>Old Origin</th><th>New Origin</th>
                  <th>Old Dest</th><th>New Dest</th><th>Prediction</th><th>Probability</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row, i) => {
                  const isFraud = Number(row.prediction) === 1 || row.result === "FRAUD";
                  return (
                    <tr key={i}>
                      <td>{row.step ?? "-"}</td>
                      <td>{row.type ?? "-"}</td>
                      <td>{row.amount !== undefined ? Number(row.amount).toLocaleString() : "-"}</td>
                      <td>{row.oldbalanceOrg !== undefined ? Number(row.oldbalanceOrg).toLocaleString() : "-"}</td>
                      <td>{row.newbalanceOrig !== undefined ? Number(row.newbalanceOrig).toLocaleString() : "-"}</td>
                      <td>{row.oldbalanceDest !== undefined ? Number(row.oldbalanceDest).toLocaleString() : "-"}</td>
                      <td>{row.newbalanceDest !== undefined ? Number(row.newbalanceDest).toLocaleString() : "-"}</td>
                      <td>
                        <span className={`predictionBadge ${isFraud ? "fraudBadge" : "legitBadge"}`}>
                          {isFraud ? "FRAUD" : "LEGITIMATE"}
                        </span>
                      </td>
                      <td>{row.fraud_probability !== undefined ? `${(Number(row.fraud_probability) * 100).toFixed(4)}%` : "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function CSVStat({ title, value, type }) {
  return <div className={`csvStat ${type}`}><span>{title}</span><strong>{value}</strong></div>;
}

/* ================================================================== */
/*  HISTORY PAGE — Manual Predictions + CSV Analyses Tabs              */
/* ================================================================== */
function HistoryPage() {
  const [tab, setTab] = useState("manual");
  const [manualHistory, setManualHistory] = useState([]);
  const [csvHistory, setCsvHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [resultFilter, setResultFilter] = useState("All");
  const [typeFilter, setTypeFilter] = useState("All");
  const [detailItem, setDetailItem] = useState(null);
  const [csvDetail, setCsvDetail] = useState(null);
  const [csvDetailLoading, setCsvDetailLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [mRes, cRes] = await Promise.all([
        fetch(`${API}/prediction-history`),
        fetch(`${API}/csv-history`)
      ]);
      if (mRes.ok) setManualHistory(await mRes.json());
      if (cRes.ok) setCsvHistory(await cRes.json());
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const clearManual = async () => {
    if (!confirm("Clear all manual prediction history?")) return;
    await fetch(`${API}/prediction-history`, { method: "DELETE" });
    setManualHistory([]);
  };

  const clearCsv = async () => {
    if (!confirm("Clear all CSV analysis history?")) return;
    await fetch(`${API}/csv-history`, { method: "DELETE" });
    setCsvHistory([]);
  };

  const exportManual = () => {
    window.open(`${API}/prediction-history/export`, "_blank");
  };

  const viewCsvDetail = async (id) => {
    setCsvDetailLoading(true);
    try {
      const res = await fetch(`${API}/csv-history/${id}`);
      if (res.ok) setCsvDetail(await res.json());
    } catch (e) {
      console.error(e);
    }
    setCsvDetailLoading(false);
  };

  const exportCsv = (id) => {
    window.open(`${API}/csv-history/${id}/export`, "_blank");
  };

  const filtered = manualHistory.filter(r => {
    if (resultFilter !== "All" && r.result !== resultFilter) return false;
    if (typeFilter !== "All" && r.type !== typeFilter) return false;
    if (search) {
      const s = search.toLowerCase();
      if (
        !(r.transaction_id || "").toLowerCase().includes(s) &&
        !(r.type || "").toLowerCase().includes(s) &&
        !String(r.amount).includes(s)
      ) {
        return false;
      }
    }
    return true;
  });

  const formatTime = (t) => {
    if (!t) return "-";
    try { return new Date(t).toLocaleString(); } catch { return t; }
  };

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>Prediction History</h2>
          <p>View all predictions from manual transactions and CSV analyses.</p>
        </div>
        <div className="modelBadge light">
          <History size={18} />
          <div><strong>History</strong><span>All Records</span></div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        <button
          className={`tabButton ${tab === "manual" ? "active" : ""}`}
          onClick={() => { setTab("manual"); setCsvDetail(null); }}
        >
          Manual Predictions {manualHistory.length > 0 && <span className="tabBadge">{manualHistory.length}</span>}
        </button>
        <button
          className={`tabButton ${tab === "csv" ? "active" : ""}`}
          onClick={() => { setTab("csv"); setDetailItem(null); }}
        >
          CSV Analyses {csvHistory.length > 0 && <span className="tabBadge">{csvHistory.length}</span>}
        </button>
      </div>

      {loading ? (
        <div className="emptyState"><p>Loading history...</p></div>
      ) : (
        <>
          {/* MANUAL TAB */}
          {tab === "manual" && (
            <>
              {detailItem ? (
                <div className="panel" style={{ padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <h3 style={{ margin: 0, fontSize: 16 }}>Prediction Details</h3>
                    <button
                      className="closeButton"
                      style={{ width: "auto", marginTop: 0, padding: "6px 14px" }}
                      onClick={() => setDetailItem(null)}
                    >
                      ← Back
                    </button>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                    {[
                      ["Transaction ID", detailItem.transaction_id],
                      ["Time", formatTime(detailItem.Time)],
                      ["Type", detailItem.type],
                      ["Amount", Number(detailItem.amount).toLocaleString()],
                      ["Currency", detailItem.currency],
                      ["Device", detailItem.device_type],
                      ["New Device", detailItem.new_device ?? "-"],
                      ["New Location", detailItem.new_location ?? "-"],
                      ["Previous Fraud Count", detailItem.previous_fraud_count ?? "-"],
                      ["Old Balance (Origin)", Number(detailItem.oldbalanceOrg).toLocaleString()],
                      ["New Balance (Origin)", Number(detailItem.newbalanceOrig).toLocaleString()],
                      ["Old Balance (Dest)", Number(detailItem.oldbalanceDest).toLocaleString()],
                      ["New Balance (Dest)", Number(detailItem.newbalanceDest).toLocaleString()],
                      ["Result", detailItem.result],
                      ["Fraud Probability", `${(Number(detailItem.fraud_probability) * 100).toFixed(4)}%`],
                      ["Threshold", detailItem.threshold],
                    ].map(([l, v]) => (
                      <div key={l} className="detailRow"><span>{l}</span><strong>{v}</strong></div>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {manualHistory.length === 0 ? (
                    <div className="emptyState">
                      <div className="emptyIcon"><History size={32} /></div>
                      <h3>No manual predictions yet</h3>
                      <p>Make a prediction from the Prediction page to see history here.</p>
                    </div>
                  ) : (
                    <div className="panel">
                      <div className="historyControls">
                        <input
                          className="historySearch"
                          placeholder="Search by ID, type, amount..."
                          value={search}
                          onChange={e => setSearch(e.target.value)}
                        />
                        <select
                          className="historyFilter"
                          value={resultFilter}
                          onChange={e => setResultFilter(e.target.value)}
                        >
                          <option value="All">All Results</option>
                          <option value="FRAUD">Fraud</option>
                          <option value="LEGITIMATE">Legitimate</option>
                        </select>
                        <select
                          className="historyFilter"
                          value={typeFilter}
                          onChange={e => setTypeFilter(e.target.value)}
                        >
                          <option value="All">All Types</option>
                          <option value="CASH_IN">CASH_IN</option>
                          <option value="CASH_OUT">CASH_OUT</option>
                          <option value="DEBIT">DEBIT</option>
                          <option value="PAYMENT">PAYMENT</option>
                          <option value="TRANSFER">TRANSFER</option>
                        </select>
                        <button className="downloadButton" onClick={exportManual}>
                          <Download size={15} />Export CSV
                        </button>
                        <button className="dangerButton" onClick={clearManual}>
                          <Trash2 size={15} />Clear
                        </button>
                      </div>
                      <div className="tableWrapper">
                        <table className="resultsTable">
                          <thead>
                            <tr>
                              <th>ID</th><th>Time</th><th>Type</th><th>Amount</th><th>Currency</th>
                              <th>Device</th><th>Result</th><th>Probability</th><th></th>
                            </tr>
                          </thead>
                          <tbody>
                            {filtered.map(r => (
                              <tr key={r.id}>
                                <td style={{ fontFamily: "monospace", fontSize: 11 }}>{r.transaction_id}</td>
                                <td>{formatTime(r.Time)}</td>
                                <td>{r.type}</td>
                                <td>{Number(r.amount).toLocaleString()}</td>
                                <td>{r.currency || "-"}</td>
                                <td>{r.device_type || "-"}</td>
                                <td>
                                  <span className={`predictionBadge ${r.result === "FRAUD" ? "fraudBadge" : "legitBadge"}`}>
                                    {r.result}
                                  </span>
                                </td>
                                <td>{(Number(r.fraud_probability) * 100).toFixed(4)}%</td>
                                <td>
                                  <button className="iconBtn" onClick={() => setDetailItem(r)} title="View details">
                                    <Eye size={15} />
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* CSV TAB */}
          {tab === "csv" && (
            <>
              {csvDetail ? (
                <div className="panel" style={{ padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <div>
                      <h3 style={{ margin: 0, fontSize: 16 }}>{csvDetail.filename}</h3>
                      <p style={{ margin: "4px 0 0", fontSize: 12, color: "#667085" }}>
                        Uploaded: {formatTime(csvDetail.uploaded_at)}
                      </p>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button className="downloadButton" onClick={() => exportCsv(csvDetail.id)}>
                        <Download size={15} />Export CSV
                      </button>
                      <button
                        className="closeButton"
                        style={{ width: "auto", marginTop: 0, padding: "6px 14px" }}
                        onClick={() => setCsvDetail(null)}
                      >
                        ← Back
                      </button>
                    </div>
                  </div>
                  <div className="csvSummary" style={{ marginBottom: 16 }}>
                    <CSVStat title="Total" value={csvDetail.total_rows} type="blue" />
                    <CSVStat title="Fraud" value={csvDetail.fraud_count} type="red" />
                    <CSVStat title="Legitimate" value={csvDetail.legitimate_count} type="green" />
                    <CSVStat title="Fraud Rate" value={`${(csvDetail.fraud_rate * 100).toFixed(2)}%`} type="orange" />
                  </div>
                  {csvDetailLoading ? (
                    <p>Loading data...</p>
                  ) : csvDetail.predictions && csvDetail.predictions.length > 0 && (
                    <div className="tableWrapper">
                      <table className="resultsTable">
                        <thead>
                          <tr>
                            {(csvDetail.columns || Object.keys(csvDetail.predictions[0])).map(c => (
                              <th key={c}>{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {csvDetail.predictions.map((row, i) => {
                            const cols = csvDetail.columns || Object.keys(row);
                            const isFraud = Number(row.prediction) === 1 || row.result === "FRAUD";
                            return (
                              <tr key={i}>
                                {cols.map(c => (
                                  <td key={c}>
                                    {c === "prediction" || c === "result" ? (
                                      <span className={`predictionBadge ${isFraud ? "fraudBadge" : "legitBadge"}`}>
                                        {c === "result" ? row[c] : row[c]}
                                      </span>
                                    ) : c === "fraud_probability" ? (
                                      row[c] !== undefined ? `${(Number(row[c]) * 100).toFixed(4)}%` : "-"
                                    ) : (
                                      row[c] ?? "-"
                                    )}
                                  </td>
                                ))}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {csvHistory.length === 0 ? (
                    <div className="emptyState">
                      <div className="emptyIcon"><Upload size={32} /></div>
                      <h3>No CSV analyses yet</h3>
                      <p>Upload a CSV file from the CSV Analysis page to see history here.</p>
                    </div>
                  ) : (
                    <div>
                      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
                        <button className="dangerButton" onClick={clearCsv}>
                          <Trash2 size={15} />Clear All CSV History
                        </button>
                      </div>
                      <div style={{ display: "grid", gap: 14 }}>
                        {csvHistory.map(c => (
                          <div key={c.id} className="panel" style={{ padding: 18 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <div>
                                <div style={{ fontWeight: 600, fontSize: 14, color: "#111827" }}>{c.filename}</div>
                                <div style={{ fontSize: 12, color: "#667085", marginTop: 3 }}>
                                  Uploaded: {formatTime(c.uploaded_at)}
                                </div>
                              </div>
                              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                                <span style={{ fontSize: 12, color: "#2563eb", fontWeight: 600 }}>{c.total_rows} rows</span>
                                <span className={`predictionBadge ${c.fraud_count > 0 ? "fraudBadge" : "legitBadge"}`}>
                                  {c.fraud_count} fraud
                                </span>
                                <button
                                  className="downloadButton"
                                  style={{ fontSize: 12, padding: "6px 12px" }}
                                  onClick={() => viewCsvDetail(c.id)}
                                >
                                  <Eye size={14} />View Details
                                </button>
                                <button
                                  className="downloadButton"
                                  style={{ fontSize: 12, padding: "6px 12px", background: "#16a34a" }}
                                  onClick={() => exportCsv(c.id)}
                                >
                                  <Download size={14} />Export
                                </button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

/* ================================================================== */
/*  ANALYTICS PAGE — Unified Real Data from Manual + CSVs              */
/* ================================================================== */
function AnalyticsPage() {
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState("All");

  useEffect(() => {
    fetch(`${API}/analytics-dataset`)
      .then(r => r.json())
      .then(data => {
        setDataset(data);
        setLoading(false);
      })
      .catch(e => {
        console.error("Analytics fetch error:", e);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="emptyState"><p>Loading analytics...</p></div>;

  const records = source === "Manual"
    ? (dataset?.manual || [])
    : source === "CSV"
      ? (dataset?.csv || [])
      : (dataset?.combined || []);

  const total = records.length;

  if (total === 0) {
    return (
      <div>
        <div className="pageIntro">
          <div>
            <h2>History Analytics</h2>
            <p>Comprehensive insights calculated from actual manual predictions and CSV analyses.</p>
          </div>
          <div className="modelBadge light">
            <ChartNoAxesCombined size={18} />
            <div><strong>Analytics</strong><span>Combined Dataset</span></div>
          </div>
        </div>
        <div className="emptyState" style={{ marginTop: 40 }}>
          <div className="emptyIcon"><ChartNoAxesCombined size={32} /></div>
          <h3>No prediction history available yet</h3>
          <p>Make a prediction or analyze a CSV to generate real analytics.</p>
        </div>
      </div>
    );
  }

  // Calculate actual statistics from records
  const fraud = records.filter(r => Number(r.prediction) === 1 || r.result === "FRAUD").length;
  const legitimate = total - fraud;
  const fraudRate = (fraud / total) * 100;
  const probs = records.map(r => Number(r.fraud_probability) || 0);
  const avgProb = (probs.reduce((a, b) => a + b, 0) / total) * 100;
  const maxProb = (Math.max(...probs)) * 100;

  // Chart 1: Fraud vs Legitimate
  const pieData = [
    { name: "Fraud", value: fraud },
    { name: "Legitimate", value: legitimate }
  ].filter(d => d.value > 0);

  // Chart 2: Predictions by transaction type
  const typeMap = {};
  records.forEach(r => {
    const t = r.type || "OTHER";
    if (!typeMap[t]) typeMap[t] = { type: t, total: 0, fraud: 0 };
    typeMap[t].total++;
    if (Number(r.prediction) === 1 || r.result === "FRAUD") {
      typeMap[t].fraud++;
    }
  });
  const typeData = Object.values(typeMap);

  // Chart 3: Probability Distribution in 10 buckets
  const probBuckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}%`,
    count: 0
  }));
  probs.forEach(p => {
    const bucketIdx = Math.min(Math.floor(p * 10), 9);
    probBuckets[bucketIdx].count++;
  });

  // Chart 4: Activity over time
  const timeMap = {};
  records.forEach(r => {
    const dateStr = (r.Time || "").slice(0, 10) || "Recent";
    if (!timeMap[dateStr]) timeMap[dateStr] = { date: dateStr, count: 0, fraud: 0 };
    timeMap[dateStr].count++;
    if (Number(r.prediction) === 1 || r.result === "FRAUD") timeMap[dateStr].fraud++;
  });
  const timeData = Object.values(timeMap).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>History Analytics</h2>
          <p>Comprehensive insights calculated from actual manual predictions and CSV analyses.</p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select
            className="historyFilter"
            value={source}
            onChange={e => setSource(e.target.value)}
          >
            <option value="All">All Data ({dataset?.combined?.length || 0})</option>
            <option value="Manual">Manual Only ({dataset?.manual?.length || 0})</option>
            <option value="CSV">CSV Only ({dataset?.csv?.length || 0})</option>
          </select>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="analyticsGrid6">
        <div className="analyticCard highlight">
          <span>Total Predictions</span>
          <strong>{total.toLocaleString()}</strong>
        </div>
        <div className="analyticCard fraud">
          <span>Fraud Predictions</span>
          <strong>{fraud.toLocaleString()}</strong>
        </div>
        <div className="analyticCard legit">
          <span>Legitimate Predictions</span>
          <strong>{legitimate.toLocaleString()}</strong>
        </div>
        <div className="analyticCard">
          <span>Fraud Rate</span>
          <strong>{fraudRate.toFixed(2)}%</strong>
        </div>
        <div className="analyticCard">
          <span>Avg Fraud Probability</span>
          <strong>{avgProb.toFixed(4)}%</strong>
        </div>
        <div className="analyticCard fraud">
          <span>Highest Probability</span>
          <strong>{maxProb.toFixed(4)}%</strong>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="chartsGrid">
        {/* Fraud vs Legitimate */}
        <div className="chartCard">
          <h3>Fraud vs Legitimate</h3>
          <p>Real distribution across {source.toLowerCase()} records</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Predictions by Transaction Type */}
        <div className="chartCard">
          <h3>Predictions by Transaction Type</h3>
          <p>Transaction breakdown and fraud occurrence</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={typeData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="type" axisLine={false} tickLine={false} fontSize={11} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
                <Bar dataKey="total" fill="#2563eb" name="Total" radius={[4, 4, 0, 0]} />
                <Bar dataKey="fraud" fill="#dc2626" name="Fraud" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Fraud Probability Distribution */}
        <div className="chartCard">
          <h3>Fraud Probability Distribution</h3>
          <p>Histogram of actual prediction confidence levels</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probBuckets}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="range" axisLine={false} tickLine={false} fontSize={10} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" name="Transactions" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Activity Over Time */}
        <div className="chartCard">
          <h3>Prediction Activity Over Time</h3>
          <p>Actual timestamped activity progression</p>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" axisLine={false} tickLine={false} fontSize={10} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#2563eb" fill="#dbeafe" name="Total Predictions" />
                <Area type="monotone" dataKey="fraud" stroke="#dc2626" fill="#fee2e2" name="Fraud Detections" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  SETTINGS PAGE (PRESERVED EXACTLY)                                  */
/* ================================================================== */
function SettingsPage() {
  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>
      <div style={{ marginBottom: "28px" }}>
        <h1 style={{ margin: 0, fontSize: "30px", fontWeight: "700", color: "#111827" }}>Settings</h1>
        <p style={{ marginTop: "8px", color: "#6b7280", fontSize: "15px" }}>Configure and monitor your FraudGuard application.</p>
      </div>
      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "24px", marginBottom: "22px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
        <h2 style={{ margin: "0 0 20px", fontSize: "20px", color: "#111827" }}>API Configuration</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "18px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "8px", fontSize: "13px", fontWeight: "600", color: "#6b7280" }}>Backend API URL</label>
            <input type="text" value={API} readOnly style={{ width: "100%", boxSizing: "border-box", padding: "12px 14px", border: "1px solid #d1d5db", borderRadius: "10px", background: "#f9fafb", color: "#374151", fontSize: "14px", outline: "none" }} />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: "8px", fontSize: "13px", fontWeight: "600", color: "#6b7280" }}>API Status</label>
            <div style={{ padding: "12px 14px", borderRadius: "10px", background: "#ecfdf5", color: "#047857", fontSize: "14px", fontWeight: "600" }}>● Connected</div>
          </div>
        </div>
      </div>
      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "24px", marginBottom: "22px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
        <h2 style={{ margin: "0 0 20px", fontSize: "20px", color: "#111827" }}>Model Configuration</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "18px" }}>
          {[["Algorithm", "XGBoost"], ["Model Version", "v2.0"], ["Features", "20"]].map(([l, v]) => (
            <div key={l} style={{ padding: "18px", borderRadius: "12px", background: "#f9fafb" }}>
              <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>{l}</p>
              <h3 style={{ margin: "8px 0 0", color: "#111827", fontSize: "18px" }}>{v}</h3>
            </div>
          ))}
        </div>
      </div>
      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "24px", marginBottom: "22px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
        <h2 style={{ margin: "0 0 20px", fontSize: "20px", color: "#111827" }}>Monitoring</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
          {[["Data Drift Monitoring", "Evidently AI"], ["Concept Drift Monitoring", "Model performance tracking"]].map(([t, s]) => (
            <div key={t} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px", border: "1px solid #e5e7eb", borderRadius: "12px" }}>
              <div><div style={{ fontWeight: "600", color: "#111827" }}>{t}</div><div style={{ marginTop: "4px", fontSize: "13px", color: "#6b7280" }}>{s}</div></div>
              <span style={{ padding: "6px 10px", borderRadius: "20px", background: "#ecfdf5", color: "#047857", fontSize: "12px", fontWeight: "600" }}>Enabled</span>
            </div>
          ))}
        </div>
      </div>
      <div style={{ background: "linear-gradient(135deg, #111827, #1f2937)", borderRadius: "18px", padding: "26px", color: "white" }}>
        <h2 style={{ margin: "0 0 18px", fontSize: "20px" }}>System Information</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "18px" }}>
          {[["Application", "FraudGuard"], ["Version", "2.0"], ["Backend", "FastAPI"], ["ML Framework", "XGBoost"]].map(([l, v]) => (
            <div key={l}><div style={{ color: "#9ca3af", fontSize: "12px" }}>{l}</div><div style={{ marginTop: "5px", fontWeight: "600" }}>{v}</div></div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  MODEL PAGE (PRESERVED EXACTLY)                                     */
/* ================================================================== */
function Model() {
  const [latestData, setLatestData] = useState(null);
  useEffect(() => {
    fetch(`${API}/concept-drift`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { if (data.length > 0) setLatestData(data[data.length - 1]); })
      .catch(e => console.error("Model performance error:", e));
  }, []);

  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>
      <div style={{ marginBottom: "28px" }}>
        <h1 style={{ margin: 0, fontSize: "30px", fontWeight: "700", color: "#111827" }}>Production Model</h1>
        <p style={{ marginTop: "8px", color: "#6b7280", fontSize: "15px" }}>XGBoost fraud detection model and performance information.</p>
      </div>
      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "24px", marginBottom: "24px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", gap: "18px", alignItems: "center" }}>
            <div style={{ width: "58px", height: "58px", borderRadius: "15px", background: "#eff6ff", display: "flex", alignItems: "center", justifyContent: "center" }}><BrainCircuit size={28} color="#2563eb" /></div>
            <div>
              <h2 style={{ margin: 0, fontSize: "22px", color: "#111827" }}>XGBoost Fraud Detection</h2>
              <p style={{ margin: "6px 0 0", color: "#6b7280", fontSize: "14px" }}>Production classification model</p>
            </div>
          </div>
          <div style={{ padding: "8px 14px", borderRadius: "20px", background: "#ecfdf5", color: "#047857", fontSize: "13px", fontWeight: "600" }}>● Production</div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "18px", marginBottom: "24px" }}>
        {[["Algorithm", "XGBoost"], ["Model File", "xgboost_fraud_model_production.json"], ["Features", "20"]].map(([l, v]) => (
          <div key={l} style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "20px" }}>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>{l}</p>
            <h3 style={{ margin: "9px 0 0", color: "#111827", fontSize: l === "Model File" ? "15px" : undefined }}>{v}</h3>
          </div>
        ))}
      </div>
      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "24px", marginBottom: "24px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
        <h2 style={{ margin: "0 0 20px", fontSize: "20px", color: "#111827" }}>Model Performance</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "18px" }}>
          {[["ROC-AUC", "roc_auc"], ["PR-AUC", "pr_auc"], ["Precision", "precision"], ["Recall", "recall"]].map(([label, key]) => (
            <div key={key}>
              <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>{label}</p>
              <h3 style={{ margin: "8px 0 0", fontSize: "26px" }}>{latestData ? Number(latestData[key]).toFixed(6) : "—"}</h3>
            </div>
          ))}
        </div>
      </div>
      <div style={{ background: "linear-gradient(135deg, #111827, #1f2937)", borderRadius: "18px", padding: "26px", color: "white", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "20px" }}>Production Model Status</h2>
          <p style={{ margin: "7px 0 0", color: "#d1d5db", fontSize: "14px" }}>XGBoost model is loaded and available for fraud prediction.</p>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: "13px", color: "#d1d5db", marginBottom: "5px" }}>F1 Score</div>
          <div style={{ fontSize: "30px", fontWeight: "700" }}>{latestData ? Number(latestData.f1_score).toFixed(6) : "—"}</div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  MONITORING PAGE (PRESERVED EXACTLY)                                */
/* ================================================================== */
function Monitoring() {
  const [conceptData, setConceptData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dataDrift, setDataDrift] = useState(null);
  const [dataDriftAvailable, setDataDriftAvailable] = useState(false);

  useEffect(() => {
    fetch(`${API}/concept-drift`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setConceptData(data); setLoading(false); })
      .catch(() => setLoading(false));

    fetch(`${API}/data-drift`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setDataDrift(data); setDataDriftAvailable(true); })
      .catch(() => setDataDriftAvailable(false));
  }, []);

  const latest = conceptData[conceptData.length - 1];

  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "28px" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "30px", fontWeight: "700", color: "#111827" }}>Model Monitoring</h1>
          <p style={{ marginTop: "8px", color: "#6b7280", fontSize: "15px" }}>Monitor data drift, concept drift, and model performance.</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "9px 15px", borderRadius: "20px", background: "#ecfdf5", color: "#047857", fontSize: "14px", fontWeight: "600" }}>
          <span style={{ width: "9px", height: "9px", borderRadius: "50%", background: "#10b981" }} />
          API Online
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "20px", marginBottom: "28px" }}>
        <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "22px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <p style={{ margin: 0, color: "#6b7280", fontSize: "13px", fontWeight: "600", textTransform: "uppercase" }}>Data Drift</p>
              <h2 style={{ margin: "8px 0 5px", fontSize: "21px", color: "#111827" }}>
                {dataDriftAvailable ? (dataDrift?.dataset_drift ? "Dataset Drift Detected" : "Dataset Stable") : "Data Drift Unavailable"}
              </h2>
              <div style={{ margin: 0, color: "#9ca3af", fontSize: "14px" }}>
                Monitoring input feature changes
                {dataDriftAvailable && dataDrift && (
                  <div style={{ margin: "8px 0 0", color: "#6b7280", fontSize: "13px", fontWeight: "600" }}>
                    {dataDrift.drifted_columns} of {dataDrift.total_columns} features drifted
                  </div>
                )}
              </div>
            </div>
            <div style={{ width: "48px", height: "48px", borderRadius: "14px", background: "#eff6ff", display: "flex", alignItems: "center", justifyContent: "center" }}><BarChart3 size={23} color="#2563eb" /></div>
          </div>
          <div style={{ marginTop: "20px", display: "inline-flex", alignItems: "center", gap: "7px", padding: "7px 12px", borderRadius: "20px", background: "#ecfdf5", color: "#047857", fontSize: "13px", fontWeight: "600" }}>● Available</div>
        </div>

        <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "22px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <p style={{ margin: 0, color: "#6b7280", fontSize: "13px", fontWeight: "600", textTransform: "uppercase" }}>Concept Drift</p>
              <h2 style={{ margin: "8px 0 5px", fontSize: "21px", color: "#111827" }}>
                {conceptData.length > 0 ? "Performance Monitored" : "Concept Drift Unavailable"}
              </h2>
              <p style={{ margin: 0, color: "#9ca3af", fontSize: "14px" }}>Performance across datasets</p>
            </div>
            <div style={{ width: "48px", height: "48px", borderRadius: "14px", background: "#f5f3ff", display: "flex", alignItems: "center", justifyContent: "center" }}><TrendingUp size={23} color="#8b5cf6" /></div>
          </div>
          <div style={{ marginTop: "20px", display: "inline-flex", alignItems: "center", gap: "7px", padding: "7px 12px", borderRadius: "20px", background: "#ecfdf5", color: "#047857", fontSize: "13px", fontWeight: "600" }}>● Available</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "18px", marginBottom: "28px" }}>
        {[
          ["Latest Fraud Rate", latest ? `${(Number(latest.fraud_rate) * 100).toFixed(3)}%` : "--", "Test dataset"],
          ["ROC-AUC", latest ? Number(latest.roc_auc).toFixed(6) : "--", "Model discrimination"],
          ["PR-AUC", latest ? Number(latest.pr_auc).toFixed(6) : "--", "Precision-recall quality"],
          ["F1 Score", latest ? Number(latest.f1_score).toFixed(6) : "--", "Classification balance"]
        ].map(([t, v, s]) => (
          <div key={t} style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", padding: "20px", boxShadow: "0 4px 15px rgba(0,0,0,0.04)" }}>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>{t}</p>
            <h2 style={{ margin: "10px 0 0", fontSize: "28px", color: "#111827" }}>{v}</h2>
            <p style={{ margin: "7px 0 0", color: "#9ca3af", fontSize: "12px" }}>{s}</p>
          </div>
        ))}
      </div>

      <div style={{ background: "#ffffff", border: "1px solid #e5e7eb", borderRadius: "16px", overflow: "hidden", boxShadow: "0 4px 15px rgba(0,0,0,0.04)", marginBottom: "28px" }}>
        <div style={{ padding: "22px 24px", borderBottom: "1px solid #e5e7eb" }}>
          <h2 style={{ margin: 0, fontSize: "20px", color: "#111827" }}>Concept Drift Results</h2>
          <p style={{ margin: "6px 0 0", color: "#9ca3af", fontSize: "14px" }}>Performance comparison across train, validation, and test datasets</p>
        </div>
        {loading ? (
          <div style={{ padding: "50px", textAlign: "center", color: "#6b7280" }}>Loading monitoring data...</div>
        ) : conceptData.length === 0 ? (
          <div style={{ padding: "50px", textAlign: "center", color: "#6b7280" }}>No monitoring data available.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ background: "#f9fafb" }}>
                  {["Period", "Rows", "Fraud Count", "Fraud Rate", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1 Score"].map(h => (
                    <th key={h} style={{ padding: "14px 16px", textAlign: "left", color: "#6b7280", fontWeight: "600", whiteSpace: "nowrap", borderBottom: "1px solid #e5e7eb" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {conceptData.map(row => (
                  <tr key={row.period} style={{ borderBottom: "1px solid #f3f4f6" }}>
                    <td style={{ padding: "15px 16px" }}>
                      <span style={{ padding: "5px 10px", borderRadius: "8px", background: row.period === "test" ? "#eff6ff" : "#f9fafb", color: row.period === "test" ? "#2563eb" : "#374151", fontWeight: "600", textTransform: "capitalize" }}>
                        {row.period}
                      </span>
                    </td>
                    <td style={{ padding: "15px 16px", color: "#374151" }}>{row.rows.toLocaleString()}</td>
                    <td style={{ padding: "15px 16px", color: "#374151" }}>{row.fraud_count.toLocaleString()}</td>
                    <td style={{ padding: "15px 16px", color: "#374151", fontWeight: "600" }}>{(row.fraud_rate * 100).toFixed(3)}%</td>
                    <td style={{ padding: "15px 16px", color: "#111827" }}>{row.roc_auc.toFixed(6)}</td>
                    <td style={{ padding: "15px 16px", color: "#111827" }}>{row.pr_auc.toFixed(6)}</td>
                    <td style={{ padding: "15px 16px", color: "#111827" }}>{row.precision.toFixed(6)}</td>
                    <td style={{ padding: "15px 16px", color: "#111827" }}>{row.recall.toFixed(6)}</td>
                    <td style={{ padding: "15px 16px", color: "#047857", fontWeight: "700" }}>{row.f1_score.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div style={{ background: "linear-gradient(135deg, #111827, #1f2937)", borderRadius: "18px", padding: "26px", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "20px", color: "white" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "20px" }}>Data Drift Report</h2>
          <p style={{ margin: "7px 0 0", color: "#d1d5db", fontSize: "14px" }}>Evidently AI analysis of feature distribution changes.</p>
        </div>
        <a href={`${API}/data_drift_report.html`} target="_blank" rel="noreferrer" style={{ textDecoration: "none", background: "#ffffff", color: "#111827", padding: "11px 18px", borderRadius: "10px", fontSize: "14px", fontWeight: "600", whiteSpace: "nowrap" }}>
          Open Drift Report →
        </a>
      </div>
    </div>
  );
}

/* ================================================================== */
/*  DASHBOARD (PRESERVED EXACTLY)                                      */
/* ================================================================== */
function Dashboard() {
  const [conceptData, setConceptData] = useState([]);
  const [apiOnline, setApiOnline] = useState(false);
  const [dataDriftAvailable, setDataDriftAvailable] = useState(false);
  const [driftedColumns, setDriftedColumns] = useState(0);
  const [totalColumns, setTotalColumns] = useState(20);
  const [datasetDrift, setDatasetDrift] = useState(false);
  const fraudData = conceptData.map(item => ({ day: item.period, transactions: item.rows, fraud: item.fraud_count }));

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));

    fetch(`${API}/concept-drift`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => setConceptData(data))
      .catch(() => {});

    fetch(`${API}/data-drift`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => {
        setDriftedColumns(data.drifted_columns);
        setTotalColumns(data.total_columns);
        setDatasetDrift(data.dataset_drift);
        setDataDriftAvailable(true);
      })
      .catch(() => setDataDriftAvailable(false));
  }, []);

  const latestData = conceptData.length > 0 ? conceptData[conceptData.length - 1] : null;

  return (
    <>
      <section className="welcome">
        <div>
          <div className="welcomeLabel">MLOPS OVERVIEW</div>
          <h2>Fraud Detection Dashboard</h2>
          <p>Monitor transactions, model performance, and production health from one place.</p>
        </div>
        <div className="modelBadge">
          <Brain size={18} />
          <div><strong>XGBoost</strong><span>Production Model</span></div>
        </div>
      </section>

      <section className="statsGrid">
        <StatCard title="Model Accuracy" value={latestData ? `${(Number(latestData.roc_auc) * 100).toFixed(4)}%` : "—"} subtitle="ROC-AUC" icon={<TrendingUp size={22} />} type="blue" />
        <StatCard title="Fraud Recall" value={latestData ? `${(Number(latestData.recall) * 100).toFixed(2)}%` : "—"} subtitle="Latest test result" icon={<ShieldAlert size={22} />} type="purple" />
        <StatCard title="API Status" value={apiOnline ? "Healthy" : "Offline"} subtitle="FastAPI backend" icon={<CheckCircle size={22} />} type="green" />
        <StatCard title="Monitoring" value={apiOnline && dataDriftAvailable ? "Active" : "Offline"} subtitle="Drift monitoring" icon={<Activity size={22} />} type="orange" />
      </section>

      <section className="dashboardGrid">
        <div className="panel chartPanel">
          <div className="panelHeader">
            <div>
              <h3>Dataset Activity</h3>
              <p>Transaction and fraud distribution across datasets</p>
            </div>
            <div className="legend">
              <span><i className="legendBlue"></i>Transactions</span>
              <span><i className="legendPurple"></i>Fraud</span>
            </div>
          </div>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={fraudData}>
                <defs>
                  <linearGradient id="transactionGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopOpacity={0.3} />
                    <stop offset="100%" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="fraudGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopOpacity={0.25} />
                    <stop offset="100%" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="day" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip />
                <Area type="monotone" dataKey="transactions" strokeWidth={2} fill="url(#transactionGradient)" />
                <Area type="monotone" dataKey="fraud" strokeWidth={2} fill="url(#fraudGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div>
              <h3>Model Performance</h3>
              <p>Production evaluation</p>
            </div>
            <Brain size={22} />
          </div>
          <div className="metricList">
            {modelMetrics.map(m => (
              <div className="metricRow" key={m.name}>
                <div className="metricInfo"><span>{m.name}</span><strong>{m.value}%</strong></div>
                <div className="progress"><div className="progressValue" style={{ width: `${m.value}%` }}></div></div>
              </div>
            ))}
          </div>
          <div className="aucBox">
            <div><span>ROC-AUC</span><strong>0.999999</strong></div>
            <div><span>PR-AUC</span><strong>0.999878</strong></div>
          </div>
        </div>
      </section>

      <section className="bottomGrid">
        <div className="panel">
          <div className="panelHeader">
            <div>
              <h3>Monitoring Status</h3>
              <p>Current MLOps monitoring components</p>
            </div>
            <Activity size={21} />
          </div>
          <MonitoringItem
            icon={<Database size={18} />}
            title="Data Drift"
            status={dataDriftAvailable ? (datasetDrift ? "Dataset Drift Detected" : "Dataset Stable") : "Offline"}
            detail={dataDriftAvailable ? `${driftedColumns} of ${totalColumns} features drifted` : "Report unavailable"}
          />
          <MonitoringItem
            icon={<BarChart3 size={18} />}
            title="Concept Drift"
            status={conceptData.length > 0 ? "Monitored" : "Offline"}
            detail={conceptData.length > 0 ? "Performance comparison" : "No monitoring data"}
          />
          <MonitoringItem
            icon={<Server size={18} />}
            title="FastAPI"
            status={apiOnline ? "Healthy" : "Offline"}
            detail={API}
          />
        </div>

        <div className="panel">
          <div className="panelHeader">
            <div>
              <h3>Dataset Summary</h3>
              <p>Fraud detection dataset</p>
            </div>
            <FileSearch size={21} />
          </div>
          <div className="datasetStats">
            <DatasetStat label="Training Rows" value={conceptData.length > 0 ? conceptData[0].rows.toLocaleString() : "—"} />
            <DatasetStat label="Validation Rows" value={conceptData.length > 1 ? conceptData[1].rows.toLocaleString() : "—"} />
            <DatasetStat label="Test Rows" value={latestData ? latestData.rows.toLocaleString() : "—"} />
            <DatasetStat label="Features" value="20" />
          </div>
          <div className="fraudRate">
            <div>
              <span>Test fraud rate</span>
              <strong>{latestData ? `${(Number(latestData.fraud_rate) * 100).toFixed(2)}%` : "—"}</strong>
            </div>
            <AlertTriangle size={20} />
          </div>
        </div>
      </section>
    </>
  );
}

/* ================================================================== */
/*  SHARED COMPONENTS                                                  */
/* ================================================================== */
function StatCard({ title, value, subtitle, icon, type }) {
  return (
    <div className={`statCard ${type}`}>
      <div className="statTop"><span>{title}</span><div className="statIcon">{icon}</div></div>
      <h3>{value}</h3>
      <p>{subtitle}</p>
    </div>
  );
}

function MonitoringItem({ icon, title, status, detail }) {
  return (
    <div className="monitorItem">
      <div className="monitorIcon">{icon}</div>
      <div className="monitorText"><strong>{title}</strong><span>{detail}</span></div>
      <div className="monitorStatus"><span className="smallDot"></span>{status}</div>
    </div>
  );
}

function DatasetStat({ label, value }) {
  return (
    <div className="datasetStat"><span>{label}</span><strong>{value}</strong></div>
  );
}

export default App;
