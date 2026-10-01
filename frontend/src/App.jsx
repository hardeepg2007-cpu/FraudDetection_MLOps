import { useState, useEffect } from "react";
import {
  LayoutDashboard,
  ShieldAlert,
  Upload,
  Activity,
  Brain,
  Settings,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Server,
  Database,
  BarChart3,
  FileSearch,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";

// Set VITE_API_BASE_URL in Vercel. The fallback keeps the local app working.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const modelMetrics = [
  { name: "Precision", value: 99.83 },
  { name: "Recall", value: 99.92 },
  { name: "F1 Score", value: 99.87 },
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];

    if (character === '"') {
      if (inQuotes && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (character === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((character === "\n" || character === "\r") && !inQuotes) {
      if (character === "\r" && text[index + 1] === "\n") {
        index += 1;
      }
      row.push(value);
      if (row.some((cell) => cell !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += character;
    }
  }

  row.push(value);
  if (row.some((cell) => cell !== "")) {
    rows.push(row);
  }

  const [headers = [], ...dataRows] = rows;
  return dataRows.map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]))
  );
}

function App() {
  const [activePage, setActivePage] = useState("Dashboard");

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Prediction", icon: ShieldAlert },
    { name: "CSV Analysis", icon: Upload },
    { name: "Monitoring", icon: Activity },
    { name: "Model", icon: Brain },
    { name: "Settings", icon: Settings },
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandIcon">
            <ShieldAlert size={25} />
          </div>
          <div>
            <div className="brandName">FraudGuard</div>
            <div className="brandSub">MLOps Platform</div>
          </div>
        </div>

        <div className="menuTitle">MAIN MENU</div>

        <nav>
          {menuItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                className={`navButton ${
                  activePage === item.name ? "active" : ""
                }`}
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
            <div className="systemHeader">
              <Server size={17} />
              <span>System Status</span>
            </div>

            <div className="systemStatus">
              <span className="onlineDot"></span>
              All systems operational
            </div>
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

          <div className="apiStatus">
            <span className="onlineDot"></span>
            <span>API Online</span>
          </div>
        </header>

        <div className="content">
          {activePage === "Dashboard" && <Dashboard />}

          {activePage === "Prediction" && <Prediction />}

          {activePage === "CSV Analysis" && <CSVAnalysis />}

          {activePage === "Monitoring" && <Monitoring />}
          {activePage === "Model" && <Model />}

          {activePage === "Settings" && <SettingsPage />}
        </div>
      </main>
    </div>
  );
}
function CSVAnalysis() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    setFile(selectedFile || null);
    setResults([]);
    setSummary(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);
    setSummary(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_BASE_URL}/predict_csv`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("CSV prediction failed");
      }

      const contentType = response.headers.get("content-type") || "";
      let predictionResults;

      if (contentType.includes("application/json")) {
        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }
        predictionResults = data.predictions || data.results || data;
      } else {
        predictionResults = parseCsv(await response.text());
      }

      if (!Array.isArray(predictionResults)) {
        throw new Error("The CSV API returned an unexpected response.");
      }

      setResults(predictionResults);

      {
        const fraudCount = predictionResults.filter(
          (row) =>
            Number(row.prediction) === 1 ||
            row.result === "FRAUD"
        ).length;

        const legitimateCount =
          predictionResults.length - fraudCount;

        setSummary({
          total: predictionResults.length,
          fraud: fraudCount,
          legitimate: legitimateCount,
        });
      }
    } catch (err) {
      setError(err.message || "Could not connect to the CSV prediction API. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>CSV Transaction Analysis</h2>
          <p>
            Upload a CSV file to analyze multiple transactions
            using the production XGBoost model.
          </p>
        </div>

        <div className="modelBadge light">
          <Upload size={18} />
          <div>
            <strong>Batch Prediction</strong>
            <span>CSV Analysis</span>
          </div>
        </div>
      </div>

      <div className="panel uploadPanel">
        <div className="uploadArea">
          <div className="uploadIcon">
            <Upload size={32} />
          </div>

          <h3>Upload Transaction CSV</h3>

          <p>
            Select a CSV file containing transaction details.
          </p>

          <label className="fileButton">
            Choose CSV File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />
          </label>

          {file && (
            <div className="selectedFile">
              <FileSearch size={17} />
              <span>{file.name}</span>
            </div>
          )}

          <button
            className="uploadButton"
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading
              ? "Analyzing Transactions..."
              : "Analyze CSV"}
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
          <CSVStat
            title="Total Transactions"
            value={summary.total}
            type="blue"
          />

          <CSVStat
            title="Fraud Detected"
            value={summary.fraud}
            type="red"
          />

          <CSVStat
            title="Legitimate"
            value={summary.legitimate}
            type="green"
          />

          <CSVStat
            title="Fraud Rate"
            value={
              summary.total
                ? `${(
                    (summary.fraud / summary.total) *
                    100
                  ).toFixed(2)}%`
                : "0%"
            }
            type="orange"
          />
        </div>
      )}

      {results.length > 0 && (
        <div className="panel resultsPanel">
          <div className="panelHeader">
            <div>
              <h3>Prediction Results</h3>
              <p>
                Results returned by the FastAPI prediction
                service
              </p>
            </div>

            <BarChart3 size={21} />
          </div>

          <div className="tableWrapper">
            <table className="resultsTable">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Type</th>
                  <th>Amount</th>
                  <th>Old Origin</th>
                  <th>New Origin</th>
                  <th>Old Destination</th>
                  <th>New Destination</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                </tr>
              </thead>

              <tbody>
                {results.map((row, index) => {
                  const isFraud =
                    row.prediction === 1 ||
                    row.result === "FRAUD";

                  return (
                    <tr key={index}>
                      <td>{row.step ?? "-"}</td>
                      <td>{row.type ?? "-"}</td>
                      <td>
                        {row.amount !== undefined
                          ? Number(row.amount).toLocaleString()
                          : "-"}
                      </td>
                      <td>
                        {row.oldbalanceOrg !== undefined
                          ? Number(
                              row.oldbalanceOrg
                            ).toLocaleString()
                          : "-"}
                      </td>
                      <td>
                        {row.newbalanceOrig !== undefined
                          ? Number(
                              row.newbalanceOrig
                            ).toLocaleString()
                          : "-"}
                      </td>
                      <td>
                        {row.oldbalanceDest !== undefined
                          ? Number(
                              row.oldbalanceDest
                            ).toLocaleString()
                          : "-"}
                      </td>
                      <td>
                        {row.newbalanceDest !== undefined
                          ? Number(
                              row.newbalanceDest
                            ).toLocaleString()
                          : "-"}
                      </td>

                      <td>
                        <span
                          className={`predictionBadge ${
                            isFraud
                              ? "fraudBadge"
                              : "legitBadge"
                          }`}
                        >
                          {isFraud
                            ? "FRAUD"
                            : "LEGITIMATE"}
                        </span>
                      </td>

                      <td>
                        {row.fraud_probability !== undefined
                          ? `${(
                              Number(
                                row.fraud_probability
                              ) * 100
                            ).toFixed(4)}%`
                          : "-"}
                      </td>
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
  return (
    <div className={`csvStat ${type}`}>
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}
function SettingsPage() {
  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>

      {/* HEADER */}
      <div style={{ marginBottom: "28px" }}>
        <h1
          style={{
            margin: 0,
            fontSize: "30px",
            fontWeight: "700",
            color: "#111827",
          }}
        >
          Settings
        </h1>

        <p
          style={{
            marginTop: "8px",
            color: "#6b7280",
            fontSize: "15px",
          }}
        >
          Configure and monitor your FraudGuard application.
        </p>
      </div>

      {/* API CONFIGURATION */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          padding: "24px",
          marginBottom: "22px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
        }}
      >
        <h2
          style={{
            margin: "0 0 20px",
            fontSize: "20px",
            color: "#111827",
          }}
        >
          API Configuration
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "18px",
          }}
        >
          <div>
            <label
              style={{
                display: "block",
                marginBottom: "8px",
                fontSize: "13px",
                fontWeight: "600",
                color: "#6b7280",
              }}
            >
              Backend API URL
            </label>

            <input
              type="text"
              value={API_BASE_URL}
              readOnly
              style={{
                width: "100%",
                boxSizing: "border-box",
                padding: "12px 14px",
                border: "1px solid #d1d5db",
                borderRadius: "10px",
                background: "#f9fafb",
                color: "#374151",
                fontSize: "14px",
                outline: "none",
              }}
            />
          </div>

          <div>
            <label
              style={{
                display: "block",
                marginBottom: "8px",
                fontSize: "13px",
                fontWeight: "600",
                color: "#6b7280",
              }}
            >
              API Status
            </label>

            <div
              style={{
                padding: "12px 14px",
                borderRadius: "10px",
                background: "#ecfdf5",
                color: "#047857",
                fontSize: "14px",
                fontWeight: "600",
              }}
            >
              ● Connected
            </div>
          </div>
        </div>
      </div>

      {/* MODEL CONFIGURATION */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          padding: "24px",
          marginBottom: "22px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
        }}
      >
        <h2
          style={{
            margin: "0 0 20px",
            fontSize: "20px",
            color: "#111827",
          }}
        >
          Model Configuration
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
            gap: "18px",
          }}
        >
          <div
            style={{
              padding: "18px",
              borderRadius: "12px",
              background: "#f9fafb",
            }}
          >
            <p
              style={{
                margin: 0,
                color: "#6b7280",
                fontSize: "13px",
              }}
            >
              Algorithm
            </p>

            <h3
              style={{
                margin: "8px 0 0",
                color: "#111827",
                fontSize: "18px",
              }}
            >
              XGBoost
            </h3>
          </div>

          <div
            style={{
              padding: "18px",
              borderRadius: "12px",
              background: "#f9fafb",
            }}
          >
            <p
              style={{
                margin: 0,
                color: "#6b7280",
                fontSize: "13px",
              }}
            >
              Model Version
            </p>

            <h3
              style={{
                margin: "8px 0 0",
                color: "#111827",
                fontSize: "18px",
              }}
            >
              v2.0
            </h3>
          </div>

          <div
            style={{
              padding: "18px",
              borderRadius: "12px",
              background: "#f9fafb",
            }}
          >
            <p
              style={{
                margin: 0,
                color: "#6b7280",
                fontSize: "13px",
              }}
            >
              Features
            </p>

            <h3
              style={{
                margin: "8px 0 0",
                color: "#111827",
                fontSize: "18px",
              }}
            >
              20
            </h3>
          </div>
        </div>
      </div>

      {/* MONITORING CONFIGURATION */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          padding: "24px",
          marginBottom: "22px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
        }}
      >
        <h2
          style={{
            margin: "0 0 20px",
            fontSize: "20px",
            color: "#111827",
          }}
        >
          Monitoring
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "16px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "16px",
              border: "1px solid #e5e7eb",
              borderRadius: "12px",
            }}
          >
            <div>
              <div
                style={{
                  fontWeight: "600",
                  color: "#111827",
                }}
              >
                Data Drift Monitoring
              </div>

              <div
                style={{
                  marginTop: "4px",
                  fontSize: "13px",
                  color: "#6b7280",
                }}
              >
                Evidently AI
              </div>
            </div>

            <span
              style={{
                padding: "6px 10px",
                borderRadius: "20px",
                background: "#ecfdf5",
                color: "#047857",
                fontSize: "12px",
                fontWeight: "600",
              }}
            >
              Enabled
            </span>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "16px",
              border: "1px solid #e5e7eb",
              borderRadius: "12px",
            }}
          >
            <div>
              <div
                style={{
                  fontWeight: "600",
                  color: "#111827",
                }}
              >
                Concept Drift Monitoring
              </div>

              <div
                style={{
                  marginTop: "4px",
                  fontSize: "13px",
                  color: "#6b7280",
                }}
              >
                Model performance tracking
              </div>
            </div>

            <span
              style={{
                padding: "6px 10px",
                borderRadius: "20px",
                background: "#ecfdf5",
                color: "#047857",
                fontSize: "12px",
                fontWeight: "600",
              }}
            >
              Enabled
            </span>
          </div>
        </div>
      </div>

      {/* SYSTEM INFORMATION */}
      <div
        style={{
          background: "linear-gradient(135deg, #111827, #1f2937)",
          borderRadius: "18px",
          padding: "26px",
          color: "white",
        }}
      >
        <h2
          style={{
            margin: "0 0 18px",
            fontSize: "20px",
          }}
        >
          System Information
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
            gap: "18px",
          }}
        >
          <div>
            <div style={{ color: "#9ca3af", fontSize: "12px" }}>
              Application
            </div>

            <div
              style={{
                marginTop: "5px",
                fontWeight: "600",
              }}
            >
              FraudGuard
            </div>
          </div>

          <div>
            <div style={{ color: "#9ca3af", fontSize: "12px" }}>
              Version
            </div>

            <div
              style={{
                marginTop: "5px",
                fontWeight: "600",
              }}
            >
              2.0
            </div>
          </div>

          <div>
            <div style={{ color: "#9ca3af", fontSize: "12px" }}>
              Backend
            </div>

            <div
              style={{
                marginTop: "5px",
                fontWeight: "600",
              }}
            >
              FastAPI
            </div>
          </div>

          <div>
            <div style={{ color: "#9ca3af", fontSize: "12px" }}>
              ML Framework
            </div>

            <div
              style={{
                marginTop: "5px",
                fontWeight: "600",
              }}
            >
              XGBoost
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
function Model() {
  const [latestData, setLatestData] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/concept-drift`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load model performance");
        }

        return response.json();
      })
      .then((data) => {
        if (data.length > 0) {
          setLatestData(data[data.length - 1]);
        }
      })
      .catch((error) => {
        console.error("Model performance error:", error);
      });
  }, []);
  
  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>

      {/* HEADER */}
      <div style={{ marginBottom: "28px" }}>
        <h1
          style={{
            margin: 0,
            fontSize: "30px",
            fontWeight: "700",
            color: "#111827",
          }}
        >
          Production Model
        </h1>

        <p
          style={{
            marginTop: "8px",
            color: "#6b7280",
            fontSize: "15px",
          }}
        >
          XGBoost fraud detection model and performance information.
        </p>
      </div>

      {/* MODEL OVERVIEW */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          padding: "24px",
          marginBottom: "24px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", gap: "18px", alignItems: "center" }}>
            <div
              style={{
                width: "58px",
                height: "58px",
                borderRadius: "15px",
                background: "#eff6ff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "28px",
              }}
            >
              🧠
            </div>

            <div>
              <h2
                style={{
                  margin: 0,
                  fontSize: "22px",
                  color: "#111827",
                }}
              >
                XGBoost Fraud Detection
              </h2>

              <p
                style={{
                  margin: "6px 0 0",
                  color: "#6b7280",
                  fontSize: "14px",
                }}
              >
                Production classification model
              </p>
            </div>
          </div>

          <div
            style={{
              padding: "8px 14px",
              borderRadius: "20px",
              background: "#ecfdf5",
              color: "#047857",
              fontSize: "13px",
              fontWeight: "600",
            }}
          >
            ● Production
          </div>
        </div>
      </div>

      {/* MODEL DETAILS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
          gap: "18px",
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            Algorithm
          </p>
          <h3 style={{ margin: "9px 0 0", color: "#111827" }}>
            XGBoost
          </h3>
        </div>

        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            Model File
          </p>
          <h3
            style={{
              margin: "9px 0 0",
              color: "#111827",
              fontSize: "15px",
            }}
          >
            xgboost_fraud_model_production.json
          </h3>
        </div>

        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            Features
          </p>
          <h3 style={{ margin: "9px 0 0", color: "#111827" }}>
            20
          </h3>
        </div>
      </div>

      {/* PERFORMANCE */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          padding: "24px",
          marginBottom: "24px",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
        }}
      >
        <h2
          style={{
            margin: "0 0 20px",
            fontSize: "20px",
            color: "#111827",
          }}
        >
          Model Performance
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
            gap: "18px",
          }}
        >
          <div>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
              ROC-AUC
            </p>
            <h3 style={{ margin: "8px 0 0", fontSize: "26px" }}>
              {latestData ? Number(latestData.roc_auc).toFixed(6) : "—"}
            </h3>
          </div>

          <div>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
              PR-AUC
            </p>
            <h3 style={{ margin: "8px 0 0", fontSize: "26px" }}>
              {latestData ? Number(latestData.pr_auc).toFixed(6) : "—"}
            </h3>
          </div>

          <div>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
              Precision
            </p>
            <h3 style={{ margin: "8px 0 0", fontSize: "26px" }}>
              {latestData ? Number(latestData.precision).toFixed(6) : "—"}
            </h3>
          </div>

          <div>
            <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
              Recall
            </p>
            <h3 style={{ margin: "8px 0 0", fontSize: "26px" }}>
              {latestData ? Number(latestData.recall).toFixed(6) : "—"}
            </h3>
          </div>
        </div>
      </div>

      {/* F1 SCORE */}
      <div
        style={{
          background: "linear-gradient(135deg, #111827, #1f2937)",
          borderRadius: "18px",
          padding: "26px",
          color: "white",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: "20px" }}>
            Production Model Status
          </h2>

          <p
            style={{
              margin: "7px 0 0",
              color: "#d1d5db",
              fontSize: "14px",
            }}
          >
            XGBoost model is loaded and available for fraud prediction.
          </p>
        </div>

        <div style={{ textAlign: "right" }}>
          <div
            style={{
              fontSize: "13px",
              color: "#d1d5db",
              marginBottom: "5px",
            }}
          >
            F1 Score
          </div>

          <div
            style={{
              fontSize: "30px",
              fontWeight: "700",
            }}
          >
            {latestData ? Number(latestData.f1_score).toFixed(6) : "—"}
          </div>
        </div>
      </div>

    </div>
  );
}
function Monitoring() {
  const [conceptData, setConceptData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dataDrift, setDataDrift] = useState(null);
  const [dataDriftAvailable, setDataDriftAvailable] = useState(false);

  useEffect(() => {
  // Concept Drift
  fetch(`${API_BASE_URL}/concept-drift`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load concept drift data");
      }

      return response.json();
    })
    .then((data) => {
      setConceptData(data);
      setLoading(false);
    })
    .catch((error) => {
      console.error("Concept drift error:", error);
      setLoading(false);
    });

  // Data Drift
  fetch(`${API_BASE_URL}/data-drift`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load data drift data");
      }

      return response.json();
    })
    .then((data) => {
      setDataDrift(data);
      setDataDriftAvailable(true);
    })
    .catch((error) => {
      console.error("Data drift error:", error);
      setDataDriftAvailable(false);
    });
}, []);

  const latest = conceptData[conceptData.length - 1];

  return (
    <div style={{ padding: "28px", maxWidth: "1400px", margin: "0 auto" }}>

      {/* HEADER */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "28px",
        }}
      >
        <div>
          <h1
            style={{
              margin: 0,
              fontSize: "30px",
              fontWeight: "700",
              color: "#111827",
            }}
          >
            Model Monitoring
          </h1>

          <p
            style={{
              marginTop: "8px",
              color: "#6b7280",
              fontSize: "15px",
            }}
          >
            Monitor data drift, concept drift, and model performance.
          </p>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "9px 15px",
            borderRadius: "20px",
            background: "#ecfdf5",
            color: "#047857",
            fontSize: "14px",
            fontWeight: "600",
          }}
        >
          <span
            style={{
              width: "9px",
              height: "9px",
              borderRadius: "50%",
              background: "#10b981",
            }}
          />
          API Online
        </div>
      </div>

      {/* MONITORING STATUS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: "20px",
          marginBottom: "28px",
        }}
      >

        {/* DATA DRIFT */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "22px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div>
              <p
                style={{
                  margin: 0,
                  color: "#6b7280",
                  fontSize: "13px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Data Drift
              </p>

              <h2
                style={{
                  margin: "8px 0 5px",
                  fontSize: "21px",
                  color: "#111827",
                }}
              >
                {dataDriftAvailable
  ? dataDrift?.dataset_drift
    ? "Dataset Drift Detected"
    : "Dataset Stable"
  : "Data Drift Unavailable"}
              </h2>

              <p
                style={{
                  margin: 0,
                  color: "#9ca3af",
                  fontSize: "14px",
                }}
              >
                Monitoring input feature changes
                {dataDriftAvailable && dataDrift && (
  <p
    style={{
      margin: "8px 0 0",
      color: "#6b7280",
      fontSize: "13px",
      fontWeight: "600",
    }}
  >
    {dataDrift.drifted_columns} of {dataDrift.total_columns} features drifted
  </p>
)}
              </p>
            </div>

            <div
              style={{
                width: "48px",
                height: "48px",
                borderRadius: "14px",
                background: "#eff6ff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "23px",
              }}
            >
              📊
            </div>
          </div>

          <div
            style={{
              marginTop: "20px",
              display: "inline-flex",
              alignItems: "center",
              gap: "7px",
              padding: "7px 12px",
              borderRadius: "20px",
              background: "#ecfdf5",
              color: "#047857",
              fontSize: "13px",
              fontWeight: "600",
            }}
          >
            ● Available
          </div>
        </div>

        {/* CONCEPT DRIFT */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "22px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div>
              <p
                style={{
                  margin: 0,
                  color: "#6b7280",
                  fontSize: "13px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Concept Drift
              </p>

              <h2
                style={{
                  margin: "8px 0 5px",
                  fontSize: "21px",
                  color: "#111827",
                }}
              >
                <h2
  style={{
    margin: "8px 0 5px",
    fontSize: "21px",
    color: "#111827",
  }}
>
  {conceptData.length > 0
    ? "Performance Monitored"
    : "Concept Drift Unavailable"}
</h2>
              </h2>

              <p
                style={{
                  margin: 0,
                  color: "#9ca3af",
                  fontSize: "14px",
                }}
              >
                Performance across datasets
              </p>
            </div>

            <div
              style={{
                width: "48px",
                height: "48px",
                borderRadius: "14px",
                background: "#f5f3ff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "23px",
              }}
            >
              📈
            </div>
          </div>

          <div
            style={{
              marginTop: "20px",
              display: "inline-flex",
              alignItems: "center",
              gap: "7px",
              padding: "7px 12px",
              borderRadius: "20px",
              background: "#ecfdf5",
              color: "#047857",
              fontSize: "13px",
              fontWeight: "600",
            }}
          >
            ● Available
          </div>
        </div>
      </div>

      {/* KPI CARDS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
          gap: "18px",
          marginBottom: "28px",
        }}
      >

        {/* FRAUD RATE */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            Latest Fraud Rate
          </p>

          <h2
            style={{
              margin: "10px 0 0",
              fontSize: "28px",
              color: "#111827",
            }}
          >
            {latest ? `${(Number(latest.fraud_rate) * 100).toFixed(3)}%` : "--"}
          </h2>

          <p style={{ margin: "7px 0 0", color: "#9ca3af", fontSize: "12px" }}>
            Test dataset
          </p>
        </div>

        {/* ROC AUC */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            ROC-AUC
          </p>

          <h2
            style={{
              margin: "10px 0 0",
              fontSize: "28px",
              color: "#111827",
            }}
          >
            {latest ? Number(latest.roc_auc).toFixed(6) : "--"}
          </h2>

          <p style={{ margin: "7px 0 0", color: "#9ca3af", fontSize: "12px" }}>
            Model discrimination
          </p>
        </div>

        {/* PR AUC */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            PR-AUC
          </p>

          <h2
            style={{
              margin: "10px 0 0",
              fontSize: "28px",
              color: "#111827",
            }}
          >
            {latest ? Number(latest.pr_auc).toFixed(6) : "--"}
          </h2>

          <p style={{ margin: "7px 0 0", color: "#9ca3af", fontSize: "12px" }}>
            Precision-recall quality
          </p>
        </div>

        {/* F1 */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "16px",
            padding: "20px",
            boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          }}
        >
          <p style={{ margin: 0, color: "#6b7280", fontSize: "13px" }}>
            F1 Score
          </p>

          <h2
            style={{
              margin: "10px 0 0",
              fontSize: "28px",
              color: "#111827",
            }}
          >
            {latest ? Number(latest.f1_score).toFixed(6) : "--"}
          </h2>

          <p style={{ margin: "7px 0 0", color: "#9ca3af", fontSize: "12px" }}>
            Classification balance
          </p>
        </div>
      </div>

      {/* PERFORMANCE TABLE */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e5e7eb",
          borderRadius: "16px",
          overflow: "hidden",
          boxShadow: "0 4px 15px rgba(0,0,0,0.04)",
          marginBottom: "28px",
        }}
      >
        <div
          style={{
            padding: "22px 24px",
            borderBottom: "1px solid #e5e7eb",
          }}
        >
          <h2
            style={{
              margin: 0,
              fontSize: "20px",
              color: "#111827",
            }}
          >
            Concept Drift Results
          </h2>

          <p
            style={{
              margin: "6px 0 0",
              color: "#9ca3af",
              fontSize: "14px",
            }}
          >
            Performance comparison across train, validation, and test datasets
          </p>
        </div>

        {loading ? (
          <div
            style={{
              padding: "50px",
              textAlign: "center",
              color: "#6b7280",
            }}
          >
            Loading monitoring data...
          </div>
        ) : conceptData.length === 0 ? (
          <div
            style={{
              padding: "50px",
              textAlign: "center",
              color: "#6b7280",
            }}
          >
            No monitoring data available.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "13px",
              }}
            >
              <thead>
                <tr style={{ background: "#f9fafb" }}>
                  {[
                    "Period",
                    "Rows",
                    "Fraud Count",
                    "Fraud Rate",
                    "ROC-AUC",
                    "PR-AUC",
                    "Precision",
                    "Recall",
                    "F1 Score",
                  ].map((heading) => (
                    <th
                      key={heading}
                      style={{
                        padding: "14px 16px",
                        textAlign: "left",
                        color: "#6b7280",
                        fontWeight: "600",
                        whiteSpace: "nowrap",
                        borderBottom: "1px solid #e5e7eb",
                      }}
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {conceptData.map((row) => (
                  <tr
                    key={row.period}
                    style={{
                      borderBottom: "1px solid #f3f4f6",
                    }}
                  >
                    <td style={{ padding: "15px 16px" }}>
                      <span
                        style={{
                          padding: "5px 10px",
                          borderRadius: "8px",
                          background:
                            row.period === "test"
                              ? "#eff6ff"
                              : "#f9fafb",
                          color:
                            row.period === "test"
                              ? "#2563eb"
                              : "#374151",
                          fontWeight: "600",
                          textTransform: "capitalize",
                        }}
                      >
                        {row.period}
                      </span>
                    </td>

                    <td style={{ padding: "15px 16px", color: "#374151" }}>
                      {row.rows.toLocaleString()}
                    </td>

                    <td style={{ padding: "15px 16px", color: "#374151" }}>
                      {row.fraud_count.toLocaleString()}
                    </td>

                    <td
                      style={{
                        padding: "15px 16px",
                        color: "#374151",
                        fontWeight: "600",
                      }}
                    >
                      {(row.fraud_rate * 100).toFixed(3)}%
                    </td>

                    <td style={{ padding: "15px 16px", color: "#111827" }}>
                      {row.roc_auc.toFixed(6)}
                    </td>

                    <td style={{ padding: "15px 16px", color: "#111827" }}>
                      {row.pr_auc.toFixed(6)}
                    </td>

                    <td style={{ padding: "15px 16px", color: "#111827" }}>
                      {row.precision.toFixed(6)}
                    </td>

                    <td style={{ padding: "15px 16px", color: "#111827" }}>
                      {row.recall.toFixed(6)}
                    </td>

                    <td
                      style={{
                        padding: "15px 16px",
                        color: "#047857",
                        fontWeight: "700",
                      }}
                    >
                      {row.f1_score.toFixed(6)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* DRIFT REPORT */}
      <div
        style={{
          background: "linear-gradient(135deg, #111827, #1f2937)",
          borderRadius: "18px",
          padding: "26px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "20px",
          color: "white",
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "20px",
            }}
          >
            Data Drift Report
          </h2>

          <p
            style={{
              margin: "7px 0 0",
              color: "#d1d5db",
              fontSize: "14px",
            }}
          >
            Evidently AI analysis of feature distribution changes.
          </p>
        </div>

        <a
          href={`${API_BASE_URL}/data_drift_report.html`}
          target="_blank"
          rel="noreferrer"
          style={{
            textDecoration: "none",
            background: "#ffffff",
            color: "#111827",
            padding: "11px 18px",
            borderRadius: "10px",
            fontSize: "14px",
            fontWeight: "600",
            whiteSpace: "nowrap",
          }}
        >
          Open Drift Report →
        </a>
      </div>

    </div>
  );
}
function Prediction() {
  const [formData, setFormData] = useState({
    step: 632,
    type: "TRANSFER",
    amount: 1000,
    oldbalanceOrg: 5000,
    newbalanceOrig: 4000,
    oldbalanceDest: 1000,
    newbalanceDest: 2000,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]:
        name === "type" ? value : Number(value),
    });
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        }
      );

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();

      setResult(data);
    } catch (error) {
      setResult({
        error:
          "Could not connect to FastAPI. Make sure the backend is running on port 8000.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="pageIntro">
        <div>
          <h2>Transaction Prediction</h2>
          <p>
            Enter transaction details and let the XGBoost production
            model determine whether the transaction is fraudulent.
          </p>
        </div>

        <div className="modelBadge light">
          <Brain size={18} />
          <div>
            <strong>XGBoost</strong>
            <span>Production Model</span>
          </div>
        </div>
      </div>

      <div className="predictionGrid">
        <div className="panel">
          <div className="panelHeader">
            <div>
              <h3>Transaction Details</h3>
              <p>Enter the financial transaction information</p>
            </div>

            <ShieldAlert size={22} />
          </div>

          <form onSubmit={handlePredict}>
            <div className="formGrid">
              <div className="formGroup">
                <label>Step</label>
                <input
                  type="number"
                  name="step"
                  value={formData.step}
                  onChange={handleChange}
                />
              </div>

              <div className="formGroup">
                <label>Transaction Type</label>

                <select
                  name="type"
                  value={formData.type}
                  onChange={handleChange}
                >
                  <option value="CASH_IN">CASH_IN</option>
                  <option value="CASH_OUT">CASH_OUT</option>
                  <option value="DEBIT">DEBIT</option>
                  <option value="PAYMENT">PAYMENT</option>
                  <option value="TRANSFER">TRANSFER</option>
                </select>
              </div>

              <div className="formGroup">
                <label>Amount</label>
                <input
                  type="number"
                  name="amount"
                  value={formData.amount}
                  onChange={handleChange}
                />
              </div>

              <div className="formGroup">
                <label>Old Balance (Origin)</label>
                <input
                  type="number"
                  name="oldbalanceOrg"
                  value={formData.oldbalanceOrg}
                  onChange={handleChange}
                />
              </div>

              <div className="formGroup">
                <label>New Balance (Origin)</label>
                <input
                  type="number"
                  name="newbalanceOrig"
                  value={formData.newbalanceOrig}
                  onChange={handleChange}
                />
              </div>

              <div className="formGroup">
                <label>Old Balance (Destination)</label>
                <input
                  type="number"
                  name="oldbalanceDest"
                  value={formData.oldbalanceDest}
                  onChange={handleChange}
                />
              </div>

              <div className="formGroup">
                <label>New Balance (Destination)</label>
                <input
                  type="number"
                  name="newbalanceDest"
                  value={formData.newbalanceDest}
                  onChange={handleChange}
                />
              </div>
            </div>

            <button
              className="predictButton"
              type="submit"
              disabled={loading}
            >
              {loading ? "Analyzing Transaction..." : "Predict Transaction"}
            </button>
          </form>
        </div>

        <div className="predictionResult">
          {!result && (
            <div className="emptyResult">
              <div className="emptyIcon">
                <ShieldAlert size={35} />
              </div>

              <h3>Prediction Result</h3>

              <p>
                Submit a transaction to see the fraud detection
                result.
              </p>
            </div>
          )}

          {result && !result.error && (
            <div
              className={`resultCard ${
                result.prediction === 1
                  ? "fraudResult"
                  : "legitimateResult"
              }`}
            >
              {result.prediction === 1 ? (
                <AlertTriangle size={42} />
              ) : (
                <CheckCircle size={42} />
              )}

              <div className="resultLabel">
                {result.result}
              </div>

              <div className="probability">
                {(result.fraud_probability * 100).toFixed(4)}%
              </div>

              <p>Fraud Probability</p>

              <div className="resultDetails">
                <div>
                  <span>Prediction</span>
                  <strong>{result.prediction}</strong>
                </div>

                <div>
                  <span>Threshold</span>
                  <strong>{result.threshold}</strong>
                </div>
              </div>
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
function Dashboard() {
  const [conceptData, setConceptData] = useState([]);
  const [apiOnline, setApiOnline] = useState(false);
  const [dataDriftAvailable, setDataDriftAvailable] = useState(false);
  const [driftedColumns, setDriftedColumns] = useState(0);
  const [totalColumns, setTotalColumns] = useState(20);
  const [datasetDrift, setDatasetDrift] = useState(false);
  const fraudData = conceptData.map((item) => ({
  day: item.period,
  transactions: item.rows,
  fraud: item.fraud_count,
}));

  useEffect(() => {
  // API Health
  fetch(`${API_BASE_URL}/health`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("API health check failed");
      }

      return response.json();
    })
    .then(() => {
      setApiOnline(true);
    })
    .catch((error) => {
      console.error("API health error:", error);
      setApiOnline(false);
    });

  // Concept Drift
  fetch(`${API_BASE_URL}/concept-drift`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load concept drift data");
      }

      return response.json();
    })
    .then((data) => {
      setConceptData(data);
    })
    .catch((error) => {
      console.error("Concept drift error:", error);
    });

  // Data Drift
  fetch(`${API_BASE_URL}/data-drift`)
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load data drift data");
      }

      return response.json();
    })
    .then((data) => {
      setDriftedColumns(data.drifted_columns);
      setTotalColumns(data.total_columns);
      setDatasetDrift(data.dataset_drift);
      setDataDriftAvailable(true);
    })
    .catch((error) => {
      console.error("Data drift error:", error);
      setDataDriftAvailable(false);
    });
}, []);

  const latestData =
    conceptData.length > 0
      ? conceptData[conceptData.length - 1]
      : null;

  return (
    <>
      <section className="welcome">
        <div>
          <div className="welcomeLabel">MLOPS OVERVIEW</div>
          <h2>Fraud Detection Dashboard</h2>
          <p>
            Monitor transactions, model performance, and production health
            from one place.
          </p>
        </div>

        <div className="modelBadge">
          <Brain size={18} />
          <div>
            <strong>XGBoost</strong>
            <span>Production Model</span>
          </div>
        </div>
      </section>

      <section className="statsGrid">
        <StatCard
          title="Model Accuracy"
          value={
  latestData
    ? `${(Number(latestData.roc_auc) * 100).toFixed(4)}%`
    : "—"
}
          subtitle="ROC-AUC"
          icon={<TrendingUp size={22} />}
          type="blue"
        />
        <StatCard
  title="Fraud Recall"
  value={
    latestData
      ? `${(Number(latestData.recall) * 100).toFixed(2)}%`
      : "—"
  }
  subtitle="Latest test result"
  icon={<ShieldAlert size={22} />}
  type="purple"
/>

        

        <StatCard
          title="API Status"
          value={apiOnline ? "Healthy" : "Offline"}
          subtitle="FastAPI backend"
          icon={<CheckCircle size={22} />}
          type="green"
        />

        <StatCard
          title="Monitoring"
          value={apiOnline && dataDriftAvailable ? "Active" : "Offline"}
          subtitle="Drift monitoring"
          icon={<Activity size={22} />}
          type="orange"
        />
      </section>

      <section className="dashboardGrid">
        <div className="panel chartPanel">
          <div className="panelHeader">
            <div>
              <h3>Dataset Activity</h3>
<p>Transaction and fraud distribution across datasets</p>
            </div>

            <div className="legend">
              <span>
                <i className="legendBlue"></i>
                Transactions
              </span>
              <span>
                <i className="legendPurple"></i>
                Fraud
              </span>
            </div>
          </div>

          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={fraudData}>
                <defs>
                  <linearGradient
                    id="transactionGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopOpacity={0.3} />
                    <stop offset="100%" stopOpacity={0.02} />
                  </linearGradient>

                  <linearGradient
                    id="fraudGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopOpacity={0.25} />
                    <stop offset="100%" stopOpacity={0.02} />
                  </linearGradient>
                </defs>

                <CartesianGrid strokeDasharray="3 3" vertical={false} />

                <XAxis
                  dataKey="day"
                  axisLine={false}
                  tickLine={false}
                />

                <YAxis
                  axisLine={false}
                  tickLine={false}
                />

                <Tooltip />

                <Area
                  type="monotone"
                  dataKey="transactions"
                  strokeWidth={2}
                  fill="url(#transactionGradient)"
                />

                <Area
                  type="monotone"
                  dataKey="fraud"
                  strokeWidth={2}
                  fill="url(#fraudGradient)"
                />
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
            {modelMetrics.map((metric) => (
              <div className="metricRow" key={metric.name}>
                <div className="metricInfo">
                  <span>{metric.name}</span>
                  <strong>{metric.value}%</strong>
                </div>

                <div className="progress">
                  <div
                    className="progressValue"
                    style={{ width: `${metric.value}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>

          <div className="aucBox">
            <div>
              <span>ROC-AUC</span>
              <strong>0.999999</strong>
            </div>

            <div>
              <span>PR-AUC</span>
              <strong>0.999878</strong>
            </div>
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
  status={
    dataDriftAvailable
      ? datasetDrift
        ? "Dataset Drift Detected"
        : "Dataset Stable"
      : "Offline"
  }
  detail={
    dataDriftAvailable
      ? `${driftedColumns} of ${totalColumns} features drifted`
      : "Report unavailable"
  }
/>

          <MonitoringItem
  icon={<BarChart3 size={18} />}
  title="Concept Drift"
  status={conceptData.length > 0 ? "Monitored" : "Offline"}
  detail={
    conceptData.length > 0
      ? "Performance comparison"
      : "No monitoring data"
  }
/>

          <MonitoringItem
            icon={<Server size={18} />}
            title="FastAPI"
            status={apiOnline ? "Healthy" : "Offline"}
            detail={API_BASE_URL}
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
            <DatasetStat
              label="Training Rows"
              value={
  conceptData.length > 0
    ? conceptData[0].rows.toLocaleString()
    : "—"
}
            />

            <DatasetStat
  label="Validation Rows"
  value={
    conceptData.length > 1
      ? conceptData[1].rows.toLocaleString()
      : "—"
  }
/>

            <DatasetStat
              label="Test Rows"
              value={latestData ? latestData.rows.toLocaleString() : "—"}
            />

            <DatasetStat
              label="Features"
              value="20"
            />
          </div>

          <div className="fraudRate">
            <div>
              <span>Test fraud rate</span>
              <strong>
  {latestData
    ? `${(Number(latestData.fraud_rate) * 100).toFixed(2)}%`
    : "—"}
</strong>
            </div>

            <AlertTriangle size={20} />
          </div>
        </div>
      </section>
    </>
  );
}

function StatCard({ title, value, subtitle, icon, type }) {
  return (
    <div className={`statCard ${type}`}>
      <div className="statTop">
        <span>{title}</span>
        <div className="statIcon">{icon}</div>
      </div>

      <h3>{value}</h3>
      <p>{subtitle}</p>
    </div>
  );
}

function MonitoringItem({ icon, title, status, detail }) {
  return (
    <div className="monitorItem">
      <div className="monitorIcon">{icon}</div>

      <div className="monitorText">
        <strong>{title}</strong>
        <span>{detail}</span>
      </div>

      <div className="monitorStatus">
        <span className="smallDot"></span>
        {status}
      </div>
    </div>
  );
}

function DatasetStat({ label, value }) {
  return (
    <div className="datasetStat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Placeholder({ icon, title, text }) {
  return (
    <div className="placeholder">
      <div className="placeholderIcon">{icon}</div>
      <h2>{title}</h2>
      <p>{text}</p>
      <span>Module ready for configuration</span>
    </div>
  );
}

export default App;
