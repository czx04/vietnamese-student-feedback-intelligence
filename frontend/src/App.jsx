import { useEffect, useState } from "react";
import { api } from "./api";

const SENTIMENT_LABELS = {
  positive: "Tích cực",
  neutral: "Trung lập",
  negative: "Tiêu cực",
};

const TOPIC_LABELS = {
  lecturer: "Giảng viên",
  program: "Chương trình",
  facility: "Cơ sở vật chất",
  others: "Khác",
};

function ResultItem({ label, value, confidence }) {
  return (
    <div className="result-item">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>Độ tin cậy: {(confidence * 100).toFixed(1)}%</small>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(undefined);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    const feedback = text.trim();

    if (!feedback) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      setResult(await api.analyze({ text: feedback }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const statusText =
    health === undefined
      ? "Đang kiểm tra"
      : health
        ? "Sẵn sàng"
        : "Mất kết nối";

  return (
    <main className="page">
      <header className="header">
        <div>
          <p className="eyebrow">Vietnamese Student Feedback</p>
          <h1>Phân tích phản hồi sinh viên</h1>
          <p className="description">
            Nhập một phản hồi để xác định cảm xúc và chủ đề.
          </p>
        </div>
        <span
          className={`status ${health ? "online" : health === null ? "offline" : ""}`}
        >
          {statusText}
        </span>
      </header>

      <section className="card">
        <form onSubmit={handleSubmit}>
          <label htmlFor="feedback">Nội dung phản hồi</label>
          <textarea
            id="feedback"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Ví dụ: Giảng viên giảng bài dễ hiểu."
            maxLength={10000}
            rows={7}
            autoFocus
          />

          <div className="form-footer">
            <small>{text.length}/10.000 ký tự</small>
            <button type="submit" disabled={loading || !text.trim()}>
              {loading ? "Đang phân tích..." : "Phân tích"}
            </button>
          </div>
        </form>

        {error && <p className="message error">{error}</p>}
      </section>

      {result && (
        <section className="card result" aria-live="polite">
          <h2>Kết quả</h2>
          <div className="result-grid">
            <ResultItem
              label="Cảm xúc"
              value={SENTIMENT_LABELS[result.sentiment.label] ?? result.sentiment.label}
              confidence={result.sentiment.confidence}
            />
            <ResultItem
              label="Chủ đề"
              value={TOPIC_LABELS[result.topic.label] ?? result.topic.label}
              confidence={result.topic.confidence}
            />
          </div>
          <p className="note">
            Độ tin cậy là điểm tham khảo của mô hình, chưa phải xác suất đã hiệu chỉnh.
          </p>
        </section>
      )}
    </main>
  );
}
