const API_ROOT = import.meta.env.VITE_API_URL ?? "";

async function request(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, options);

  if (!response.ok) {
    let message = "Không thể kết nối tới máy chủ.";

    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Keep the generic error when the server does not return JSON.
    }

    throw new Error(message);
  }

  return response.json();
}

export const api = {
  health: () => request("/health"),
  analyze: (payload) =>
    request("/api/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
