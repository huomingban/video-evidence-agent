const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:9090";

async function readResponse(response, fallbackMessage) {
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error(body?.detail || `${fallbackMessage}（HTTP ${response.status}）`);
  }
  return body;
}

async function fetchWithTimeout(url, options, timeoutMs, fallbackMessage) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return await readResponse(response, fallbackMessage);
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`${fallbackMessage}超时，请检查后端日志或缩短视频长度。`);
    }
    if (error instanceof TypeError) {
      throw new Error("无法连接后端，请确认 9090 端口的服务正在运行。");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function request(path, options = {}) {
  return fetchWithTimeout(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  }, 120000, "请求");
}

export const api = {
  seedDemo(videoId) {
    return request("/api/demo/seed", {
      method: "POST",
      body: JSON.stringify({ video_id: videoId }),
    });
  },
  uploadVideo(videoId, file) {
    const formData = new FormData();
    formData.append("video_id", videoId);
    formData.append("file", file);
    return fetchWithTimeout(`${API_BASE}/api/videos/upload`, {
      method: "POST",
      body: formData,
    }, 30 * 60 * 1000, "视频上传和转录");
  },
  listEvidence(videoId) {
    return request(`/api/evidence?video_id=${encodeURIComponent(videoId)}`);
  },
  ask(question, videoId) {
    return request("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question, video_id: videoId }),
    });
  },
};
