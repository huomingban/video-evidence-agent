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
      throw new Error(`${fallbackMessage}超时，后端未能在限定时间内返回结果，请查看后端日志后重试。`);
    }
    if (error instanceof TypeError) {
      throw new Error("无法连接后端，请确认 9090 端口的服务正在运行。");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function request(path, options = {}, timeoutMs = 120000, fallbackMessage = "请求") {
  return fetchWithTimeout(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  }, timeoutMs, fallbackMessage);
}

export const api = {
  seedDemo(videoId) {
    return request("/api/demo/seed", {
      method: "POST",
      body: JSON.stringify({ video_id: videoId }),
    });
  },
  uploadVideo(videoId, file, background = true, onProgress) {
    const formData = new FormData();
    formData.append("video_id", videoId);
    formData.append("file", file);
    formData.append("background", String(background));
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/videos/upload`);
      xhr.timeout = 30 * 60 * 1000;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
      xhr.onload = () => {
        try {
          const contentType = xhr.getResponseHeader("content-type") || "";
          const body = contentType.includes("application/json") ? JSON.parse(xhr.responseText) : null;
          if (xhr.status < 200 || xhr.status >= 300) {
            throw new Error(body?.detail || `视频上传失败（HTTP ${xhr.status}）`);
          }
          resolve(body);
        } catch (error) {
          reject(error);
        }
      };
      xhr.onerror = () => reject(new Error("无法连接后端，请确认 9090 端口的服务正在运行。"));
      xhr.ontimeout = () => reject(new Error("视频上传超时，请检查文件大小和后端状态。"));
      xhr.send(formData);
    });
  },
  getTask(taskId) {
    return request(`/api/tasks/${encodeURIComponent(taskId)}`);
  },
  listVideos() {
    return request("/api/videos");
  },
  deleteVideo(videoId) {
    return request(`/api/videos/${encodeURIComponent(videoId)}`, {
      method: "DELETE",
    });
  },
  getVideoMemory(videoId) {
    return request(`/api/videos/${encodeURIComponent(videoId)}/memory`);
  },
  listReports(videoId) {
    return request(`/api/videos/${encodeURIComponent(videoId)}/reports`);
  },
  deleteReport(reportId) {
    return request(`/api/reports/${encodeURIComponent(reportId)}`, {
      method: "DELETE",
    });
  },
  listEvidence(videoId) {
    return request(`/api/evidence?video_id=${encodeURIComponent(videoId)}`);
  },
  ask(question, videoId, sessionId) {
    return request("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question, video_id: videoId, session_id: sessionId || undefined }),
    }, 240000, "报告生成请求");
  },
  createAnalysis(question, videoId, sessionId) {
    return request("/api/analysis", {
      method: "POST",
      body: JSON.stringify({ question, video_id: videoId, session_id: sessionId || undefined }),
    }, 30000, "Submit analysis task");
  },
  getAnalysisTask(taskId) {
    return request(`/api/analysis/${encodeURIComponent(taskId)}`, {}, 30000, "Query analysis task");
  },
};
