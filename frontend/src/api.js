const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:9090";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "请求失败");
  }
  return body;
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
    return fetch(`${import.meta.env.VITE_API_BASE || "http://127.0.0.1:9090"}/api/videos/upload`, {
      method: "POST",
      body: formData,
    }).then(async (response) => {
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.detail || "上传失败");
      }
      return body;
    });
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
