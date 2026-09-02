<script setup>
import { onMounted, ref } from "vue";
import { api } from "./api";

const videoId = ref("demo-video");
const question = ref("");
const evidence = ref([]);
const videos = ref([]);
const sessionId = ref(null);
const conversation = ref([]);
const result = ref(null);
const loading = ref(false);
const loadingMessage = ref("");
const message = ref("");
const selectedFile = ref(null);

function handleFileChange(event) {
  const file = event.target.files?.[0] || null;
  selectedFile.value = file;
  if (file && videoId.value === "demo-video") {
    videoId.value = file.name
      .replace(/\.[^/.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 180) || "uploaded-video";
    loadEvidence();
  }
}

async function loadEvidence() {
  try {
    evidence.value = (await api.listEvidence(videoId.value)).items;
  } catch (error) {
    message.value = error.message;
  }
}

async function loadVideos() {
  try {
    videos.value = (await api.listVideos()).items;
  } catch (error) {
    message.value = error.message;
  }
}

async function loadMemory() {
  try {
    const memory = await api.getVideoMemory(videoId.value);
    const session = memory.sessions?.[0];
    sessionId.value = session?.session_id || null;
    conversation.value = session?.messages || [];
  } catch (error) {
    message.value = error.message;
  }
}

async function selectVideo(video) {
  videoId.value = video.video_id;
  selectedFile.value = null;
  result.value = null;
  await loadEvidence();
  await loadMemory();
}

async function removeVideo(video) {
  if (!window.confirm(`Delete ${video.filename}? This removes the video and its evidence.`)) return;
  loading.value = true;
  loadingMessage.value = "Deleting video and evidence...";
  message.value = "";
  try {
    await api.deleteVideo(video.video_id);
    await loadVideos();
    if (videoId.value === video.video_id) {
      videoId.value = "demo-video";
      evidence.value = [];
      result.value = null;
      await loadEvidence();
      sessionId.value = null;
      conversation.value = [];
    }
    message.value = "Video and its evidence were deleted.";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function seedDemo() {
  loading.value = true;
  loadingMessage.value = "正在准备演示证据...";
  message.value = "";
  try {
    await api.seedDemo(videoId.value);
    await loadEvidence();
    result.value = null;
    message.value = "演示证据已准备好，可以开始提问。";
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function uploadVideo() {
  if (!selectedFile.value) {
    message.value = "请先选择一个视频文件。";
    return;
  }

  loading.value = true;
  loadingMessage.value = "正在上传并转录视频，首次加载语音模型可能需要较长时间...";
  message.value = "";
  try {
    const payload = await api.uploadVideo(videoId.value, selectedFile.value);
    await loadEvidence();
    await loadVideos();
    result.value = null;
    const sourceSummary = [
      `ASR ${payload.asr_count ?? 0} 条`,
      `OCR ${payload.ocr_count ?? 0} 条`,
    ].join("，");
    const ocrNotice = payload.ocr_error ? ` OCR 未完成：${payload.ocr_error}` : "";
    message.value = `已上传 ${payload.filename}，${sourceSummary}证据已写入。${ocrNotice}`;
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

async function ask() {
  if (!question.value.trim()) return;
  loading.value = true;
  loadingMessage.value = "正在检索证据并请求 Kimi...";
  message.value = "";
  try {
    result.value = await api.ask(question.value.trim(), videoId.value, sessionId.value);
    sessionId.value = result.value.session_id || sessionId.value;
    await loadMemory();
  } catch (error) {
    message.value = error.message;
  } finally {
    loading.value = false;
    loadingMessage.value = "";
  }
}

function formatSeconds(seconds) {
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const rest = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

onMounted(async () => {
  await loadVideos();
  await loadEvidence();
  await loadMemory();
});
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">TRACEABLE VIDEO QA</p>
        <h1>VideoEvidence Agent</h1>
      </div>
      <div class="header-actions">
        <label class="upload-button">
          <input type="file" accept="video/*" @change="handleFileChange" />
          选择视频
        </label>
        <button class="secondary-button" :disabled="loading || !selectedFile" @click="uploadVideo">
          上传并转录
        </button>
        <button class="secondary-button" :disabled="loading" @click="seedDemo">
          准备演示数据
        </button>
      </div>
    </header>

    <section class="workspace">
      <aside class="evidence-panel">
        <div class="panel-heading">
          <div>
            <span class="section-label">证据时间轴</span>
            <h2>{{ evidence.length }} 条证据</h2>
          </div>
          <span
            :class="['status-dot', videoId === 'demo-video' ? 'is-demo' : 'is-uploaded']"
            :title="videoId === 'demo-video' ? '当前为演示数据' : '当前为上传视频数据'"
          ></span>
        </div>

        <label class="field-label" for="video-id">视频标识</label>
        <input id="video-id" v-model="videoId" class="text-input" @change="loadEvidence" />

        <div v-if="videos.length" class="video-library">
          <div class="library-heading">Saved videos</div>
          <div v-for="video in videos" :key="video.video_id" class="video-row">
            <button class="video-select" :class="{ 'is-active': video.video_id === videoId }" @click="selectVideo(video)">
              <strong>{{ video.filename }}</strong>
              <span>{{ video.evidence_count }} evidence items</span>
            </button>
            <button class="delete-button" :disabled="loading" title="Delete video" @click="removeVideo(video)">×</button>
          </div>
        </div>

        <div v-if="evidence.length" class="evidence-list">
          <article v-for="item in evidence" :key="item.id" class="evidence-item">
            <div class="timestamp">
              <span class="evidence-source">{{ item.source || "ASR" }}</span>
              {{ formatSeconds(item.start_seconds) }} - {{ formatSeconds(item.end_seconds) }}
            </div>
            <p>{{ item.text }}</p>
          </article>
        </div>
        <p v-else class="empty-state">还没有证据。请先选择视频并上传转录，或准备演示数据。</p>
      </aside>

      <section class="answer-panel">
        <div class="panel-heading">
          <div>
            <span class="section-label">证据问答</span>
            <h2>只根据已检索证据回答</h2>
          </div>
          <span
            v-if="result"
            :class="['grounding-badge', result.grounded ? 'is-grounded' : 'is-refused']"
          >
            {{ result.grounded ? "有证据支持" : "证据不足" }}
          </span>
        </div>

        <form class="ask-form" @submit.prevent="ask">
          <textarea
            v-model="question"
            class="question-input"
            rows="4"
            placeholder="例如：这个项目应该如何准备面试？"
          ></textarea>
          <button class="primary-button" :disabled="loading || !question.trim()">
            {{ loading ? "检索中..." : "检索并回答" }}
          </button>
        </form>

        <p v-if="loadingMessage" class="notice progress-notice">{{ loadingMessage }}</p>
        <p v-if="message" class="notice">{{ message }}</p>

        <div v-if="conversation.length" class="conversation-section">
          <div class="conversation-heading">Session memory</div>
          <div v-for="(item, index) in conversation" :key="`${item.created_at}-${index}`" class="conversation-item">
            <span>{{ item.role === "user" ? "You" : "Agent" }}</span>
            <p>{{ item.content }}</p>
          </div>
        </div>

        <div v-if="result" class="result">
          <p class="answer">{{ result.answer }}</p>
          
          <div v-if="result.trace && result.trace.length" class="trace-section">
            <details open>
              <summary>Agent 思考过程 ({{ result.trace.length }} 步)</summary>
              <ol class="trace-list">
                <li v-for="(step, index) in result.trace" :key="index" class="trace-step">
                  {{ step }}
                </li>
              </ol>
            </details>
          </div>
          
          <div v-if="result.citations.length" class="citation-list">
            <strong>关键引用：</strong>
            <div v-for="citation in result.citations" :key="citation.evidence_id" class="citation">
              <span class="citation-time">{{ citation.timestamp }}</span>
              <span>{{ citation.text }}</span>
            </div>
          </div>
        </div>
        <div v-else class="result-placeholder">
          <strong>回答会显示在这里</strong>
          <span>每条结论都应该能够回到具体时间段。</span>
        </div>
      </section>
    </section>
  </main>
</template>
