const detectForm = document.getElementById("detect-form");
const searchForm = document.getElementById("search-form");
const detectStatus = document.getElementById("detect-status");
const searchStatus = document.getElementById("search-status");
const detectResult = document.getElementById("detect-result");
const searchResult = document.getElementById("search-result");

function setStatus(target, message, type = "") {
  target.textContent = message;
  target.className = `status ${type}`.trim();
}

function createImageCard(title, imageSrc, extraHtml = "") {
  return `
    <section class="image-card">
      <img src="${imageSrc}" alt="${title}" />
      <div class="content">
        <strong>${title}</strong>
        ${extraHtml}
      </div>
    </section>
  `;
}

function renderDetections(data) {
  const details = data.detections
    .map(
      (item) => `
        <li>人脸 ${item.index}：框 [${item.bbox.join(", ")}]，置信度 ${item.confidence}</li>
      `
    )
    .join("");

  const faces = data.cropped_faces
    .map(
      (src, index) => `
        <section class="face-card">
          <img src="${src}" alt="face-${index + 1}" />
          <div class="content">
            <span class="tag">裁剪人脸 ${index + 1}</span>
          </div>
        </section>
      `
    )
    .join("");

  detectResult.innerHTML = `
    ${createImageCard("检测结果预览", data.annotated_image)}
    <section class="meta-card">
      <div class="content">
        <strong>检测信息</strong>
        <ul class="metrics">
          <li>检测到人脸数量：${data.face_count}</li>
          ${details || "<li>未检测到人脸</li>"}
        </ul>
      </div>
    </section>
    <div class="face-list">${faces}</div>
  `;
}

function renderMatches(data) {
  const matchHtml = data.matches.length
    ? data.matches
        .map(
          (item, index) => `
            <section class="match-card">
              <img src="${item.stored_path}" alt="${item.filename}" />
              <div class="content">
                <strong>Top ${index + 1}: ${item.filename}</strong>
                <ul class="metrics">
                  <li>距离：${item.distance}</li>
                  <li>相似度分数：${item.similarity_score}</li>
                  <li>入库时间：${item.created_at}</li>
                  <li>库中图片人脸数量：${item.face_count}</li>
                  <li>主人脸置信度：${item.face_confidence}</li>
                </ul>
              </div>
            </section>
          `
        )
        .join("")
    : `
      <section class="meta-card">
        <div class="content">
          <strong>检索结果</strong>
          <p>数据库里暂时没有可比对的其它图片。</p>
        </div>
      </section>
    `;

  searchResult.innerHTML = `
    ${createImageCard("检索图片预览", data.detection.annotated_image)}
    <section class="meta-card">
      <div class="content">
        <strong>入库信息</strong>
        <ul class="metrics">
          <li>文件名：${data.query_filename}</li>
          <li>是否因重名跳过入库：${data.duplicate_skipped ? "是" : "否"}</li>
          <li>检测到人脸数量：${data.detection.face_count}</li>
          <li>用于比对的人脸序号：${data.detection.primary_face_index}</li>
          <li>用于比对的人脸置信度：${data.detection.primary_confidence}</li>
        </ul>
      </div>
    </section>
    <div class="match-list">${matchHtml}</div>
  `;
}

detectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  detectResult.innerHTML = "";
  setStatus(detectStatus, "正在进行人脸检测...");

  const file = document.getElementById("detect-file").files[0];
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "检测失败");
    }

    renderDetections(data);
    setStatus(detectStatus, "检测完成。", "success");
  } catch (error) {
    setStatus(detectStatus, error.message, "error");
  }
});

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  searchResult.innerHTML = "";
  setStatus(searchStatus, "正在入库并执行相似度检索...");

  const file = document.getElementById("search-file").files[0];
  const topK = document.getElementById("top-k").value || 5;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", topK);

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "检索失败");
    }

    renderMatches(data);
    setStatus(searchStatus, "检索完成。", "success");
  } catch (error) {
    setStatus(searchStatus, error.message, "error");
  }
});
