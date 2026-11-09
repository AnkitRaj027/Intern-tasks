// Auspify Netflix ML Studio - Client-Side Controller

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initStatusCheck();
  initTask1Recommender();
  initTask2Classifier();
  initTask3RatingPredictor();
  initTask4Clustering();
  initEvaluations();
});

/* ===================================================
 * 1. TAB NAVIGATION
 * =================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(pane => pane.classList.remove("active"));

      btn.classList.add("active");
      const activePane = document.getElementById(targetTab);
      if (activePane) {
        activePane.classList.add("active");
      }
    });
  });
}

/* ===================================================
 * 2. SYSTEM STATUS CHECK
 * =================================================== */
async function initStatusCheck() {
  const text = document.getElementById("statusText");

  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      text.textContent = `Models Online • ${data.total_titles.toLocaleString()} Titles Indexed`;
    }
  } catch (err) {
    text.textContent = "Offline / Standalone Mode";
  }
}

/* ===================================================
 * 3. TASK 1: RECOMMENDATION ENGINE
 * =================================================== */
function initTask1Recommender() {
  const input = document.getElementById("recSearchInput");
  const btn = document.getElementById("btnGetRecommendations");
  const dropdown = document.getElementById("searchSuggestions");
  const resultsContainer = document.getElementById("recommendationsGrid");
  const resultsHeader = document.getElementById("recResultsHeader");
  const queryDetails = document.getElementById("queryTitleDetails");
  const queryName = document.getElementById("queryName");
  const queryMeta = document.getElementById("queryMeta");

  // Autocomplete search debounce
  let debounceTimer;
  input.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    const val = e.target.value.trim();

    if (val.length < 2) {
      dropdown.style.display = "none";
      return;
    }

    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/titles?q=${encodeURIComponent(val)}&limit=8`);
        if (res.ok) {
          const suggestions = await res.json();
          renderSuggestions(suggestions);
        }
      } catch (err) {
        console.error("Autocomplete error:", err);
      }
    }, 200);
  });

  function renderSuggestions(list) {
    if (!list || list.length === 0) {
      dropdown.style.display = "none";
      return;
    }

    dropdown.innerHTML = list.map(item => `
      <div class="dropdown-item" data-title="${escapeHtml(item.title)}">
        <span class="item-main">${escapeHtml(item.title)}</span>
        <span class="item-tag">${item.type} • ${item.release_year}</span>
      </div>
    `).join("");

    dropdown.style.display = "block";

    dropdown.querySelectorAll(".dropdown-item").forEach(el => {
      el.addEventListener("click", () => {
        const title = el.getAttribute("data-title");
        input.value = title;
        dropdown.style.display = "none";
        fetchRecommendations(title);
      });
    });
  }

  // Hide dropdown on outside click
  document.addEventListener("click", (e) => {
    if (!dropdown.contains(e.target) && e.target !== input) {
      dropdown.style.display = "none";
    }
  });

  // Quick try pill buttons
  document.querySelectorAll(".pill-btn[data-title]").forEach(btn => {
    btn.addEventListener("click", () => {
      const title = btn.getAttribute("data-title");
      input.value = title;
      dropdown.style.display = "none";
      fetchRecommendations(title);
    });
  });

  btn.addEventListener("click", () => {
    const title = input.value.trim();
    if (title) fetchRecommendations(title);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const title = input.value.trim();
      if (title) {
        dropdown.style.display = "none";
        fetchRecommendations(title);
      }
    }
  });

  async function fetchRecommendations(title) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-sm"></span> <span>Searching...</span>`;

    try {
      const res = await fetch("/api/task1/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, top_n: 10 })
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert(errorData.error || "Title not found.");
        return;
      }

      const data = await res.json();

      // Show query info
      queryDetails.style.display = "block";
      queryName.textContent = data.query_title;
      queryMeta.textContent = `Format: ${data.query_type}  |  Categories: ${data.query_genres || "N/A"}`;

      // Render recommendations
      resultsHeader.style.display = "block";
      resultsContainer.innerHTML = data.recommendations.map(rec => {
        const pct = Math.round(rec.similarity_score * 100);
        return `
          <div class="recom-card">
            <div>
              <div class="recom-top">
                <div class="recom-title">${escapeHtml(rec.title)}</div>
                <div class="match-badge">${pct}% match</div>
              </div>
              <div class="recom-meta-row">
                <span>${rec.type}</span>
                <span>•</span>
                <span>${rec.release_year}</span>
                <span>•</span>
                <span>${rec.rating || "Unrated"}</span>
              </div>
              <div class="recom-genres-text">
                ${escapeHtml(rec.listed_in || "N/A")}
              </div>
            </div>
            ${rec.director && rec.director.toLowerCase() !== "not given" && rec.director.toLowerCase() !== "unknown" ? `
              <div class="recom-director-text">
                Dir: ${escapeHtml(rec.director)}
              </div>
            ` : ""}
          </div>
        `;
      }).join("");

    } catch (err) {
      alert("Error contacting recommendation server: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>Generate Recommendations</span>`;
    }
  }
}

/* ===================================================
 * 4. TASK 2: CONTENT TYPE CLASSIFIER
 * =================================================== */
function initTask2Classifier() {
  const form = document.getElementById("formTask2");
  const resultBox = document.getElementById("t2ResultBox");
  const badge = document.getElementById("t2Badge");
  const expl = document.getElementById("t2Explanation");
  const btn = document.getElementById("btnPredictType");

  // Presets
  document.getElementById("presetNolanMovie").addEventListener("click", () => {
    document.getElementById("t2Director").value = "Christopher Nolan";
    document.getElementById("t2Country").value = "United States";
    document.getElementById("t2Year").value = "2023";
    document.getElementById("t2Rating").value = "PG-13";
    document.getElementById("t2Genres").value = "Action & Adventure, Sci-Fi";
  });

  document.getElementById("presetCrimeSeries").addEventListener("click", () => {
    document.getElementById("t2Director").value = "unknown";
    document.getElementById("t2Country").value = "United States";
    document.getElementById("t2Year").value = "2022";
    document.getElementById("t2Rating").value = "TV-MA";
    document.getElementById("t2Genres").value = "Crime TV Shows, TV Dramas, TV Thrillers";
  });

  document.getElementById("presetDocuFilm").addEventListener("click", () => {
    document.getElementById("t2Director").value = "Kirsten Johnson";
    document.getElementById("t2Country").value = "United States";
    document.getElementById("t2Year").value = "2020";
    document.getElementById("t2Rating").value = "PG-13";
    document.getElementById("t2Genres").value = "Documentaries";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      director: document.getElementById("t2Director").value.trim(),
      country: document.getElementById("t2Country").value.trim(),
      release_year: parseInt(document.getElementById("t2Year").value) || 2023,
      rating: document.getElementById("t2Rating").value,
      listed_in: document.getElementById("t2Genres").value.trim()
    };

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-sm"></span> <span>Classifying...</span>`;

    try {
      const res = await fetch("/api/task2/predict-type", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      resultBox.style.display = "block";
      badge.textContent = data.predicted_type.toUpperCase();

      if (data.predicted_type === "Movie") {
        badge.className = "result-badge-display result-badge-movie";
        expl.textContent = `The Linear SVM pipeline identifies these metadata tokens as characteristic of Feature Films based on genre profiles and release parameters.`;
      } else {
        badge.className = "result-badge-display result-badge-tv";
        expl.textContent = `The Linear SVM pipeline identifies these metadata tokens as characteristic of an Episodic Television Series.`;
      }

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>Run Classification</span>`;
    }
  });
}

/* ===================================================
 * 5. TASK 3: AUDIENCE RATING PREDICTOR
 * =================================================== */
function initTask3RatingPredictor() {
  const form = document.getElementById("formTask3");
  const resultBox = document.getElementById("t3ResultBox");
  const badge = document.getElementById("t3Badge");
  const confText = document.getElementById("t3ConfidencePct");
  const confFill = document.getElementById("t3ConfidenceFill");
  const probsBox = document.getElementById("t3ProbabilitiesBreakdown");
  const btn = document.getElementById("btnPredictRating");

  // Presets
  document.getElementById("presetT3Kids").addEventListener("click", () => {
    document.getElementById("t3Type").value = "TV Show";
    document.getElementById("t3Director").value = "unknown";
    document.getElementById("t3Country").value = "United States";
    document.getElementById("t3Year").value = "2021";
    document.getElementById("t3Duration").value = "3 Seasons";
    document.getElementById("t3Genres").value = "Kids' TV, TV Comedies";
  });

  document.getElementById("presetT3Mature").addEventListener("click", () => {
    document.getElementById("t3Type").value = "Movie";
    document.getElementById("t3Director").value = "Martin Scorsese";
    document.getElementById("t3Country").value = "United States";
    document.getElementById("t3Year").value = "2019";
    document.getElementById("t3Duration").value = "209 min";
    document.getElementById("t3Genres").value = "Dramas, Crime Movies";
  });

  document.getElementById("presetT3Teen").addEventListener("click", () => {
    document.getElementById("t3Type").value = "TV Show";
    document.getElementById("t3Director").value = "unknown";
    document.getElementById("t3Country").value = "United States";
    document.getElementById("t3Year").value = "2018";
    document.getElementById("t3Duration").value = "4 Seasons";
    document.getElementById("t3Genres").value = "TV Dramas, Teen TV Shows, Romantic TV Shows";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      type: document.getElementById("t3Type").value,
      director: document.getElementById("t3Director").value.trim(),
      country: document.getElementById("t3Country").value.trim(),
      release_year: parseInt(document.getElementById("t3Year").value) || 2023,
      duration: document.getElementById("t3Duration").value.trim(),
      listed_in: document.getElementById("t3Genres").value.trim()
    };

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-sm"></span> <span>Estimating...</span>`;

    try {
      const res = await fetch("/api/task3/predict-rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      resultBox.style.display = "block";
      badge.textContent = data.predicted_rating;

      const conf = data.confidence || 50;
      confText.textContent = `${conf}%`;
      confFill.style.width = `${conf}%`;

      if (data.top_probabilities && data.top_probabilities.length > 0) {
        probsBox.innerHTML = data.top_probabilities.map(p => `
          <div class="prob-chip">
            <span style="color: var(--text-muted);">${escapeHtml(p.rating)}:</span>
            <span class="prob-chip-value">${p.probability}%</span>
          </div>
        `).join("");
      } else {
        probsBox.innerHTML = "";
      }

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>Estimate Maturity Rating</span>`;
    }
  });
}

/* ===================================================
 * 6. TASK 4: CONTENT SEGMENTATION CLUSTERING
 * =================================================== */
function initTask4Clustering() {
  const form = document.getElementById("formTask4");
  const resultBox = document.getElementById("t4ClusterResultBox");
  const nameDisplay = document.getElementById("t4AssignedName");
  const btn = document.getElementById("btnAssignCluster");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      type: document.getElementById("t4Type").value,
      country: "United States",
      rating: "TV-MA",
      release_year: parseInt(document.getElementById("t4Year").value) || 2020,
      listed_in: document.getElementById("t4Genres").value.trim(),
      duration: document.getElementById("t4Duration").value.trim()
    };

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-sm"></span> <span>Assigning...</span>`;

    try {
      const res = await fetch("/api/task4/assign-cluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Classification failed");

      resultBox.style.display = "block";
      nameDisplay.textContent = data.cluster_name;

    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span>Assign Cluster</span>`;
    }
  });
}

/* Security Helper */
function escapeHtml(text) {
  if (!text) return "";
  return text.toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* ===================================================
 * 7. MODEL EVALUATION & BENCHMARKS (TASKS 1-4)
 * =================================================== */
async function initEvaluations() {
  const container = document.getElementById("evalSectionsContainer");
  const filterChips = document.querySelectorAll(".eval-filter-chip");
  const sortSelect = document.getElementById("evalSortSelect");
  const kpiCards = document.querySelectorAll(".eval-kpi-card");

  if (!container) return;

  let evalData = null;
  let activeFilter = "all";
  let activeSort = "pipeline";
  const rowSorts = {
    task1: "rank-asc",
    task2: "rank-asc",
    task3: "rank-asc",
    task4: "rank-asc"
  };

  const performanceRankOrder = {
    task2: 1, // Linear SVM 100%
    task1: 2, // Weighted TF-IDF 93.5%
    task4: 3, // K-Means Silhouette 0.3015
    task3: 4  // Random Forest 50.63%
  };

  try {
    const res = await fetch("/api/evaluations");
    if (!res.ok) throw new Error("Could not load evaluations data");
    evalData = await res.json();
    renderAll();
  } catch (err) {
    container.innerHTML = `
      <div style="padding: 2rem; text-align: center; color: var(--brand-accent);">
        Failed to load evaluation benchmarks: ${escapeHtml(err.message)}
      </div>
    `;
    return;
  }

  // Filter chips click handler
  filterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      filterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      activeFilter = chip.getAttribute("data-task");
      renderAll();
    });
  });

  // Sort dropdown change handler
  if (sortSelect) {
    sortSelect.addEventListener("change", (e) => {
      activeSort = e.target.value;
      renderAll();
    });
  }

  // KPI cards click handler (quick filter + scroll)
  kpiCards.forEach(card => {
    card.addEventListener("click", () => {
      const target = card.getAttribute("data-filter-target");
      if (target) {
        activeFilter = target;
        filterChips.forEach(c => {
          if (c.getAttribute("data-task") === target) {
            c.classList.add("active");
          } else {
            c.classList.remove("active");
          }
        });
        renderAll();
        container.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  function getSortedTaskKeys() {
    let keys = Object.keys(evalData);

    if (activeFilter !== "all") {
      keys = keys.filter(k => k === activeFilter);
    }

    if (activeSort === "pipeline") {
      const order = ["task1", "task2", "task3", "task4"];
      keys.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    } else if (activeSort === "performance") {
      keys.sort((a, b) => (performanceRankOrder[a] || 99) - (performanceRankOrder[b] || 99));
    } else if (activeSort === "alphabetical") {
      keys.sort((a, b) => {
        const titleA = evalData[a].title.toLowerCase();
        const titleB = evalData[b].title.toLowerCase();
        return titleA.localeCompare(titleB);
      });
    }

    return keys;
  }

  function renderAll() {
    if (!evalData) return;

    const taskKeys = getSortedTaskKeys();

    if (taskKeys.length === 0) {
      container.innerHTML = `<div style="text-align: center; padding: 2rem; color: var(--text-muted);">No evaluation data matching the selected filter.</div>`;
      return;
    }

    container.innerHTML = taskKeys.map(key => renderTaskBlock(key, evalData[key])).join("");

    // Attach row sorting event listeners
    attachRowSortListeners();
  }

  function attachRowSortListeners() {
    container.querySelectorAll(".eval-row-sort-select").forEach(sel => {
      sel.addEventListener("change", (e) => {
        const taskKey = sel.getAttribute("data-task-key");
        rowSorts[taskKey] = e.target.value;
        const block = container.querySelector(`.eval-task-block[data-task-id="${taskKey}"]`);
        if (block) {
          const tableWrapper = block.querySelector(".eval-table-wrapper");
          if (tableWrapper) {
            tableWrapper.outerHTML = renderTableWrapper(taskKey, evalData[taskKey]);
            attachRowSortListeners();
          }
        }
      });
    });
  }

  function getSortedRows(taskKey, rows) {
    const sortVal = rowSorts[taskKey] || "rank-asc";
    const cloned = [...rows];

    if (sortVal === "rank-asc") {
      cloned.sort((a, b) => (a.rank || 0) - (b.rank || 0));
    } else if (sortVal === "rank-desc") {
      cloned.sort((a, b) => (b.rank || 0) - (a.rank || 0));
    } else if (sortVal === "name-asc") {
      cloned.sort((a, b) => a.model.localeCompare(b.model));
    }
    return cloned;
  }

  function renderTableWrapper(taskKey, task) {
    const currentSort = rowSorts[taskKey] || "rank-asc";
    const sortedRows = getSortedRows(taskKey, task.rows || []);

    return `
      <div class="eval-table-wrapper">
        <div class="eval-table-header-row">
          <span class="eval-table-title">Performance Comparison</span>
          <div class="eval-row-sort">
            <label for="sort_${taskKey}">Sort rows:</label>
            <select id="sort_${taskKey}" class="eval-row-sort-select" data-task-key="${taskKey}">
              <option value="rank-asc" ${currentSort === "rank-asc" ? "selected" : ""}>Rank: Best First</option>
              <option value="rank-desc" ${currentSort === "rank-desc" ? "selected" : ""}>Rank: Lowest First</option>
              <option value="name-asc" ${currentSort === "name-asc" ? "selected" : ""}>Model Name (A → Z)</option>
            </select>
          </div>
        </div>

        <div class="eval-table-scroll">
          <table class="eval-data-table">
            <thead>
              <tr>
                <th style="width: 80px;">Rank</th>
                ${(task.columns || []).map(col => `<th>${escapeHtml(col)}</th>`).join("")}
              </tr>
            </thead>
            <tbody>
              ${sortedRows.map(r => `
                <tr class="${r.is_best ? "row-champ" : ""}">
                  <td>
                    ${r.is_best 
                      ? `<span class="rank-badge-champ">#${r.rank} Best</span>` 
                      : `<span class="rank-badge-num">#${r.rank}</span>`}
                  </td>
                  <td class="eval-cell-model">${escapeHtml(r.model)}</td>
                  ${Object.keys(r).filter(k => k.startsWith("col")).sort().map(k => `
                    <td class="eval-cell-num">${escapeHtml(r[k] !== undefined ? r[k] : "-")}</td>
                  `).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>

        ${task.verdict ? `
          <div class="eval-verdict-box">
            <div class="eval-verdict-label">Architecture Verdict & Insight</div>
            <p class="eval-verdict-text">${escapeHtml(task.verdict)}</p>
          </div>
        ` : ""}
      </div>
    `;
  }

  function renderTaskBlock(taskKey, task) {
    const artifacts = task.artifacts || [];

    return `
      <div class="eval-task-block" data-task-id="${taskKey}">
        <div class="eval-task-header">
          <div class="eval-task-title-group">
            <div class="eval-task-pill">${escapeHtml(task.task_id)} • Pipeline Benchmark</div>
            <h3 class="eval-task-heading">${escapeHtml(task.title)}</h3>
            <p class="eval-task-caption">${escapeHtml(task.subtitle)}</p>
          </div>
          <div>
            <span class="badge badge-neutral">${escapeHtml(task.sample_info)}</span>
          </div>
        </div>

        <div class="eval-task-body">
          ${renderTableWrapper(taskKey, task)}

          <div class="eval-artifacts-panel">
            ${artifacts.length > 0 ? artifacts.map(art => `
              <div class="eval-artifact-card">
                <img src="${escapeHtml(art.src)}" alt="${escapeHtml(art.caption)}" class="eval-artifact-img" loading="lazy">
                <div class="eval-artifact-caption">${escapeHtml(art.caption)}</div>
              </div>
            `).join("") : `
              <div class="eval-artifact-card" style="padding: 1.5rem; text-align: center; color: var(--text-muted); font-size: 0.75rem;">
                No visual artifacts registered for this benchmark.
              </div>
            `}
          </div>
        </div>
      </div>
    `;
  }
}
