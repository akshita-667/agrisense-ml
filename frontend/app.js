// AgriSense frontend: talks to the Flask API in app.py.
(function () {
  "use strict";

  // Same origin by default (Flask serves this page). To host the frontend
  // elsewhere, set window.AGRISENSE_API_BASE = "https://your-api.example.com"
  // before this script loads.
  const API_BASE = window.AGRISENSE_API_BASE || "";

  const MIN_SHOWN_PCT = 1;

  const FIELDS = ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"];

  const UNITS = { N: "kg/ha", P: "kg/ha", K: "kg/ha", ph: "", temperature: "°C", humidity: "%", rainfall: "mm" };

  const FEATURE_NAMES = {
    N: "Nitrogen", P: "Phosphorus", K: "Potassium", ph: "Soil pH",
    temperature: "Temperature", humidity: "Humidity", rainfall: "Rainfall",
  };

  // Maps the API's explanation label back to a field, so we can show units.
  const LABEL_TO_FIELD = {
    "Nitrogen level": "N", "Phosphorus level": "P", "Potassium level": "K",
    "Soil pH": "ph", "Temperature": "temperature", "Humidity": "humidity", "Rainfall": "rainfall",
  };

  const CROPS = {
    apple: ["Apple", "🍎"], banana: ["Banana", "🍌"], blackgram: ["Black gram (Urad)", "🫘"],
    chickpea: ["Chickpea (Chana)", "🫘"], coconut: ["Coconut", "🥥"], coffee: ["Coffee", "☕"],
    cotton: ["Cotton", "☁️"], grapes: ["Grapes", "🍇"], jute: ["Jute", "🌿"],
    kidneybeans: ["Kidney beans (Rajma)", "🫘"], lentil: ["Lentil (Masoor)", "🫘"], maize: ["Maize", "🌽"],
    mango: ["Mango", "🥭"], mothbeans: ["Moth beans", "🫘"], mungbean: ["Mung bean (Moong)", "🫛"],
    muskmelon: ["Muskmelon", "🍈"], orange: ["Orange", "🍊"], papaya: ["Papaya", "🥭"],
    pigeonpeas: ["Pigeon pea (Tur/Arhar)", "🫛"], pomegranate: ["Pomegranate", "🔴"], rice: ["Rice", "🌾"],
    watermelon: ["Watermelon", "🍉"],
  };

  // Example inputs taken from typical rows in the training data.
  const PRESETS = {
    rice: { N: 90, P: 42, K: 43, ph: 6.5, temperature: 20.9, humidity: 82, rainfall: 203 },
    chickpea: { N: 40, P: 72, K: 77, ph: 7.3, temperature: 18, humidity: 17, rainfall: 82 },
    cotton: { N: 118, P: 42, K: 19, ph: 6.9, temperature: 24, humidity: 80, rainfall: 75 },
  };

  const $ = (sel) => document.querySelector(sel);
  const form = $("#crop-form");
  const submitBtn = $("#submit-btn");
  const formError = $("#form-error");
  const recList = $("#rec-list");
  const empty = $("#results-empty");
  const notesBox = $("#results-notes");

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else node.setAttribute(k, v);
      }
    }
    (children || []).forEach((c) => c && node.append(c));
    return node;
  }

  function cropInfo(key) {
    const known = CROPS[key];
    if (known) return { name: known[0], emoji: known[1] };
    return { name: key.charAt(0).toUpperCase() + key.slice(1), emoji: "🌱" };
  }

  function fmt(n) {
    return Number.isInteger(n) ? String(n) : String(Math.round(n * 10) / 10);
  }

  // ---------- Errors ----------
  function clearErrors() {
    formError.hidden = true;
    formError.textContent = "";
    FIELDS.forEach((f) => {
      document.querySelector(`[data-field="${f}"]`).classList.remove("has-error");
      $(`#${f}-error`).textContent = "";
      $(`#${f}`).removeAttribute("aria-invalid");
    });
  }

  function showFieldError(field, message) {
    const wrap = document.querySelector(`[data-field="${field}"]`);
    if (!wrap) return false;
    wrap.classList.add("has-error");
    $(`#${field}-error`).textContent = message;
    const input = $(`#${field}`);
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", `${field}-error`);
    return true;
  }

  function showFormError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  // Quick checks in the browser so farmers get instant feedback.
  // The API validates again, so this is only for convenience.
  function validateLocally(values) {
    let ok = true;
    FIELDS.forEach((f) => {
      const input = $(`#${f}`);
      const raw = input.value.trim();
      if (raw === "") {
        showFieldError(f, "Please enter a value.");
        ok = false;
        return;
      }
      const n = Number(raw);
      const min = Number(input.min);
      const max = Number(input.max);
      if (!Number.isFinite(n)) {
        showFieldError(f, "Please enter a number.");
        ok = false;
      } else if (n < min || n > max) {
        showFieldError(f, `Must be between ${min} and ${max}${UNITS[f] ? " " + UNITS[f] : ""}.`);
        ok = false;
      } else {
        values[f] = n;
      }
    });
    return ok;
  }

  // ---------- Results ----------
  function parseExplanation(text) {
    const m = /^(.+?) \(([-\d.]+)\) (supports|works against)/.exec(text);
    if (!m) return { text, positive: !/against/.test(text) };
    const field = LABEL_TO_FIELD[m[1]];
    const name = field ? FEATURE_NAMES[field] : m[1];
    const u = field ? UNITS[field] : "";
    const unit = !u ? "" : u === "%" ? "%" : " " + u;
    const positive = m[3] === "supports";
    return {
      text: `${name} ${m[2]}${unit} ${positive ? "suits this crop" : "is less suitable"}`,
      positive,
    };
  }

  function renderRecommendation(rec, index) {
    const info = cropInfo(rec.crop);
    const pct = Math.max(0, Math.min(100, Number(rec.confidence_pct) || 0));

    const why = el("ul", { class: "why" },
      (rec.explanation || []).map((line) => {
        const p = parseExplanation(line);
        return el("li", null, [
          el("span", { class: `sign ${p.positive ? "plus" : "minus"}`, "aria-hidden": "true", text: p.positive ? "+" : "−" }),
          el("span", { text: p.text }),
        ]);
      })
    );

    const meterFill = el("span", { style: "width:0%" });
    const item = el("li", { class: `rec${index === 0 ? " top" : ""}` }, [
      el("div", { class: "rec-head" }, [
        el("span", { class: "rec-emoji", "aria-hidden": "true", text: info.emoji }),
        el("div", { class: "rec-title" }, [
          el("div", { class: "rec-rank", text: index === 0 ? "Best match" : `Option ${index + 1}` }),
          el("div", { class: "rec-name", text: info.name }),
        ]),
        el("span", { class: "rec-pct", text: `${fmt(pct)}%` }),
      ]),
      el("div", {
        class: "meter", role: "meter", "aria-label": `Match for ${info.name}`,
        "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(pct),
      }, [meterFill]),
      el("div", { class: "why-title", text: "Why" }),
      why,
    ]);
    requestAnimationFrame(() => requestAnimationFrame(() => { meterFill.style.width = `${pct}%`; }));
    return item;
  }

  function renderResults(data) {
    empty.hidden = true;
    // Hide options the model gives almost no chance (always keep the best one).
    const all = data.recommendations || [];
    const shown = all.filter((r, i) => i === 0 || Number(r.confidence_pct) >= MIN_SHOWN_PCT);
    recList.replaceChildren(...shown.map(renderRecommendation));
    if (shown.length < all.length) {
      recList.append(el("li", { class: "rec-more", text: "Other crops were a poor match for these conditions." }));
    }

    // Field-level warnings are the same for every crop, so show them once.
    const warnings = [...new Set((data.recommendations || []).flatMap((r) => r.warnings || []))];
    const notes = data.notes || [];
    notesBox.replaceChildren();
    if (notes.length) {
      notesBox.append(el("div", { class: "notice" }, [
        el("strong", { text: "Some values are unusual" }),
        el("ul", null, notes.map((n) => el("li", { text: n }))),
      ]));
    }
    if (warnings.length) {
      notesBox.append(el("div", { class: "notice" }, [
        el("strong", { text: "General soil tips" }),
        el("ul", null, warnings.map((w) => el("li", { text: w }))),
      ]));
    }

    if (window.matchMedia("(max-width: 900px)").matches) {
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function resetResults() {
    recList.replaceChildren();
    notesBox.replaceChildren();
    empty.hidden = false;
  }

  // ---------- Submit ----------
  function setLoading(on) {
    submitBtn.disabled = on;
    submitBtn.classList.toggle("loading", on);
    submitBtn.querySelector(".btn-label").textContent = on ? "Checking your field…" : "Recommend crops";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();
    const values = {};
    if (!validateLocally(values)) {
      const firstBad = form.querySelector('[aria-invalid="true"]');
      if (firstBad) firstBad.focus();
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...values, top_k: 3 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (data.details) {
          const unmatched = Object.entries(data.details).filter(([f, msg]) => !showFieldError(f, msg));
          if (unmatched.length) showFormError(unmatched.map(([, m]) => m).join(" "));
        } else {
          showFormError(data.error || `Something went wrong (error ${res.status}). Please try again.`);
        }
        return;
      }
      renderResults(data);
    } catch (err) {
      showFormError("Could not reach the AgriSense server. Check that it is running and try again.");
    } finally {
      setLoading(false);
    }
  });

  form.addEventListener("reset", () => {
    clearErrors();
    resetResults();
  });

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[btn.dataset.preset];
      clearErrors();
      FIELDS.forEach((f) => { $(`#${f}`).value = preset[f]; });
    });
  });

  // ---------- Model info ----------
  async function loadModelInfo() {
    try {
      const res = await fetch(`${API_BASE}/api/model-info`);
      if (!res.ok) return;
      const info = await res.json();
      if (typeof info.test_accuracy === "number") {
        $("#stat-accuracy").textContent = `${(info.test_accuracy * 100).toFixed(1)}%`;
      }
      const models = Object.keys(info.model_comparison || {});
      if (models.length) $("#stat-models").textContent = String(models.length);
      if (info.n_classes) $("#stat-crops").textContent = String(info.n_classes);

      const imp = Object.entries(info.feature_importance || {}).sort((a, b) => b[1] - a[1]);
      const max = imp.length ? imp[0][1] : 1;
      $("#importance-bars").replaceChildren(...imp.map(([f, v]) => {
        const fill = el("div", { class: "bar-fill", style: `width:${(v / max) * 100}%` });
        return el("li", { class: "bar-row" }, [
          el("span", { text: FEATURE_NAMES[f] || f }),
          el("div", { class: "bar-track", "aria-hidden": "true" }, [fill]),
          el("span", { class: "bar-val", text: `${Math.round(v * 100)}%` }),
        ]);
      }));
    } catch (_) {
      // Model info is a nice-to-have, so the page still works without it.
    }
  }

  loadModelInfo();
})();
