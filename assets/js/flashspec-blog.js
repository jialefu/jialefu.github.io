(function () {
  "use strict";

  function onReady(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback);
    } else {
      callback();
    }
  }

  function setText(root, selector, value) {
    root.querySelectorAll(selector).forEach(function (node) {
      node.textContent = value;
    });
  }

  function formatCount(value) {
    if (value >= 1000000) return (value / 1000000).toFixed(value >= 10000000 ? 0 : 2) + "M";
    if (value >= 1000) return (value / 1000).toFixed(value >= 100000 ? 0 : 1) + "k";
    return String(Math.round(value));
  }

  function formatBytesLike(bytes) {
    var units = ["B", "KB", "MB", "GB"];
    var value = bytes;
    var unit = 0;

    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }

    var decimals = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 2;
    return value.toFixed(decimals) + " " + units[unit];
  }

  function softmaxLse(logits) {
    var maxLogit = Math.max.apply(null, logits);
    var sum = logits.reduce(function (acc, logit) {
      return acc + Math.exp(logit - maxLogit);
    }, 0);
    return maxLogit + Math.log(sum);
  }

  function sampleGumbel() {
    var eps = Number.EPSILON;
    var u = Math.min(1 - eps, Math.max(eps, Math.random()));
    return -Math.log(-Math.log(u));
  }

  function initTrafficExplorer() {
    var root = document.getElementById("flashspec-memory-explorer");
    if (!root) return;

    var vocabInput = root.querySelector("[data-memory-vocab-input]");
    var gammaInput = root.querySelector("[data-memory-gamma-input]");
    var standardBar = root.querySelector("[data-memory-standard-bar]");
    var flashspecBar = root.querySelector("[data-memory-flashspec-bar]");
    var tileSize = 8192;

    function update() {
      var vocab = Number(vocabInput.value);
      var gamma = Number(gammaInput.value);
      var tiles = Math.ceil(vocab / tileSize);

      var standardVectors = 2 * gamma + 2;
      var standardValues = standardVectors * vocab;
      var standardBytes = standardValues * 2;

      var compactValues = gamma * (4 * tiles + 1);
      var compactBytes = compactValues * 8;
      var ratio = Math.max(1, Math.round(standardValues / compactValues));
      var compactWidth = Math.max(4, Math.min(100, (compactValues / standardValues) * 100));

      setText(root, "[data-memory-vocab]", formatCount(vocab));
      setText(root, "[data-memory-gamma]", String(gamma));
      setText(root, "[data-memory-ratio]", ratio.toLocaleString() + "x");
      setText(root, "[data-memory-standard-values]", formatCount(standardValues));
      setText(root, "[data-memory-standard-bytes]", formatBytesLike(standardBytes));
      setText(root, "[data-memory-standard-tensors]", String(standardVectors));
      setText(root, "[data-memory-flashspec-values]", formatCount(compactValues));
      setText(root, "[data-memory-flashspec-bytes]", formatBytesLike(compactBytes));

      if (standardBar) standardBar.style.setProperty("--flashspec-meter", "100%");
      if (flashspecBar) flashspecBar.style.setProperty("--flashspec-meter", compactWidth + "%");
    }

    vocabInput.addEventListener("input", update);
    gammaInput.addEventListener("input", update);
    update();
  }

  function initOnePassInspector() {
    var root = document.getElementById("flashspec-one-pass-inspector");
    if (!root) return;

    var logits = [-1.2, -0.35, 0.55, 1.05, 2.0, 0.18, 1.5, -0.85, 0.9, 0.05, -0.48, 1.25];
    var draftIndex = 4;
    var mode = "acceptance";
    var gumbels = logits.map(sampleGumbel);

    var barsRoot = root.querySelector("[data-inspector-bars]");
    var barsTitle = root.querySelector("[data-inspector-bars-title]");
    var uniformInput = root.querySelector("[data-inspector-uniform-input]");
    var modeButtons = root.querySelectorAll("[data-inspector-mode]");
    var resampleButton = root.querySelector("[data-inspector-resample]");

    function formatNumber(value) {
      if (!Number.isFinite(value)) return "-inf";
      return value.toFixed(2);
    }

    function updateButtons() {
      modeButtons.forEach(function (button) {
        var isActive = button.getAttribute("data-inspector-mode") === mode;
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
    }

    function normalizedHeight(value, minValue, maxValue) {
      if (!Number.isFinite(value)) return 8;
      if (maxValue === minValue) return 55;
      return 16 + ((value - minValue) / (maxValue - minValue)) * 80;
    }

    function renderBars(values, recoveredIndex) {
      var finiteValues = values.filter(Number.isFinite);
      var minValue = Math.min.apply(null, finiteValues);
      var maxValue = Math.max.apply(null, finiteValues);

      barsRoot.replaceChildren();

      values.forEach(function (value, index) {
        var token = document.createElement("div");
        token.className = "flashspec-token";

        if (index === draftIndex) token.classList.add("is-draft");
        if (mode === "recovery" && index === recoveredIndex) token.classList.add("is-recovered");
        if (!Number.isFinite(value)) token.classList.add("is-masked");

        var bar = document.createElement("div");
        bar.className = "flashspec-token-bar";
        bar.style.setProperty("--flashspec-bar", normalizedHeight(value, minValue, maxValue) + "%");

        var label = document.createElement("div");
        label.className = "flashspec-token-label";
        label.textContent = "t" + index;

        var val = document.createElement("div");
        val.className = "flashspec-token-value";
        val.textContent = formatNumber(value);

        token.append(bar, label, val);
        barsRoot.append(token);
      });
    }

    function render() {
      var uniform = Number(uniformInput.value);
      var lse = softmaxLse(logits);
      var selectedLogit = logits[draftIndex];
      var prob = Math.exp(selectedLogit - lse);
      var accepted = uniform <= prob;

      var scores = logits.map(function (logit, index) {
        return index === draftIndex ? -Infinity : logit + gumbels[index];
      });
      var recoveredIndex = scores.reduce(function (bestIndex, score, index) {
        return score > scores[bestIndex] ? index : bestIndex;
      }, scores[0] === -Infinity ? 1 : 0);

      setText(root, "[data-inspector-decision]", accepted ? "accept" : "recover");
      setText(root, "[data-inspector-draft-token]", "tok_" + draftIndex);
      setText(root, "[data-inspector-selected-logit]", selectedLogit.toFixed(2));
      setText(root, "[data-inspector-lse]", lse.toFixed(2));
      setText(root, "[data-inspector-prob]", prob.toFixed(2));
      setText(root, "[data-inspector-uniform]", uniform.toFixed(2));
      setText(root, "[data-inspector-uniform-summary]", uniform.toFixed(2));
      setText(root, "[data-inspector-recovered]", "tok_" + recoveredIndex);

      if (barsTitle) {
        barsTitle.textContent = mode === "acceptance" ? "Toy target logits" : "Logits plus Gumbel noise";
      }

      renderBars(mode === "acceptance" ? logits : scores, recoveredIndex);
      updateButtons();
    }

    modeButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        mode = button.getAttribute("data-inspector-mode");
        render();
      });
    });

    resampleButton.addEventListener("click", function () {
      gumbels = logits.map(sampleGumbel);
      mode = "recovery";
      render();
    });

    uniformInput.addEventListener("input", render);
    render();
  }

  onReady(function () {
    initTrafficExplorer();
    initOnePassInspector();
  });
})();
