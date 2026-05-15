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

    function update() {
      var vocab = Number(vocabInput.value);
      var gamma = Number(gammaInput.value);

      var standardVectors = 2 * gamma + 2;
      var standardValues = standardVectors * vocab;
      var compactValues = gamma * 4;
      var ratio = Math.max(1, Math.round(standardValues / compactValues));

      setText(root, "[data-memory-vocab]", formatCount(vocab));
      setText(root, "[data-memory-gamma]", String(gamma));
      setText(root, "[data-memory-ratio]", ratio.toLocaleString() + "x");
      setText(root, "[data-memory-standard-values]", formatCount(standardValues));
      setText(root, "[data-memory-flashspec-values]", formatCount(compactValues));
    }

    vocabInput.addEventListener("input", update);
    gammaInput.addEventListener("input", update);
    update();
  }

  function initGumbelInspector() {
    var root = document.getElementById("flashspec-gumbel-inspector");
    if (!root) return;

    var logits = [
      0.35, 0.72, 1.05, 1.34, 1.88, 0.62, 1.55, 0.28, 1.12,
      0.86, 1.42, 0.52, 1.22, 0.68, 1.7, 0.44, 1.0,
    ];
    var draftIndex = 4;
    var gumbels = logits.map(function () {
      return Math.max(0.08, sampleGumbel() + 0.35);
    });
    var board = root.querySelector("[data-gumbel-board]");
    var resampleButton = root.querySelector("[data-gumbel-resample]");

    function formatNumber(value) {
      if (!Number.isFinite(value)) return "-inf";
      return value.toFixed(2);
    }

    function normalizedHeight(value, minValue, maxValue) {
      if (!Number.isFinite(value)) return 7;
      if (maxValue === minValue) return 55;
      return 16 + ((value - minValue) / (maxValue - minValue)) * 80;
    }

    function renderSimpleBar(value, minValue, maxValue, variant) {
      var bar = document.createElement("div");
      bar.className = "flashspec-gumbel-bar";
      if (variant) bar.classList.add(variant);
      bar.style.setProperty("--flashspec-bar", normalizedHeight(value, minValue, maxValue) + "%");
      return bar;
    }

    function renderStackBar(logit, noise, maxScore) {
      var total = logit + noise;
      var bar = document.createElement("div");
      var logitPart = document.createElement("div");
      var noisePart = document.createElement("div");

      bar.className = "flashspec-gumbel-bar is-stack";
      bar.style.setProperty("--flashspec-bar", normalizedHeight(total, 0, maxScore) + "%");
      logitPart.className = "flashspec-gumbel-stack-logit";
      noisePart.className = "flashspec-gumbel-stack-noise";
      logitPart.style.setProperty("--flashspec-logit-part", (logit / total) * 100 + "%");
      noisePart.style.setProperty("--flashspec-noise-part", (noise / total) * 100 + "%");

      bar.append(noisePart, logitPart);
      return bar;
    }

    function renderRow(label, sublabel, values, sampledIndex, maskDraft, variant) {
      var finiteValues = values.filter(Number.isFinite);
      var minValue = Math.min.apply(null, finiteValues);
      var maxValue = Math.max.apply(null, finiteValues);

      var row = document.createElement("div");
      row.className = "flashspec-gumbel-row";

      var labelNode = document.createElement("div");
      labelNode.className = "flashspec-gumbel-label";
      labelNode.innerHTML = label + "<span>" + sublabel + "</span>";

      var bars = document.createElement("div");
      bars.className = "flashspec-gumbel-bars";

      values.forEach(function (value, index) {
        var cell = document.createElement("div");
        cell.className = "flashspec-gumbel-cell";

        if (index === draftIndex) cell.classList.add("is-draft");
        if (index === sampledIndex) cell.classList.add("is-sampled");
        if (maskDraft && index === draftIndex) cell.classList.add("is-masked");

        var token = document.createElement("div");
        token.className = "flashspec-gumbel-token";
        token.title = "t" + index + " = " + formatNumber(value);
        token.textContent = "t" + index;

        cell.append(renderSimpleBar(value, minValue, maxValue, variant), token);
        bars.append(cell);
      });

      row.append(labelNode, bars);
      return row;
    }

    function renderStackRow(scores, sampledIndex) {
      var maxScore = Math.max.apply(null, scores.filter(Number.isFinite));
      var row = document.createElement("div");
      var labelNode = document.createElement("div");
      var bars = document.createElement("div");

      row.className = "flashspec-gumbel-row";
      labelNode.className = "flashspec-gumbel-label";
      labelNode.innerHTML = "noisy scores<span>l_i + g_i</span>";
      bars.className = "flashspec-gumbel-bars";

      scores.forEach(function (score, index) {
        var cell = document.createElement("div");
        var token = document.createElement("div");

        cell.className = "flashspec-gumbel-cell";
        if (index === draftIndex) cell.classList.add("is-draft", "is-masked");
        if (index === sampledIndex) cell.classList.add("is-sampled");

        token.className = "flashspec-gumbel-token";
        token.title = "t" + index + " = " + formatNumber(score);
        token.textContent = "t" + index;

        if (index === draftIndex) {
          cell.append(renderSimpleBar(0.2, 0, 1, null), token);
        } else {
          cell.append(renderStackBar(logits[index], gumbels[index], maxScore), token);
        }

        bars.append(cell);
      });

      row.append(labelNode, bars);
      return row;
    }

    function render() {
      var scores = logits.map(function (logit, index) {
        return index === draftIndex ? -Infinity : logit + gumbels[index];
      });
      var sampledIndex = scores.reduce(function (bestIndex, score, index) {
        return score > scores[bestIndex] ? index : bestIndex;
      }, scores[0] === -Infinity ? 1 : 0);

      board.replaceChildren(
        renderRow("target logits", "l_i", logits, null, false, null),
        renderRow("Gumbel noise", "g_i", gumbels, null, false, "is-noise"),
        renderStackRow(scores, sampledIndex),
      );

      setText(root, "[data-gumbel-sampled]", "tok_" + sampledIndex);
      setText(root, "[data-gumbel-draft]", "tok_" + draftIndex);
    }

    resampleButton.addEventListener("click", function () {
      gumbels = logits.map(function () {
        return Math.max(0.08, sampleGumbel() + 0.35);
      });
      render();
    });

    render();
  }

  function initVerifyTabs() {
    var root = document.getElementById("flashspec-verify-tabs");
    if (!root) return;

    var buttons = root.querySelectorAll("[data-verify-tab]");
    var panels = root.querySelectorAll("[data-verify-panel]");

    function activate(tab) {
      buttons.forEach(function (button) {
        var active = button.getAttribute("data-verify-tab") === tab;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });

      panels.forEach(function (panel) {
        panel.hidden = panel.getAttribute("data-verify-panel") !== tab;
      });
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        activate(button.getAttribute("data-verify-tab"));
      });
    });

    activate("overview");
  }

  onReady(function () {
    initTrafficExplorer();
    initGumbelInspector();
    initVerifyTabs();
  });
})();
