// Keep section navigation and the internship disclosures in sync with the page.
document.addEventListener("DOMContentLoaded", () => {
  const sections = Array.from(document.querySelectorAll(".academic-home > section[id]"));
  if (!sections.length) return;
  const links = Array.from(document.querySelectorAll(".nav-links a[href*='#']"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let pending = false;
  const update = () => {
    let current = sections[0].id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 140) current = section.id;
    }
    for (const link of links) {
      if (link.hash === `#${current}`) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
    pending = false;
  };
  window.addEventListener("scroll", () => {
    if (!pending) {
      pending = true;
      window.requestAnimationFrame(update);
    }
  }, { passive: true });

  const disclosures = [];
  document.querySelectorAll("[data-organization-group]").forEach((group) => {
    const grid = group.querySelector(".organization-options");
    const buttons = Array.from(group.querySelectorAll(".organization-button"));
    const states = buttons.map((button) => ({
      button,
      panel: document.getElementById(button.getAttribute("aria-controls")),
      open: false,
      heightAnimation: null,
      contentAnimation: null,
    }));
    let active = null;

    const finish = (state) => {
      state.heightAnimation?.cancel();
      state.contentAnimation?.cancel();
      state.heightAnimation = state.contentAnimation = null;
      state.panel.hidden = !state.open;
      state.panel.style.height = "";
    };
    const place = (state) => {
      // Put the full-width detail after the last button in the selected row.
      const columns = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
      const index = buttons.indexOf(state.button);
      const rowEnd = Math.min(buttons.length - 1, Math.floor(index / columns) * columns + columns - 1);
      buttons[rowEnd].insertAdjacentElement("afterend", state.panel);
    };
    const setOpen = (state, open) => {
      const { panel, button } = state;
      const content = panel.querySelector(".organization-panel-inner");
      const startHeight = panel.hidden ? 0 : panel.getBoundingClientRect().height;
      const contentStyle = getComputedStyle(content);
      const startOpacity = panel.hidden ? 0 : Number(contentStyle.opacity);
      const startTransform = panel.hidden ? "translateY(-8px)" : contentStyle.transform;
      finish(state);
      state.open = open;
      button.setAttribute("aria-expanded", String(open));
      panel.setAttribute("aria-hidden", String(!open));
      panel.inert = !open;
      panel.hidden = false;

      if (reducedMotion.matches) {
        finish(state);
        update();
        return;
      }
      const endHeight = open ? content.getBoundingClientRect().height : 0;
      const duration = Math.min(420, 280 + Math.abs(endHeight - startHeight) * 0.25);
      const options = { duration, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "both" };
      panel.style.height = `${startHeight}px`;
      const animation = panel.animate([{ height: `${startHeight}px` }, { height: `${endHeight}px` }], options);
      state.heightAnimation = animation;
      state.contentAnimation = content.animate([
        { opacity: startOpacity, transform: startTransform },
        { opacity: open ? 1 : 0, transform: open ? "translateY(0)" : "translateY(-8px)" },
      ], { ...options, duration: open ? duration : Math.min(180, duration) });
      animation.onfinish = () => {
        if (state.heightAnimation !== animation) return;
        finish(state);
        update();
      };
    };
    const close = (state, restoreFocus = false) => {
      if (restoreFocus) state.button.focus({ preventScroll: true });
      setOpen(state, false);
      if (active === state) active = null;
    };
    states.forEach((state) => {
      state.button.addEventListener("click", () => {
        if (state.open) {
          close(state);
          return;
        }
        if (active) close(active);
        place(state);
        active = state;
        setOpen(state, true);
      });
      state.panel.querySelector(".organization-close").addEventListener("click", () => close(state, true));
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && active) {
        event.preventDefault();
        close(active, true);
      }
    });
    disclosures.push(() => {
      states.forEach(finish);
      if (active) place(active);
    });
  });

  let resizeFrame;
  const settle = () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      disclosures.forEach((refresh) => refresh());
      update();
    });
  };
  window.addEventListener("resize", settle);
  reducedMotion.addEventListener("change", settle);
  update();
});
