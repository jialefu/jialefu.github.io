// Keep section navigation in sync with the home page as the reader scrolls.
document.addEventListener("DOMContentLoaded", () => {
  const sections = Array.from(document.querySelectorAll(".academic-home > section[id]"));
  if (!sections.length) return;
  const links = Array.from(document.querySelectorAll(".nav-links a[href*='#']"));
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
  update();
});
