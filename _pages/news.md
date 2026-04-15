---
layout: page
title: News
permalink: /news/
description: Research milestones, paper acceptances, and academic updates.
---

<p class="page-intro">
  Recent milestones across papers, research internships, and graduate study.
</p>

<div class="timeline-list news-page">
  {% for item in site.data.news %}
    <article class="timeline-panel">
      <div class="timeline-label">{{ item.time }}</div>
      <div class="timeline-copy">
        <h3>
          {% if item.url %}
            <a href="{{ item.url }}">{{ item.title }}</a>
          {% else %}
            {{ item.title }}
          {% endif %}
        </h3>
        <p>{{ item.description }}</p>
      </div>
    </article>
  {% endfor %}
</div>
