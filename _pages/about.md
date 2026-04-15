---
layout: home
title: Home
permalink: /
---

{% assign profile = site.data.profile %}

<section class="hero-card">
  <div class="hero-copy">
    <div class="hero-kicker">{{ profile.role }}</div>
    <h1>{{ profile.name }}</h1>
    <p class="hero-headline">{{ profile.headline }}</p>
    <p class="hero-meta">
      Based in {{ profile.location }}, I am a master's student at
      <a href="{{ profile.school_url }}">Southeast University</a> and a member of
      <a href="{{ profile.lab_url }}">PALM Lab</a>, advised by
      <a href="{{ profile.advisor.url }}">{{ profile.advisor.name }}</a>.
    </p>
    <div class="hero-actions">
      {% for link in profile.links %}
        <a class="btn btn-sm z-depth-0" href="{{ link.url }}">
          <i class="{{ link.icon }}"></i>{{ link.label }}
        </a>
      {% endfor %}
    </div>
    <div class="hero-jump-links">
      <a href="#publications">Selected publications</a>
      <a href="#news">News</a>
      <a href="#experience">Experience</a>
    </div>
  </div>

  <div class="hero-profile">
    <div class="profile-frame">
      <img src="{{ '/assets/img/jiale-fu.jpg' | relative_url }}" alt="Portrait of Jiale Fu">
    </div>
    <div class="profile-note">
      <p><strong>Affiliation</strong><br>{{ profile.affiliation }}</p>
      <p><strong>Research</strong><br>Efficient inference, speculative decoding, and reasoning for large language models.</p>
    </div>
  </div>
</section>

<section class="info-grid">
  <div class="surface-card">
    <h2>Bio</h2>
    <p>{{ profile.bio }}</p>
  </div>
  <div class="surface-card">
    <h2>Research Focus</h2>
    <div class="tag-list">
      {% for tag in profile.research_tags %}
        <span class="tag-pill">{{ tag }}</span>
      {% endfor %}
    </div>
  </div>
</section>

<section id="publications" class="home-section">
  <div class="section-heading">
    <div>
      <div class="section-kicker">Research Output</div>
      <h2>Selected Publications</h2>
    </div>
    <a class="btn btn-sm z-depth-0 section-link" href="{{ '/publications/' | relative_url }}">All selected papers</a>
  </div>
  {% include publication_cards.liquid items=site.data.publications %}
</section>

<section id="news" class="home-section">
  <div class="section-heading">
    <div>
      <div class="section-kicker">Recent Updates</div>
      <h2>News</h2>
    </div>
    <a class="btn btn-sm z-depth-0 section-link" href="{{ '/news/' | relative_url }}">News archive</a>
  </div>
  <div class="timeline-list">
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
</section>

<section id="experience" class="home-section">
  <div class="section-heading">
    <div>
      <div class="section-kicker">Background</div>
      <h2>Experience</h2>
    </div>
  </div>
  <div class="timeline-list">
    {% for item in site.data.experience %}
      <article class="timeline-panel">
        <div>
          <div class="timeline-label">{{ item.period }}</div>
          <div class="timeline-meta">{{ item.location }}</div>
        </div>
        <div class="timeline-copy">
          <h3>{{ item.organization }} · {{ item.role }}</h3>
          <p>{{ item.description }}</p>
        </div>
      </article>
    {% endfor %}
  </div>
</section>
