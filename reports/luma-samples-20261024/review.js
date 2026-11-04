'use strict';
const samples = JSON.parse(document.getElementById('sample-data').textContent);
let threshold = 75;

function updateGateLabels() {
  document.getElementById('threshold-label').textContent = threshold ? `T = ${threshold}` : '关闭';
  for (const sample of samples) {
    const node = document.querySelector(`[data-sample="${sample.sample_id}"] .gate-state`);
    node.className = 'gate-state';
    if (threshold === 0) {
      node.textContent = '预览：不按亮度区分';
    } else if (sample.max_luma < threshold) {
      node.textContent = `本片段全部帧低于 ${threshold}`;
    } else if (sample.min_luma >= threshold) {
      node.textContent = `本片段全部帧不低于 ${threshold}`;
      node.classList.add('above');
    } else {
      node.textContent = `本片段跨越 ${threshold}；窗口均值需另判`;
      node.classList.add('crossing');
    }
  }
}

for (const button of document.querySelectorAll('[data-threshold]')) {
  button.addEventListener('click', () => {
    threshold = Number(button.dataset.threshold);
    for (const b of document.querySelectorAll('[data-threshold]')) {
      const active = b === button;
      b.classList.toggle('selected', active);
      b.setAttribute('aria-pressed', String(active));
    }
    updateGateLabels();
  });
}

for (const button of document.querySelectorAll('[data-filter]')) {
  button.addEventListener('click', () => {
    const selected = button.dataset.filter;
    for (const b of document.querySelectorAll('[data-filter]')) {
      const active = b === button;
      b.classList.toggle('selected', active);
      b.setAttribute('aria-pressed', String(active));
    }
    for (const section of document.querySelectorAll('section[data-bin]')) {
      section.hidden = selected !== 'all' && section.dataset.bin !== selected;
      if (section.hidden) section.querySelectorAll('video').forEach(video => video.pause());
    }
    const count = samples.filter(sample => selected === 'all' || sample.bin_index === Number(selected)).length;
    document.getElementById('visible-count').textContent = `正在显示${count}段`;
  });
}

for (const details of document.querySelectorAll('details')) {
  details.addEventListener('toggle', () => {
    const video = details.querySelector('video');
    if (!details.open) video.pause();
    else if (video.readyState === 0) video.load();
  });
}
for (const video of document.querySelectorAll('video')) {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
  });
}
updateGateLabels();
