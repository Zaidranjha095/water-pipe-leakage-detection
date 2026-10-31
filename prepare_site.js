const fs = require("fs");
const path = require("path");

const root = __dirname;
const web = path.join(root, "web");

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(root, relativePath), "utf8"));
}

function copyFile(fromRelative, toRelative) {
  const destination = path.join(web, toRelative);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(path.join(root, fromRelative), destination);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function replaceMarked(html, name, content) {
  const pattern = new RegExp(`<!--${name}:START-->[\\s\\S]*?<!--${name}:END-->`);
  if (!pattern.test(html)) {
    throw new Error(`Missing ${name} marker in web/index.html`);
  }
  return html.replace(pattern, `<!--${name}:START-->\n${content}\n<!--${name}:END-->`);
}

function linkItem(url, label) {
  if (url && String(url).trim()) {
    const safe = escapeHtml(url.trim());
    return `<li><a href="${safe}">${escapeHtml(label)}</a><span>${safe}</span></li>`;
  }
  return `<li><span class="pending">${escapeHtml(label)} will be added before final submission.</span></li>`;
}

function confusionNote(metrics) {
  const classes = metrics.class_order;
  const matrix = metrics.test.confusion_matrix;
  const pairs = [];
  classes.forEach((trueName, row) => {
    classes.forEach((predName, column) => {
      if (row !== column && matrix[row][column] > 0) {
        pairs.push([matrix[row][column], trueName, predName]);
      }
    });
  });
  pairs.sort((a, b) => b[0] - a[0]);
  if (!pairs.length) return "Every test clip was classified correctly.";
  const text = pairs
    .slice(0, 3)
    .map(([count, trueName, predName]) => `${count} ${trueName} predicted as ${predName}`)
    .join("; ");
  return `Largest test confusions: ${text}.`;
}

function inline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return html;
}

function splitRow(row) {
  return row.split("|").slice(1, -1).map((cell) => cell.trim());
}

function renderTable(rows) {
  const parsed = rows.map(splitRow).filter((cells) => cells.length);
  const bodyRows = parsed.filter((cells) => !cells.every((cell) => /^:?-+:?$/.test(cell)));
  const header = bodyRows[0];
  const rest = bodyRows.slice(1);
  const head = header.map((cell) => `<th>${inline(cell)}</th>`).join("");
  const body = rest
    .map((cells) => `<tr>${cells.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderImage(line) {
  const match = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (!match) return `<p>${inline(line)}</p>`;
  const src = match[2]
    .replace(/^\.\.\/outputs\//, "images/")
    .replace(/^\.\.\/samples\//, "samples/");
  return `<figure class="figure"><img src="${escapeHtml(src)}" alt="${escapeHtml(match[1])}"><figcaption>${escapeHtml(match[1])}</figcaption></figure>`;
}

function isSpecial(line) {
  return (
    line.startsWith("#") ||
    line.startsWith("```") ||
    line.startsWith("|") ||
    line.startsWith("- ") ||
    line.startsWith("> ") ||
    line.startsWith("![")
  );
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith("```")) {
      const buffer = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        buffer.push(lines[index]);
        index += 1;
      }
      html.push(`<pre><code>${escapeHtml(buffer.join("\n"))}</code></pre>`);
      index += 1;
      continue;
    }
    if (line.startsWith("|")) {
      const table = [];
      while (index < lines.length && lines[index].startsWith("|")) {
        table.push(lines[index]);
        index += 1;
      }
      html.push(renderTable(table));
      continue;
    }
    if (line.startsWith("### ")) {
      html.push(`<h3>${inline(line.slice(4))}</h3>`);
      index += 1;
      continue;
    }
    if (line.startsWith("## ")) {
      html.push(`<h2>${inline(line.slice(3))}</h2>`);
      index += 1;
      continue;
    }
    if (line.startsWith("# ")) {
      html.push(`<h1>${inline(line.slice(2))}</h1>`);
      index += 1;
      continue;
    }
    if (line.startsWith("> ")) {
      html.push(`<blockquote><p>${inline(line.slice(2))}</p></blockquote>`);
      index += 1;
      continue;
    }
    if (line.startsWith("- ")) {
      const items = [];
      while (index < lines.length && lines[index].startsWith("- ")) {
        items.push(`<li>${inline(lines[index].slice(2))}</li>`);
        index += 1;
      }
      html.push(`<ul>${items.join("")}</ul>`);
      index += 1;
      continue;
    }
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("![")) {
      html.push(renderImage(line.trim()));
      index += 1;
      continue;
    }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isSpecial(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    html.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }
  return html.join("\n");
}

function reportPage(body) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="index, follow">
  <title>Project report · WaterGuard AI</title>
  <link rel="icon" href="favicon.svg">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,600&family=Fraunces:opsz,wght@9..144,560&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="site-header">
    <div class="wrap header-inner">
      <a class="brand" href="index.html"><span class="mark" aria-hidden="true"></span><span><strong>WaterGuard AI</strong><small>Project report</small></span></a>
      <nav>
        <a href="index.html">Overview</a>
        <a href="index.html#demo">Live demo</a>
        <a href="index.html#downloads">Downloads</a>
        <a href="downloads/presentation.pptx">Presentation</a>
      </nav>
    </div>
  </header>
  <main class="article">
    ${body}
  </main>
  <footer class="site-footer"><div class="wrap"><p>Zaid Ahmed · SU92-BSCSM-F23-553 · Deep Learning</p></div></footer>
</body>
</html>
`;
}

const required = [
  "weights/model.pth",
  "weights/model.onnx",
  "weights/config.json",
  "weights/class_names.json",
  "model_manifest.json",
  "outputs/metrics.json",
  "outputs/training_curves.png",
  "outputs/confusion_matrix.png",
  "presentation/presentation.pptx",
  "report/PROJECT_REPORT.md",
];

for (const relativePath of required) {
  if (!fs.existsSync(path.join(root, relativePath))) {
    throw new Error(`Missing ${relativePath}. Train the model and run python build_docs.py first.`);
  }
}

copyFile("weights/model.pth", "downloads/model.pth");
copyFile("weights/model.onnx", "downloads/model.onnx");
copyFile("weights/config.json", "downloads/config.json");
copyFile("weights/class_names.json", "downloads/class_names.json");
copyFile("model_manifest.json", "downloads/model_manifest.json");
copyFile("outputs/metrics.json", "downloads/metrics.json");
copyFile("outputs/training_curves.png", "images/training_curves.png");
copyFile("outputs/confusion_matrix.png", "images/confusion_matrix.png");
copyFile("presentation/presentation.pptx", "downloads/presentation.pptx");
fs.cpSync(path.join(root, "samples"), path.join(web, "samples"), { recursive: true });

const config = readJson("project.config.json");
const metrics = readJson("outputs/metrics.json");
const classes = metrics.class_order;
const counts = metrics.counts;

const CLASS_LABELS = {
  leak: "Leak",
  no_leak: "No leak",
  environmental_noise: "Environmental noise",
};

const datasetHtml = `<ul class="dist">
${classes
  .map((name) => {
    const count = counts.all[name];
    const width = ((count / counts.total) * 100).toFixed(1);
    const label = CLASS_LABELS[name] || name;
    return `<li><span>${escapeHtml(label)}</span><i style="--w:${width}%"></i><strong>${count}</strong></li>`;
  })
  .join("\n")}
</ul>
<p>${counts.total} clips total. Split ${counts.train.total} train / ${counts.val.total} validation / ${counts.test.total} test.</p>`;

const rows = classes
  .map((name) => {
    const row = metrics.test.per_class[name];
    return `<tr><td>${escapeHtml(name)}</td><td>${pct(row.precision)}</td><td>${pct(row.recall)}</td><td>${pct(row.f1)}</td><td>${row.support}</td></tr>`;
  })
  .join("");

const device = metrics.training.gpu || "CPU";
const resultsHtml = `<div class="stats">
  <article class="stat"><span>Test accuracy</span><strong>${pct(metrics.test.accuracy)}</strong></article>
  <article class="stat"><span>Macro F1</span><strong>${pct(metrics.test.macro_f1)}</strong></article>
  <article class="stat"><span>Best validation accuracy</span><strong>${pct(metrics.training.best_val_accuracy)}</strong></article>
  <article class="stat"><span>Epochs run</span><strong>${metrics.training.epochs_ran}</strong></article>
</div>
<p class="fine">Trained on ${escapeHtml(device)}. The network has ${Number(metrics.training.parameter_count).toLocaleString("en-US")} parameters. Weighted F1 is ${pct(metrics.test.weighted_f1)}. Best validation accuracy ${pct(metrics.training.best_val_accuracy)} is the peak on the 150-clip val split, not the last epoch. ${escapeHtml(confusionNote(metrics))}</p>
<div class="table-wrap"><table>
  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
  <tbody>${rows}</tbody>
</table></div>`;

const studentHtml = `<p class="byline">${escapeHtml(config.studentName)} · ${escapeHtml(config.rollNumber)} · ${escapeHtml(config.course)}</p>`;
const linksHtml = `<ul class="link-list">
${linkItem(config.githubUrl, "GitHub repository")}
${linkItem(config.youtubeUrl, "YouTube walkthrough")}
${linkItem(config.linkedinUrl, "LinkedIn post")}
${linkItem(config.vercelUrl, "Vercel deployment")}
</ul>`;

let indexHtml = fs.readFileSync(path.join(web, "index.html"), "utf8");
indexHtml = replaceMarked(indexHtml, "STUDENT", studentHtml);
indexHtml = replaceMarked(indexHtml, "DATASET", datasetHtml);
indexHtml = replaceMarked(indexHtml, "RESULTS", resultsHtml);
indexHtml = replaceMarked(indexHtml, "LINKS", linksHtml);
fs.writeFileSync(path.join(web, "index.html"), indexHtml);

const markdown = fs.readFileSync(path.join(root, "report", "PROJECT_REPORT.md"), "utf8");
fs.writeFileSync(path.join(web, "report.html"), reportPage(renderMarkdown(markdown)));
console.log("Prepared web/downloads, web/images, web/samples, and web/report.html");
