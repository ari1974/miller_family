# Double Accelerated Math Practice — GitHub Pages Static App

Files:

- `index.html`
- `math-practice.js`

Upload both files to the same folder in a GitHub Pages repo.

Local test:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

Notes:

- The app is static: no server, no build step, no external libraries.
- Answers in progress are stored in `localStorage` on the device.
- Completed scores and missed problem numbers are stored in `localStorage` and also summarized in a cookie.
- Cookies may not work when opening `index.html` directly from the filesystem, so use the local HTTP server for testing.
- The JS avoids nullish coalescing, optional chaining, arrows, `const`, `let`, classes, and template literals so older `node --check` versions can parse it.

Adding a new test:

Edit `math-practice.js` and add another object to the `TESTS` array. Each question supports:

```js
{
  prompt: "Problem text",
  preferredAnswer: "Answer displayed during review",
  acceptedAnswers: ["Answer", "Equivalent answer"],
  topic: "Optional topic label",
  visual: {
    type: "linePlot",
    title: "Line plot title",
    columns: [
      { label: "1/4", marks: 2 },
      { label: "1/2", marks: 3 },
      { label: "3/4", marks: 1 }
    ]
  }
}
```
