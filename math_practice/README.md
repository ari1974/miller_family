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

Grading model:

The test is graded leniently, the way a 4th-grade teacher would: if the math is
right, the answer gets full credit, regardless of which equivalent form was used
(mixed number vs improper fraction, `cm^2` vs `square centimeters`, with or
without commas, etc.).

After the score is shown, the review screen surfaces two separate sections:

1. **Missed problems** &mdash; only the answers that were genuinely wrong.
2. **Format tips** &mdash; answers that got credit but used a non-preferred form.
   The preferred form (what the real placement test most likely expects) is
   shown next to the student's answer, with the note "counted as correct".

Internally each question has both `acceptedAnswers` (just the preferred canonical
form) and `looseAcceptedAnswers` (all the equivalent forms). The lenient grade is
based on `looseAcceptedAnswers`. The strict match against `acceptedAnswers` is
used only to decide whether to surface a Format tip.

Adding a new test:

Edit `math-practice.js` and add another object to the `TESTS` array. Each question supports:

```js
{
  prompt: "Problem text",
  preferredAnswer: "Canonical answer in the exact form that scores credit",
  acceptedAnswers: ["Canonical answer"],   // strict: usually just [preferredAnswer]
  looseAcceptedAnswers: [                   // loose: equivalent forms for the
    "Canonical answer",                     //        Format-error diagnostic
    "Equivalent improper fraction",
    "Decimal equivalent",
    "Alternate unit spelling"
  ],
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
