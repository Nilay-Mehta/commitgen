# Deploy commitgen Web Wrapper

This wrapper is a static site. It works as a portfolio demo on Vercel without
running the model server. The "Try local Ollama" mode only works when someone
has Ollama running locally with the `commitgen` model; otherwise the page falls
back to the browser preview.

## Vercel

1. Push this repo to GitHub.
2. Go to https://vercel.com/new.
3. Import `Nilay-Mehta/commitgen`.
4. Keep the default framework as `Other`.
5. Leave build command and output directory empty.
6. Deploy.

The root URL serves `web/index.html` through `vercel.json`.

## Local Preview

From the repo root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/web/
```
