# Screenshot Guide

Screenshots are not generated automatically. Capture them manually after running the app, API and GitHub Actions workflow.

Suggested screenshots:

| File | What To Capture |
| --- | --- |
| `streamlit_home.png` | Streamlit app before input or before generating a report |
| `streamlit_prediction_results.png` | predicted tags and similar-content output |
| `streamlit_json_output.png` | structured metadata JSON section |
| `fastapi_docs.png` | FastAPI Swagger docs at `/docs` |
| `github_actions_ci.png` | passing GitHub Actions CI screen |
| `repo_overview.png` | GitHub README/project overview |

Recommended location:

```text
docs/screenshots/
```

Avoid committing screenshots that expose local file paths, secrets, private data or raw dataset contents.
