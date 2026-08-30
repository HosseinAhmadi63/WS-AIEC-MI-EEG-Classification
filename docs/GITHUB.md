# Publish this repository on GitHub

The repository name, package metadata, DOI link, and clone URL are already
frozen as `WS-AIEC-MI-EEG-Classification` under the `HosseinAhmadi63` account.

## GitHub CLI method

Install and authenticate the GitHub CLI once, then run these commands from the
repository root:

```bash
git init
git add .
git commit -m "Add complete WS-AIEC MI-EEG reproduction pipeline"
git branch -M main
gh repo create HosseinAhmadi63/WS-AIEC-MI-EEG-Classification --public --source=. --remote=origin --push
```

The last command creates the repository and pushes `main`. If the empty GitHub
repository has already been created in the browser, use this exact final pair
instead:

```bash
git remote add origin https://github.com/HosseinAhmadi63/WS-AIEC-MI-EEG-Classification.git
git push -u origin main
```

## Browser upload method

1. Create a public repository named `WS-AIEC-MI-EEG-Classification` at
   <https://github.com/new>.
2. Leave **Add a README**, **Add .gitignore**, and **Choose a license** disabled;
   all three are already included here.
3. On the empty repository page, choose **uploading an existing file**.
4. Upload the complete contents of this folder, preserving the directory tree.
5. Use commit message `Add complete WS-AIEC MI-EEG reproduction pipeline`.

After publication, GitHub will detect `CITATION.cff` and expose **Cite this
repository** in the repository sidebar.
