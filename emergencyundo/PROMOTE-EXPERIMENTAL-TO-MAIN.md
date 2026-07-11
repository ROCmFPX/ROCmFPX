# Emergency Undo: Experimental-to-Main Promotion

## Protected recovery points

- Old GitHub `main`: `5b3956605309dd3e6beed49c8f3a41423ba71d25`
- Tested experimental source:
  `a6a93765f7ce9779c13f9881164a65f7a9f31198`
- Remote archive branch:
  `archive/main-before-experimental-2026-07-11`
- Local full backup: `/home/caf/ROCmFPXMAIN`
- Local complete-history bundle:
  `/home/caf/ROCmFPXMAIN/ROCmFPX-main-2026-07-11.bundle`

## Important rule

Do not reset or force-push the shared `main` branch. Undo the promotion with a
new reviewed revert pull request so history and author credit remain intact.

## Find the completed promotion

```bash
PROMOTION_PR=$(gh pr list \
  --repo charlie12345/ROCmFPX \
  --state merged \
  --head agent/promote-experimental-to-main-2026-07-11 \
  --json number \
  --jq '.[0].number')

MERGE_SHA=$(gh pr view "$PROMOTION_PR" \
  --repo charlie12345/ROCmFPX \
  --json mergeCommit \
  --jq '.mergeCommit.oid')

printf 'promotion PR=%s merge=%s\n' "$PROMOTION_PR" "$MERGE_SHA"
```

Confirm that both values are non-empty before continuing.

## Create the rollback pull request

```bash
git clone https://github.com/charlie12345/ROCmFPX.git \
  /home/caf/ROCmFPX-PROMOTION-ROLLBACK
cd /home/caf/ROCmFPX-PROMOTION-ROLLBACK
git switch main
git pull --ff-only origin main
git switch -c emergency/revert-experimental-promotion

# The promotion is merged with GitHub's merge-commit method. Parent 1 is the
# previous main line, so -m 1 restores the pre-promotion tree.
git revert -m 1 "$MERGE_SHA"

scripts/check-rocmfpx-reference.sh
scripts/check-rocmfpx-ranked-policy.sh
cmake -S . -B build-rollback -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build build-rollback --target test-backend-ops -j 2
./build-rollback/bin/test-backend-ops test \
  -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b CPU

git push -u origin emergency/revert-experimental-promotion
gh pr create \
  --repo charlie12345/ROCmFPX \
  --base main \
  --head emergency/revert-experimental-promotion \
  --title "revert: experimental-to-main promotion" \
  --body "Reverts the promotion merge after local validation."
```

Wait for GitHub checks and merge that rollback PR using **Create a merge
commit**. The remote archive and local bundle are recovery references, not a
reason to rewrite `main`.
