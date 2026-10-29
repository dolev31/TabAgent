# Maintaining TabAgent: reviewing contributions

How a maintainer takes a contribution from the first comment to the merge. The contributor's side of the same bar is
in [CONTRIBUTING.md](CONTRIBUTING.md); who decides, and how a disputed number is settled, is in
[GOVERNANCE.md](GOVERNANCE.md).

## 1. Triage, within a few days

- **First-time contributors' CI waits for approval** (the repository requires it). Before approving a run, check that
  the pull request touches no workflow, build or release file: `.github/`, `scripts/`, `pyproject.toml`, `uv.lock`,
  `Makefile`. If it does, read those lines first: a fork's workflow change runs with your approval.
- **Label agent use.** When the description names an agent under "Written with:", add `agent-assisted`.
- **Assign on claim.** When someone comments "I'm taking this", assign the issue to them at once. GitHub only
  assigns a non-collaborator who has commented on the issue, so ask a pull request's author to comment there first.
- **One pull request per issue.** A duplicate is closed with thanks, credit for any idea the merged one adopts, and a
  pointer to an unclaimed issue on the [Wanted](docs/project/wanted.md) page.
- **Waivers are labels, given with a reason in the thread:** `no-issue` for a small fix with no issue,
  `no-changelog` when nothing a user sees changed, `no-new-test` for a pure refactor, `gate-waived: <gate>` for one
  rule of the diff gate that cannot apply.

## 2. Push back on shallow contributions

A contribution is shallow when it meets the letter of an issue and not its purpose. Typical signs:

- it fixes the one line the issue named, while the same pattern sits elsewhere;
- its only tests use fabricated logs, for a problem that lives in real logs or a real agent;
- its evidence is a claim ("this speeds up decide") rather than a result;
- its description and checklist were filled in, but nobody read the diff;
- it is formatting, spelling or a rename with no issue behind it.

How to push back:

1. **Find the concrete gap in the code first.** Search for the pattern, read the call sites, run the reproduction. A
   request that names a file and line is acted on; "please be more thorough" is not.
2. **Raise the bar item by item**, each with what to do and how it will be checked: the missed call site, a
   reproduction on real logs or a public dataset with the numbers on main and on the branch, the audit of every place
   the pattern occurs.
3. **Say so openly when you raise it mid-review.** If you called something ready, say that was premature and why the
   bar moved. Then raise the issue's "Done when" list to match, so the next contributor sees the bar before they
   start, and drop `good first issue` if the work outgrew it.
4. **Offer to split** a large ask into this pull request and a follow-up issue.

## 3. Review

- Read every line, then the code around it. Most real findings are in the lines the diff did not touch.
- Check the method's guarantees in [CONTRIBUTING.md](CONTRIBUTING.md#the-methods-guarantees) on every change that
  touches them: leakage safety, task-grouped evaluation, the sandbox, saved heads that keep loading.
- Check the pull request against its issue's "Done when" list, item by item.
- **Retest before you merge.** Check out the head in a separate worktree, run `make pr-ready`, and confirm the tests
  the pull request adds fail on main. Run the reproduction or the measurement the description reports, at least a
  small subset, and expect it within the reported spread.
- Check every file whose content you cannot judge at a glance: a lock file should change only what `pyproject.toml`
  asked for, and a regenerated golden fixture only what the change explains.
- Write the review so it can be acted on without a reply: what is wrong, where, and the change that fixes it, with
  the code when it is short.

## 4. Merge

- Every required check passes (the list is in [CONTRIBUTING.md](CONTRIBUTING.md#the-bar), under "Required to
  merge"). A failing optional check is read, not ignored: rerun it, and if it comes from main rather than the pull
  request, open an issue for it.
- Review threads are resolved, and the description still matches the diff.
- Squash-merge with a subject that says what changed, and a body whose closing line ("Closes #N") closes the issue.
  The ruleset lets an administrator bypass it; never use that to merge a contribution past a red check.
- Thank the contributor in the thread, and point to a next issue that builds on what they just did.
- Watch main's CI on the merge commit.

## 5. Credit

- Credit follows merged work: the commit's author, and the release notes, which name each contributor and the pull
  request they landed.
- Never trade credit, mentions or a merge for stars, follows or posts. GitHub's Acceptable Use Policies (section 4)
  prohibit engagement incentivized by rewards, and contributors' agents act on what a thread asks.

## 6. Write issues that can be executed

A contributor, or the agent working for them, should be able to go from the issue to a mergeable pull request
without asking a question. Every issue has:

- **Why**: the problem, and what changes when it is solved.
- **What exists**: the files, functions and line numbers involved, checked against main.
- **What to do**: numbered steps. The first is "claim and design", with the questions the design must answer. Every
  command shown was run (with `--dry-run` where one exists).
- **Done when**: a checklist a reviewer can verify, including the evidence the pull request must show and the
  reproduction a maintainer will rerun.
- **Start here**: the two to four files to read first.
- **The claim line**: "Comment 'I'm taking this' to claim it. Open a draft pull request within two weeks of the go,
  or the claim lapses."
- **Next to it**: one related issue, so finishing one leads to the next.

## 7. Talking to contributors' agents

Many contributors work through an AI agent that reads the thread. Write for the person, in visible text: an agent
acts on clear steps and exact commands anyway. Never put instructions aimed at an agent in hidden comments or
formatting; that is prompt injection against the contributor.

## 8. Releases

Release when something a user can see has landed and someone is waiting for it, or a contributor's first change is
in. Otherwise batch, about once a month. A release is a version tag ([docs/releasing.md](docs/releasing.md)): it
publishes to PyPI and archives the release on Zenodo with its DOI.
