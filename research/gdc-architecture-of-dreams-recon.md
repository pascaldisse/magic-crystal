# GDC "The Architecture of 'Dreams'" (Liam de Valmency) — UNAVAILABLE, PARKED (sonnet, 07-16)

Session page: https://schedule.gdconf.com/session/the-architecture-of-dreams/868814
(GDC 2020, Programming track) · retrieved 2026-07-16 · GDC Vault search:
https://www.gdcvault.com (queried, zero hits for this title) · Wayback
snapshot of the schedule page (2019-12-16) confirms the LISTING but the
session detail is JS/API-rendered, not present in static HTML capture.

## Status: talk content UNVERIFIED — likely never publicly released
- GDC 2020 (Mar 16–20, 2020) was CANCELLED as an in-person event (COVID-19,
  announced ~2 weeks before) → normal outcome for indoor sessions: no
  recording exists unless separately produced for "GDC 2020 digital" (a
  small subset of talks were re-recorded/streamed later that year). This
  session does not appear in GDC's public YouTube channel, GDC Vault
  (checked directly + Brave-search site:gdcvault.com — no result), or any
  indexed transcript/summary site (checked Brave search across multiple
  phrasings — zero technical-content hits, only the announcement blurb).
- ⇒ NOTHING beyond the abstract below is confirmable. Do not invent
  architecture details attributed to this talk.

## The only real text: session abstract (via gamedeveloper.com announcement, 2023 archive of 2019 GDC preview post)
> "a tour through the current Dreams code base, highlighting useful code
> patterns, tricks, and systems that have allowed the game to be shipped
> in a stable form, while still allowing for flexibility and iteration in
> response to the changing needs of Dreams players... Marvel at previous
> iterations of the game's code... how the code has been designed to
> support a broad range of user-generated content..."
Speaker: Liam de Valmency, Senior Principal Programmer, Media Molecule.
Source: https://www.gamedeveloper.com/programming/see-how-media-molecule-architected-the-code-i-dreams-i-are-made-of-at-gdc-

⇒ Confirms ONLY: talk was about code patterns/stability/iteration
supporting UGC breadth — no edit/play unification specifics, no CSG
evaluation pipeline detail, no scheduling numbers. All three of the
requested topics (edit/play unification, CSG eval pipeline, scheduling)
are UNVERIFIED from this source.

## What IS independently verifiable (different sources, kept separate from the GDC talk's authority)
- dreams-recon.md (this repo, prior pass) already covers Evans SIGGRAPH
  2015 CSG/evaluator pipeline in detail — see
  evans-siggraph2015-numbers.md (this pass) for the numeric version; that
  IS a confirmed primary source, unlike this GDC talk.
- No confirmed public source (interview, patent, or talk) was found in
  this pass describing whether Dreams unifies edit-mode and play-mode
  under ONE simulation loop at the engine level — Media Molecule
  marketing language ("Imp possesses puppets, which is also playtesting")
  implies tight edit/play coupling at the UX layer, but that is a UX
  claim, not confirmed engine architecture. Mark UNVERIFIED, do not equate
  the two.

## Next-pass leads (unexplored this pass, may recover more)
- GDC Vault membership access (paid) may hold the actual recording under a
  different internal ID than the public schedule slug — not attempted
  (no credentials available in this environment).
- de Valmency's personal talks/blog/Twitter (@ handle not confirmed) may
  contain slide reposts — not searched exhaustively this pass.
- Dreams patents (Sony/Media Molecule, USPTO) may describe the edit/play
  data model formally — not searched this pass, flagged for next pass.
