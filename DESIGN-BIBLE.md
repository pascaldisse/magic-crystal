# THE DESIGN BIBLE — the Laws of Realms (Pascal, 07-16)

Not the engine's law — the law of WORLDS. These are the rules within
which realms may be created in the Forge. Everything else is HERESY.
Verbatim below; the engine grounding (heresy-lints) follows.

"The world is not a level. It's a living system."

## Core Commandments of Design
Non-negotiables — the sacred laws by which all game elements shall be
judged. Break them at your peril.

I. Thou shalt not place invisible walls
Why: Breaks immersion, punishes curiosity. Bad: Assassin's Creed —
rooftop chases blocked by invisible boundaries. Good: Elden Ring — if
it looks climbable or reachable, it probably is.

II. Thou shalt not kill players with arbitrary kill boxes
Why: Death should have cause and logic. Bad: Final Fantasy XIII —
falling slightly off-path results in death. Good: Dark Souls — falling
is dangerous, but predictable and fair.

III. Thou shalt banish the yellow paint
Why: Environmental language should be subtle and intelligent. Bad:
Uncharted — climbable ledges marked with yellow paint. Good: Breath of
the Wild — climbable surfaces are intuitive without visual hand-holding.

IV. Thou shalt offer player-driven, non-branching choices
Why: "Dynamic" doesn't mean "dialogue tree." Bad: Mass Effect —
illusion of choice, outcomes converge. Good: Deus Ex: Mankind Divided —
multiple solutions emerge from core systems.

V. Thou shalt not lock gameplay features to specific missions
Why: Temporary mechanics break consistency. Bad: GTA V — train drivable
in only one mission. Good: Minecraft — systems and tools persistently
available.

VI. Thou shalt create dynamic, system-driven climbing
Why: Repetition kills wonder. Bad: Tomb Raider Reboot — fixed scripted
routes. Good: Breath of the Wild — any surface climbable with stamina.

VII. Thou shalt let players climb fences
Why: If the player wants to go over it, don't punish curiosity. Bad:
Watch Dogs — low fences block traversal. Good: Metal Gear Solid V —
most barriers can be climbed, crossed, or circumvented.

VIII. Thou shalt not use quest markers
Why: Encourages map-gazing instead of world engagement. Bad: Skyrim —
quest compass trivializes exploration. Good: Morrowind — navigation by
in-world dialogue and direction.

IX. Thou shalt not include quick-time events
Why: Illusion of control. Let players play or just show a cutscene.
Bad: Resident Evil 6. Good: Half-Life 2 — seamless narrative, no QTEs.

X. Thou shalt abolish "detective mode"
Why: Undermines visual and environmental design. Bad: Batman: Arkham —
the game lived in x-ray vision. Good: Outer Wilds — investigation via
real observation and logic.

## Design Ethos
- REAL PUZZLES ONLY: spatial reasoning + experimentation, never
  arbitrary/mechanical. Good: The Witness. Bad: generic Ubisoft towers.
- NPCS SHOULD MATTER: react, remember, inform. Good: Pathologic 2.
  Bad: Skyrim's generic lines.
- PROPER LIP SYNC: facial animation must carry emotional delivery.
- NO FETCH QUESTS / BORING COLLECTIBLES: every side thing serves story,
  mechanics, or character. Good: Hollow Knight. Bad: Far Cry.
- INTERACTIVE ENVIRONMENTS: worlds respond; players experiment, modify,
  affect. Good: Divinity OS2 — every system interacts with every other.
  Bad: Cyberpunk 2077 — detail without interactivity.
- NO ESCORT MISSIONS: companions competent and autonomous. Good: The
  Last of Us — Ellie needs no protection.
- ALWAYS INCLUDE CHEAT CODES: fun, creativity, replayability. Good:
  DOOM, GTA San Andreas. Bad: locked modern AAA.
- NO UNSKIPPABLE INTROS: into gameplay fast, or skippable. Good:
  Half-Life 2. Bad: RDR2's forced prologue.

## Open World Principles
- IF YOU CAN SEE IT, YOU CAN GO THERE. Terrain is a challenge, not a
  lie. Good: BotW, Elden Ring. Bad: Cyberpunk's unreachable skyline.
- EVERY NPC SHOULD BE KILLABLE. Consequence-driven freedom. Good:
  Fallout New Vegas. Bad: Skyrim's immortal "essential" NPCs.
- GOOD ENDINGS: infinite map or sandbox closure — freedom and tools,
  not finality. Good: Minecraft, San Andreas. Bad: GTA V post-story,
  RDR2's linear closure.

## Core Gameplay Loop Template
EXPLORE → EXPERIMENT → ENGAGE → EXPRESS — these verbs should apply to
any major mechanic.

---

# ENGINE GROUNDING — heresy is LINTABLE

The Convictions organ (world-lint, RAIN.md) grows a HERESY BOOK:
commandments enforced as exact lints over world data, not taste.

| commandment | heresy-lint |
|---|---|
| I. invisible walls | collider volume with no visible geometry within its bounds → HERESY flag |
| II. kill boxes | damage/void volume with no visible cause (no lava essence, no fall) → flag |
| III. yellow paint | no lint needed — affordance comes from geometry truth; the vocabulary simply lacks a "climb marker" component |
| V. mission-locked features | component grants scoped to a single quest state → flag for review |
| VI/VII. systemic traversal | climbing = solver + stamina, a property of MATTER not of marked meshes; fences are just geometry |
| VIII. quest markers | the schema has no marker/compass component; direction lives in dialogue data |
| X. detective mode | the player never gets the Matrix vision — that organ is for agents/debugging (RAIN.md law: seeing is a debugging organ); players investigate with eyes |
| killable NPCs | there IS no essential/immortal flag in the schema — consequence handled by world logic, not invulnerability |
| cheat codes | the incantation console IS the cheat surface, always available — creator power is a right (the Vow extends to players) |
| interactive env | the Athanor mandate: every essence interacts through the quality square — Divinity is the floor |
| unskippable intros | realms boot into play; cutscene vessels carry a skip sigil by default |

Realm ordeals: NARUKO and every future realm pass the Heresy Book
before any rite is called complete.
