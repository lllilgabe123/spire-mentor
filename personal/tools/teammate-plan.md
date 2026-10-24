# Plan: make Spire Mentor's fight advice teammate-aware in co-op

*Client-side only: every step reads your own game and changes only what the AI on your PC believes. Nothing is sent to other players.*

## What the research found
1. **Enemy attacks hit everyone.** In the game code, an enemy attack with no explicit target (`AttackCommand.FromMonster`) defaults to `TargetingAllOpponents`, and 211 enemy attacks use that default. So in co-op both players take each hit, and the AI is mostly *right* about the damage coming at you.
2. **Enemies have co-op HP.** Spire Mentor sends the real (co-op-scaled) enemy HP, so the AI sees the bigger health bars.
3. **The AI doesn't know your teammate exists.** Spire Mentor sends nothing about them, and the engine's `allies` slot (already referenced in 30 places) is unused in the fight loop. So in its simulations only *your* cards damage the enemies.
4. **Result:** the AI thinks co-op fights last much longer than they do, and you take hits for every one of those extra turns. That explains the tracker's numbers on your A9 co-op run: 47% average win estimate against 8/8 fights won, and 38 HP expected lost against 17 actual.

## Step 1: teammate damage as an estimate (recommended first; about a day of work)
- **Mod:** read the fight history in your own game and work out how much damage your teammate has dealt per turn so far this fight. Send that one number with each advice request. It's read-only.
- **Engine:** at the end of each simulated player turn, apply that much damage to the enemies (to the most-damaged enemy first, like a focused teammate), the same way a poison tick would.
- **Check:** the engine's tests must still pass (solo fights send 0, so nothing changes there). The co-op test harness runs with the AI-following bot, and the tracker should show the co-op win estimate moving toward reality.
- **Limits:** this doesn't cover teammate block, buffs and debuffs (Vulnerable they apply, and so on), cards that target allies, or a teammate dying.

## Step 2: debuffs your teammate puts on enemies (small)
Vulnerable, Weak and similar debuffs your teammate applies are already part of the enemy state Spire Mentor sends, so the AI already sees them. This only needs checking with a test.

## Step 3: full teammate (several days, only if the tracker still shows a big gap)
Send the teammate's HP, block, powers and hand size. Fill the engine's `allies` slot. Simulate their turn with a simple policy (or the run AI). Handle ally-target cards (AnyAlly), and teammate death. Every piece needs testing against the real game, the way the author tested single-player.

## How we'll know it worked
Rerun `tracker.py` after a few co-op runs with each step. The co-op "AI's average win estimate" should move toward the real win rate, and "HP lost: predicted vs actual" should close in on the solo gap.
